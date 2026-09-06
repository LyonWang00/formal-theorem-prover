"""TRL adapter implementing two-level rewards and lagged global scaling.

The adapter targets the project's pinned TRL 1.12.0 runtime.  It intentionally
keeps one pass@8 group in GPU generation memory and accumulates several groups
before one optimizer update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from adaptive_grpo_core import LaggedWindowScale, RewardConfig, shape_group_rewards


@dataclass(frozen=True)
class AdaptiveTrainerConfig:
    problems_per_update: int = 16
    scale_ema_decay: float = 0.90
    initial_advantage_scale: float = 0.50
    num_iterations: int = 1
    cycle_stop_global_step: int | None = None
    rollout_cycle_id: str | None = None
    screen_old_policy_sha256: str | None = None
    selection_manifest_sha256: str | None = None
    reward: RewardConfig = RewardConfig()
    state_filename: str = "adaptive_grpo_state.json"


def make_adaptive_grpo_trainer(
    base_cls: type, default_config: AdaptiveTrainerConfig | None = None
) -> type:
    """Return a GRPOTrainer subclass without importing TRL in this module."""

    class AdaptiveTwoLevelGRPOTrainer(base_cls):
        def __init__(
            self,
            *args: Any,
            adaptive_config: AdaptiveTrainerConfig | None = None,
            **kwargs: Any,
        ) -> None:
            adaptive_config = adaptive_config or default_config
            if adaptive_config is None:
                raise ValueError("adaptive_config is required")
            self.adaptive_config = adaptive_config
            self.adaptive_config.reward.validate()
            self._adaptive_unconsumed_advantages: list[float] = []
            self._adaptive_scale = LaggedWindowScale(
                groups_per_window=adaptive_config.problems_per_update,
                ema_decay=adaptive_config.scale_ema_decay,
                initial_scale=adaptive_config.initial_advantage_scale,
                epsilon=adaptive_config.reward.scale_epsilon,
            )
            self._adaptive_group_log: list[dict[str, Any]] = []
            self._adaptive_groups_seen = 0
            super().__init__(*args, **kwargs)
            self._validate_adaptive_contract()
            if adaptive_config.cycle_stop_global_step is not None:
                from transformers import TrainerCallback

                target_step = int(adaptive_config.cycle_stop_global_step)
                if target_step <= 0:
                    raise ValueError("cycle_stop_global_step must be positive")

                class StopAtFrozenCycleBoundary(TrainerCallback):
                    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
                        del args, kwargs
                        if int(state.global_step) >= target_step:
                            control.should_training_stop = True
                        return control

                self.add_callback(StopAtFrozenCycleBoundary())

        def _validate_adaptive_contract(self) -> None:
            args = self.args
            expected_g = self.adaptive_config.reward.num_generations
            checks = {
                "num_generations": (getattr(args, "num_generations", None), expected_g),
                "generation_batch_size": (
                    getattr(args, "generation_batch_size", None), expected_g
                ),
                "num_iterations": (
                    getattr(args, "num_iterations", None),
                    self.adaptive_config.num_iterations,
                ),
                "scale_rewards": (getattr(args, "scale_rewards", None), "none"),
                "loss_type": (getattr(args, "loss_type", None), "dapo"),
            }
            mismatches = [
                f"{name}={actual!r} (expected {expected!r})"
                for name, (actual, expected) in checks.items()
                if actual != expected
            ]
            epsilon = getattr(args, "epsilon", None)
            epsilon_high = getattr(args, "epsilon_high", epsilon)
            if epsilon is not None and epsilon_high not in (None, epsilon):
                mismatches.append(
                    f"epsilon_high={epsilon_high!r} differs from epsilon={epsilon!r}; "
                    "Clip-Higher is disabled in this contract"
                )
            if getattr(args, "delta", None) is not None:
                mismatches.append("delta must be None; two-sided Clip-Higher is disabled")
            if mismatches:
                raise ValueError("Invalid adaptive GRPO contract: " + "; ".join(mismatches))
            lineage = (
                self.adaptive_config.rollout_cycle_id,
                self.adaptive_config.screen_old_policy_sha256,
                self.adaptive_config.selection_manifest_sha256,
            )
            if any(value is not None for value in lineage) and any(
                value is None for value in lineage
            ):
                raise ValueError("dynamic cycle lineage fields must be supplied together")
            target = self.adaptive_config.cycle_stop_global_step
            if target is not None and int(getattr(args, "max_steps", -1)) < int(target):
                raise ValueError(
                    "cycle stop must not exceed the full-run max_steps scheduler horizon"
                )

        def _generate_and_score_completions(self, inputs):  # type: ignore[no-untyped-def]
            import torch

            output = super()._generate_and_score_completions(inputs)
            if not self.model.training:
                # Validation must not alter the normalization state used by the
                # next optimizer window.
                return output
            group_size = self.adaptive_config.reward.num_generations
            binary_centered = output["advantages"].detach().float()
            # With binary 0/1 rewards and no scaling, positive centered values
            # are precisely successful completions for every effective group.
            # All-zero groups (0/G or G/G) are deliberately inactive either way.
            binary_local = (binary_centered > 0).float()
            sampling_logps = output.get("sampling_per_token_logps")
            if sampling_logps is None:
                sampling_logps = output.get("old_per_token_logps")
            if sampling_logps is None:
                raise RuntimeError(
                    "TRL did not return sampling/old-policy log probabilities; "
                    "trajectory-layer shaping cannot be computed safely"
                )
            mask = output["completion_mask"].detach().float()
            sampling_logps = sampling_logps.detach().float()
            valid = mask * torch.isfinite(sampling_logps).float()
            valid_counts = valid.sum(dim=1)
            sequence_means = (torch.nan_to_num(sampling_logps) * valid).sum(
                dim=1
            ) / valid_counts.clamp(min=1.0)
            sequence_means = sequence_means.masked_fill(valid_counts == 0, float("nan"))
            global_binary = self.accelerator.gather(binary_local).cpu().tolist()
            global_means = self.accelerator.gather(sequence_means).cpu().tolist()
            if len(global_binary) != len(global_means) or len(global_binary) % group_size:
                raise RuntimeError("distributed reward/log-prob groups are misaligned")

            global_centered: list[float] = []
            new_records: list[dict[str, Any]] = []
            for offset in range(0, len(global_binary), group_size):
                group = shape_group_rewards(
                    global_binary[offset : offset + group_size],
                    global_means[offset : offset + group_size],
                    self.adaptive_config.reward,
                )
                global_centered.extend(group.centered_advantages)
                record = {
                        "group_index": self._adaptive_groups_seen,
                        "global_step_before_update": int(self.state.global_step),
                        "successes": group.successes,
                        "attempts": group_size,
                        "problem_weight": group.problem_weight,
                        "effective": group.effective,
                        "trajectory_factors": list(group.trajectory_factors),
                        "advantage_scale_used": self._adaptive_scale.scale,
                        "old_policy_mean_logprobs": global_means[
                            offset : offset + group_size
                        ],
                        "rollout_cycle_id": self.adaptive_config.rollout_cycle_id,
                        "screen_old_policy_sha256": self.adaptive_config.screen_old_policy_sha256,
                        "selection_manifest_sha256": self.adaptive_config.selection_manifest_sha256,
                        "reward_source": "fresh_training_rollout_current_cycle_only",
                        # Generation happened synchronously in the current
                        # _generate_and_score call, before this step updates.
                        "rollout_policy_step": int(self.state.global_step),
                    }
                self._adaptive_groups_seen += 1
                self._adaptive_group_log.append(record)
                new_records.append(record)

            local_count = len(binary_local)
            local_start = self.accelerator.process_index * local_count
            local_centered = global_centered[local_start : local_start + local_count]
            if len(local_centered) != local_count:
                raise RuntimeError("distributed adaptive advantage slice is incomplete")
            output["advantages"] = torch.tensor(
                local_centered,
                device=binary_centered.device,
                dtype=output["advantages"].dtype,
            ) / max(self._adaptive_scale.scale, self.adaptive_config.reward.scale_epsilon)

            # All problems inside this optimizer window use the old scale.  Only
            # after shaping the whole generation batch do we update next window.
            self._adaptive_unconsumed_advantages.extend(global_centered)
            while len(self._adaptive_unconsumed_advantages) >= group_size:
                centered_group = self._adaptive_unconsumed_advantages[:group_size]
                del self._adaptive_unconsumed_advantages[:group_size]
                self._adaptive_scale.observe_group(centered_group)
            self._append_group_audit(new_records)
            return output

        def _append_group_audit(self, records: list[dict[str, Any]]) -> None:
            if not records or not self.is_world_process_zero():
                return
            target = Path(self.args.output_dir) / "adaptive_groups.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        def _adaptive_state_path(self, directory: str | Path) -> Path:
            return Path(directory) / self.adaptive_config.state_filename

        def _write_adaptive_state(self, directory: str | Path) -> None:
            if not self.is_world_process_zero():
                return
            target = self._adaptive_state_path(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "adaptive_two_level_grpo_state_v1",
                "trainer_config": {
                    **asdict(self.adaptive_config),
                    "reward": asdict(self.adaptive_config.reward),
                },
                "scale": self._adaptive_scale.state_dict(),
                "unconsumed_advantages": self._adaptive_unconsumed_advantages,
                "recent_groups": self._adaptive_group_log[-256:],
                "groups_seen": self._adaptive_groups_seen,
                "global_step": int(self.state.global_step),
            }
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)

        def _load_adaptive_state(self, directory: str | Path) -> None:
            target = self._adaptive_state_path(directory)
            if not target.exists():
                raise FileNotFoundError(
                    f"{target} is missing; adaptive resume would change normalization state"
                )
            payload = json.loads(target.read_text(encoding="utf-8"))
            self._adaptive_scale.load_state_dict(payload["scale"])
            self._adaptive_unconsumed_advantages = [
                float(value) for value in payload.get("unconsumed_advantages", [])
            ]
            self._adaptive_groups_seen = int(payload.get("groups_seen", 0))

        def train(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            resume = kwargs.get("resume_from_checkpoint")
            if resume is True:
                candidates = sorted(
                    Path(self.args.output_dir).glob("checkpoint-*"),
                    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
                )
                if not candidates:
                    raise FileNotFoundError("no checkpoint exists for adaptive resume")
                resume = candidates[-1]
            if isinstance(resume, (str, Path)):
                self._load_adaptive_state(resume)
            result = super().train(*args, **kwargs)
            # A non-divisible final step (for example the approved 149-step
            # matched-rollout budget) must still have a resumable checkpoint,
            # including optimizer, scheduler, RNG, and adaptive scale state.
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            if not checkpoint.exists():
                self._save_checkpoint(self.model, kwargs.get("trial"))
            return result

        def _save_checkpoint(self, model, trial):  # type: ignore[no-untyped-def]
            result = super()._save_checkpoint(model, trial)
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            self._write_adaptive_state(checkpoint)
            return result

        def save_model(self, output_dir=None, _internal_call=False):  # type: ignore[no-untyped-def]
            result = super().save_model(output_dir, _internal_call)
            self._write_adaptive_state(output_dir or self.args.output_dir)
            return result

    AdaptiveTwoLevelGRPOTrainer.__name__ = "AdaptiveTwoLevelGRPOTrainer"
    return AdaptiveTwoLevelGRPOTrainer
