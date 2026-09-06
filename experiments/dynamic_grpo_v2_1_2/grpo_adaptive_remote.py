#!/usr/bin/env python3
"""Launch the existing GRPO pipeline with the adaptive two-level trainer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

from accelerate import PartialState

import grpo_remote as base

from adaptive_grpo_core import RewardConfig, dataset_is_update_aligned, make_memory_safe_layout
from adaptive_trainer import AdaptiveTrainerConfig, make_adaptive_grpo_trainer


DDP_TIMEOUT_SECONDS = 3600


def initialize_distributed_context_with_timeout():  # type: ignore[no-untyped-def]
    """Initialize the process group with the adaptive run watchdog timeout."""

    state = PartialState(timeout=timedelta(seconds=DDP_TIMEOUT_SECONDS))
    context = base.DistributedContext(
        state=state,
        world_size=int(state.num_processes),
        rank=int(state.process_index),
        local_rank=int(state.local_process_index),
        is_main_process=bool(state.is_main_process),
    )
    if not base.torch.cuda.is_available():
        raise RuntimeError("QLoRA GRPO requires CUDA on every Accelerate process")
    visible_devices = base.torch.cuda.device_count()
    if context.local_rank < 0 or context.local_rank >= visible_devices:
        raise RuntimeError(
            "LOCAL_RANK is outside the visible CUDA device set: "
            f"local_rank={context.local_rank}, visible_devices={visible_devices}"
        )
    return context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dynamic_cycle_contract(
    *,
    manifest_path: Path,
    train_path: Path,
    expected_cycle_id: str,
    expected_old_policy_sha256: str,
) -> dict:
    """Bind one training invocation to exactly one frozen rollout cycle."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("cycle_identity", {})
    checks = {
        "schema": (manifest.get("schema"), "dynamic_grpo_cycle_schedule_v2"),
        "cycle_id": (identity.get("cycle_id"), expected_cycle_id),
        "old_policy_sha256": (
            identity.get("old_policy_sha256"),
            expected_old_policy_sha256,
        ),
        "train_schedule_sha256": (
            manifest.get("train_schedule_sha256"),
            _sha256(train_path),
        ),
        "reward_source": (
            manifest.get("reward_source"),
            "fresh_training_rollout_current_cycle_only",
        ),
        "screen_receipts_enter_reward": (
            manifest.get("screen_receipts_enter_reward"),
            False,
        ),
        "history_enters_reward": (manifest.get("history_enters_reward"), False),
        "cross_cycle_receipts_allowed": (
            manifest.get("cross_cycle_receipts_allowed"),
            False,
        ),
        "strategy_drift_scaling": (
            manifest.get("strategy_drift_scaling"),
            False,
        ),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    rows = [json.loads(line) for line in train_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 256:
        mismatches.append(f"training rows={len(rows)}, expected 256")
    problem_ids: set[str] = set()
    for index, row in enumerate(rows):
        problem_id = str(row.get("id", row.get("problem_id", "")))
        if not problem_id or problem_id in problem_ids:
            mismatches.append(f"row {index} has missing/duplicate problem ID {problem_id!r}")
            continue
        problem_ids.add(problem_id)
        meta = row.get("dynamic_cycle", {})
        if int(row.get("repeat", -1)) != 1:
            mismatches.append(f"{problem_id}: repeat must be exactly one")
        if meta.get("cycle_id") != expected_cycle_id:
            mismatches.append(f"{problem_id}: row cycle ID mismatch")
        if meta.get("old_policy_sha256") != expected_old_policy_sha256:
            mismatches.append(f"{problem_id}: row old-policy hash mismatch")
        if meta.get("selection_manifest_sha256") != identity.get(
            "selection_manifest_sha256"
        ):
            mismatches.append(f"{problem_id}: row selection hash mismatch")
        if meta.get("reward_source") != "fresh_training_rollout_current_cycle_only":
            mismatches.append(f"{problem_id}: stale/screen reward source requested")
    if mismatches:
        raise ValueError("Invalid dynamic cycle training contract: " + "; ".join(mismatches[:20]))
    return manifest


def configure_persistent_pantograph_service() -> None:
    """Attach every DDP rank to its allocation-lifetime warm worker."""

    if not os.environ.get("PANTOGRAPH_POOL_SOCKET_DIR"):
        return
    from persistent_pantograph_service import PersistentVerificationPool

    os.environ["PANTOGRAPH_POOL_CLIENT_MODE"] = "rank"
    base.VerificationPool = PersistentVerificationPool


def parse_adaptive_args() -> tuple[argparse.Namespace, base.GRPOTrainConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adaptive_problems_per_update", type=int, default=16)
    parser.add_argument("--adaptive_scale_ema_decay", type=float, default=0.90)
    parser.add_argument("--adaptive_initial_scale", type=float, default=0.50)
    parser.add_argument("--adaptive_trajectory_rank_beta", type=float, default=0.125)
    parser.add_argument("--adaptive_cycle_schedule_manifest", type=Path)
    parser.add_argument("--adaptive_cycle_id")
    parser.add_argument("--adaptive_old_policy_sha256")
    parser.add_argument("--adaptive_cycle_stop_global_step", type=int)
    adaptive, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        ordinary = base.parse_args()
    finally:
        sys.argv = original_argv
    return adaptive, ordinary


def main() -> None:
    configure_persistent_pantograph_service()
    adaptive_args, config = parse_adaptive_args()
    # The process group is created here, before GRPOConfig exists. Install the
    # timeout-aware constructor before the first PartialState initialization.
    base.initialize_distributed_context = initialize_distributed_context_with_timeout
    distributed = base.initialize_distributed_context()
    layout = make_memory_safe_layout(
        world_size=distributed.world_size,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        problems_per_update=adaptive_args.adaptive_problems_per_update,
        num_iterations=config.num_iterations,
    )
    dynamic_manifest = None
    dynamic_fields = (
        adaptive_args.adaptive_cycle_schedule_manifest,
        adaptive_args.adaptive_cycle_id,
        adaptive_args.adaptive_old_policy_sha256,
    )
    if any(value is not None for value in dynamic_fields):
        if any(value is None for value in dynamic_fields):
            raise ValueError(
                "dynamic cycle manifest, cycle ID, and old-policy hash are all required"
            )
        dynamic_manifest = validate_dynamic_cycle_contract(
            manifest_path=adaptive_args.adaptive_cycle_schedule_manifest,
            train_path=Path(config.train_file),
            expected_cycle_id=adaptive_args.adaptive_cycle_id,
            expected_old_policy_sha256=adaptive_args.adaptive_old_policy_sha256,
        )
        if adaptive_args.adaptive_cycle_stop_global_step is None:
            raise ValueError("dynamic training requires an explicit cycle stop step")
    elif adaptive_args.adaptive_cycle_stop_global_step is not None:
        raise ValueError("cycle stop step is only valid with a frozen dynamic cycle")
    # Account for every replayed microstep so one optimizer update still spans
    # the requested number of distinct problem groups.
    config = replace(
        config,
        gradient_accumulation_steps=layout.gradient_accumulation_steps,
    )
    reward_kwargs = {}
    if dynamic_manifest is not None:
        dynamic_config = dynamic_manifest["config"]
        if dynamic_config.get("strategy_drift_scaling") is not False:
            raise ValueError("strategy-drift scaling must remain disabled")
        reward_kwargs = {
            "weight_success_1": float(dynamic_config["weight_success_1"]),
            "weight_success_2": float(dynamic_config["weight_success_2"]),
            "weight_success_3": float(dynamic_config["weight_success_3"]),
            "weight_success_4": float(dynamic_config["weight_success_4"]),
        }
    reward_config = RewardConfig(
        num_generations=config.num_generations,
        trajectory_rank_beta=adaptive_args.adaptive_trajectory_rank_beta,
        **reward_kwargs,
    )
    trainer_config = AdaptiveTrainerConfig(
        problems_per_update=layout.problems_per_update,
        scale_ema_decay=adaptive_args.adaptive_scale_ema_decay,
        initial_advantage_scale=adaptive_args.adaptive_initial_scale,
        num_iterations=layout.num_iterations,
        cycle_stop_global_step=adaptive_args.adaptive_cycle_stop_global_step,
        rollout_cycle_id=(
            dynamic_manifest["cycle_identity"]["cycle_id"]
            if dynamic_manifest is not None else None
        ),
        screen_old_policy_sha256=(
            dynamic_manifest["cycle_identity"]["old_policy_sha256"]
            if dynamic_manifest is not None else None
        ),
        selection_manifest_sha256=(
            dynamic_manifest["cycle_identity"]["selection_manifest_sha256"]
            if dynamic_manifest is not None else None
        ),
        reward=reward_config,
    )

    original_supported_kwargs = base.supported_config_kwargs
    original_trl_support = base.validate_trl_grpo_support
    original_adapter_validation = base.validate_epoch2_adapter

    def validate_initial_or_resume_adapter(config):  # type: ignore[no-untyped-def]
        """Keep the SFT gate for cycle 0 and validate GRPO checkpoints on resume."""

        if not config.resume_from_checkpoint:
            return original_adapter_validation(config)
        adapter_path = Path(config.adapter_path).expanduser().resolve()
        resume_path = Path(config.resume_from_checkpoint).expanduser().resolve()
        output_path = Path(config.output_dir).expanduser().resolve()
        if adapter_path != resume_path:
            raise ValueError(
                f"resume adapter differs from checkpoint: {adapter_path} != {resume_path}"
            )
        if resume_path.parent != output_path:
            raise ValueError(
                f"resume checkpoint must be directly under output_dir: {resume_path}"
            )
        required = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        )
        missing = [name for name in required if not (resume_path / name).is_file()]
        if missing:
            raise ValueError(f"resume checkpoint is incomplete: missing {missing}")
        rng_states = sorted(resume_path.glob("rng_state_*.pth"))
        if len(rng_states) != 4:
            raise ValueError(
                f"resume checkpoint requires four rank RNG states, found {len(rng_states)}"
            )
        state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
        global_step = int(state.get("global_step", -1))
        expected_name = f"checkpoint-{global_step}"
        if global_step <= 0 or resume_path.name != expected_name:
            raise ValueError(
                f"invalid resume checkpoint identity: {resume_path.name}, step={global_step}"
            )
        if global_step >= trainer_config.cycle_stop_global_step:
            raise ValueError(
                f"resume step {global_step} must precede cycle stop "
                f"{trainer_config.cycle_stop_global_step}"
            )
        checkpoint_epoch = float(state.get("epoch", -1))
        if checkpoint_epoch < 0:
            raise ValueError("resume checkpoint has no valid trainer epoch")
        # Reuse the original adapter/base/LoRA checks.  Only its SFT-epoch
        # assertion is replaced because this is now a GRPO trainer checkpoint.
        peft_config = original_adapter_validation(
            replace(config, expected_sft_epoch=checkpoint_epoch)
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "ADAPTIVE_RESUME_ADAPTER_VALIDATION "
                + json.dumps(
                    {
                        "status": "PASS",
                        "checkpoint": str(resume_path),
                        "global_step": global_step,
                        "checkpoint_epoch": checkpoint_epoch,
                        "cycle_stop_global_step": trainer_config.cycle_stop_global_step,
                        "optimizer_state_present": True,
                        "scheduler_state_present": True,
                        "rng_state_count": len(rng_states),
                    },
                    sort_keys=True,
                )
            )
        return peft_config

    def validate_problem_aligned_batch(*, ticket_count, config, distributed):  # type: ignore[no-untyped-def]
        del config, distributed
        if not dataset_is_update_aligned(ticket_count, layout):
            raise ValueError(
                f"{ticket_count} prompt tickets are not divisible by "
                f"{layout.prompt_rows_per_update} problems/update"
            )
        return (
            layout.world_size
            * layout.per_device_train_batch_size
            * layout.gradient_accumulation_steps
        )

    def adaptive_supported_kwargs(config_cls, values):  # type: ignore[no-untyped-def]
        if config_cls.__name__ == "GRPOConfig":
            values = dict(values)
            values.update(
                {
                    "generation_batch_size": layout.generation_batch_size,
                    "gradient_accumulation_steps": layout.gradient_accumulation_steps,
                    "num_iterations": layout.num_iterations,
                    "scale_rewards": "none",
                    "loss_type": "dapo",
                    "epsilon": 0.2,
                    "epsilon_high": 0.2,
                    "delta": None,
                    # Every dynamic cycle presents a newly frozen schedule.
                    # Resume optimizer/scheduler/RNG state, but never skip rows
                    # based on the previous cycle's dataloader position.
                    "ignore_data_skip": dynamic_manifest is not None,
                    # Lean reward compilation can make ranks arrive at a
                    # collective more than ten minutes apart.  Keep the
                    # distributed watchdog above that verified worst case;
                    # this changes only failure tolerance, not optimization.
                    "ddp_timeout": DDP_TIMEOUT_SECONDS,
                }
            )
        return original_supported_kwargs(config_cls, values)

    GRPOConfig, GRPOTrainer = original_trl_support()
    AdaptiveTrainer = make_adaptive_grpo_trainer(GRPOTrainer, trainer_config)

    def adaptive_trl_support():  # type: ignore[no-untyped-def]
        return GRPOConfig, AdaptiveTrainer

    def adaptive_startup_report(
        config,
        *,
        physical_train_rows,
        train_dataset,
        validation_dataset,
        distributed,
        effective_global_batch_size,
    ):  # type: ignore[no-untyped-def]
        del effective_global_batch_size
        payload = {
            "schema": "adaptive_two_level_grpo_startup_v1",
            "trl_version": base.package_version("trl"),
            "transformers_version": base.package_version("transformers"),
            "torch_version": base.torch.__version__,
            "physical_train_rows": physical_train_rows,
            "repeat_expanded_train_tickets": len(train_dataset),
            "validation_samples": len(validation_dataset) if validation_dataset else 0,
            "num_generations": layout.num_generations,
            "generation_batch_size": layout.generation_batch_size,
            "unique_prompts_per_generation_batch": 1,
            "problems_per_optimizer_update": layout.problems_per_update,
            "gradient_accumulation_steps": layout.gradient_accumulation_steps,
            "expected_generation_batches_per_epoch": len(train_dataset),
            "expected_optimizer_steps_per_epoch": len(train_dataset)
            // layout.problems_per_update,
            "num_iterations": layout.num_iterations,
            "scale_rewards": "none_plus_lagged_centered_rms_ema",
            "loss_type": "dapo",
            "clip_higher": False,
            "distributed": distributed.as_dict(),
            "reward_backend": "persistent_pantograph_pool_binary_plus_two_level_shaping",
            "rollout_backend": "vllm",
            "sft_adapter_path": config.adapter_path,
            "dynamic_cycle": (
                dynamic_manifest["cycle_identity"] if dynamic_manifest is not None else None
            ),
            "screen_receipts_enter_reward": False,
            "strategy_drift_scaling": False,
            "ddp_timeout_seconds": DDP_TIMEOUT_SECONDS,
        }
        print("ADAPTIVE_GRPO_STARTUP " + json.dumps(payload, ensure_ascii=False))

    base.validate_ddp_ticket_divisibility = validate_problem_aligned_batch
    base.validate_epoch2_adapter = validate_initial_or_resume_adapter
    base.supported_config_kwargs = adaptive_supported_kwargs
    base.validate_trl_grpo_support = adaptive_trl_support
    base.print_startup_report = adaptive_startup_report
    base.patch_vllm_none_vocab_tokens()
    base.save_training_config(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if distributed.rank == 0:
        source_hashes = {}
        for filename in (
            "grpo_remote.py",
            "grpo_adaptive_remote.py",
            "adaptive_trainer.py",
            "adaptive_grpo_core.py",
        ):
            path = Path(__file__).resolve().parent / filename
            source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "adaptive_run_contract.json").write_text(
            json.dumps({
                "schema": "adaptive_two_level_grpo_contract_v1",
                "layout": asdict(layout),
                "trainer": {
                    **asdict(trainer_config),
                    "reward": asdict(reward_config),
                },
                "invariants": {
                    "clip_higher": False,
                    "group_centering": True,
                    "group_standard_deviation": False,
                    "lagged_global_ema_scale": True,
                    "problem_balanced_outer_loss": True,
                    "pass32_gradient_multiplier": 1,
                    "generation_peak_completions": layout.generation_batch_size,
                    "screen_receipts_enter_reward": False,
                    "history_enters_reward": False,
                    "cross_cycle_receipts_allowed": False,
                    "strategy_drift_scaling": False,
                },
                "dynamic_cycle": dynamic_manifest,
                "source_sha256": source_hashes,
            }, ensure_ascii=False, indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    trainer, tokenizer, reward_func = base.build_trainer(config)
    if not isinstance(trainer, AdaptiveTrainer):
        reward_func.close()
        raise RuntimeError("adaptive trainer substitution failed")
    try:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trainer.save_model(config.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(config.output_dir)
        trainer.save_state()
        base.save_training_config(config)
    finally:
        reward_func.close()
        if base.torch.distributed.is_available() and base.torch.distributed.is_initialized():
            base.torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
