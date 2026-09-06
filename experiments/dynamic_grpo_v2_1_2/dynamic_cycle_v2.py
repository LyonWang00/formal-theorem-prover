#!/usr/bin/env python3
"""Fail-closed planning primitives for periodic adaptive GRPO cycles.

The screen/top-up receipts decide which *problems* are trained.  They are not
training rewards.  Training rewards must be produced by fresh rollout groups
tagged with the same cycle ID.  This distinction keeps the first dynamic
experiment comparable with the existing online TRL trainer and prevents a
receipt from an older policy/cycle from silently entering the loss.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


class CycleContractError(ValueError):
    """Raised when an artifact cannot be proven to belong to this cycle."""


@dataclass(frozen=True)
class DynamicCycleConfig:
    screen_problems: int = 512
    steps_per_cycle: int = 16
    problems_per_step: int = 16
    pass8_attempts: int = 8
    pass16_topup_limit: int = 128
    pass32_topup_limit: int = 64
    mastered_successes_at_pass8: int = 5
    min_effective_groups_per_step: int = 4
    effective_step_scale_cap: float = 1.5
    # More exploration-oriented than the static E2 weights (1.20/1.10/1/.90).
    weight_success_1: float = 1.30
    weight_success_2: float = 1.10
    weight_success_3: float = 0.80
    weight_success_4: float = 0.60
    trajectory_rank_beta: float = 0.125
    strategy_drift_scaling: bool = False

    @property
    def training_slots(self) -> int:
        return self.steps_per_cycle * self.problems_per_step

    def validate(self) -> None:
        if self.screen_problems < self.training_slots:
            raise CycleContractError("screen pool must cover every training slot")
        if self.pass8_attempts != 8:
            raise CycleContractError("the v2 routing contract requires pass@8")
        if self.mastered_successes_at_pass8 != 5:
            raise CycleContractError("the v2 mastered boundary is fixed at 5/8")
        if not 0 < self.min_effective_groups_per_step <= self.problems_per_step:
            raise CycleContractError("invalid minimum effective groups per step")
        if self.effective_step_scale_cap < 1.0:
            raise CycleContractError("effective scale cap must be at least one")
        if self.strategy_drift_scaling:
            raise CycleContractError(
                "strategy-drift scaling is intentionally disabled for this experiment"
            )


@dataclass(frozen=True)
class CycleIdentity:
    cycle_id: str
    old_policy_sha256: str
    selection_manifest_sha256: str

    def validate(self) -> None:
        if not self.cycle_id.strip():
            raise CycleContractError("cycle_id is empty")
        for name, value in (
            ("old_policy_sha256", self.old_policy_sha256),
            ("selection_manifest_sha256", self.selection_manifest_sha256),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise CycleContractError(f"{name} is not a lowercase SHA-256 digest")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_key(seed: int, *parts: object) -> str:
    return sha256_bytes(":".join([str(seed), *(str(part) for part in parts)]).encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CycleContractError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_id(row: dict[str, Any]) -> str:
    for field in ("problem_id", "id"):
        if field in row:
            return str(row[field])
    raise CycleContractError("row has neither problem_id nor id")


def is_success(row: dict[str, Any]) -> bool:
    value = row.get("success", row.get("reward"))
    if value is None:
        raise CycleContractError("receipt has neither success nor reward")
    return bool(value) if isinstance(value, bool) else float(value) > 0.5


def assert_receipt_identity(
    receipt: dict[str, Any], identity: CycleIdentity, *, reward_source: str
) -> None:
    identity.validate()
    expected = {
        "rollout_cycle_id": identity.cycle_id,
        "old_policy_sha256": identity.old_policy_sha256,
        "selection_manifest_sha256": identity.selection_manifest_sha256,
        "reward_source": reward_source,
    }
    mismatches = [
        f"{key}={receipt.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if receipt.get(key) != value
    ]
    if mismatches:
        raise CycleContractError("receipt lineage mismatch: " + "; ".join(mismatches))


def validate_receipt_set(
    receipts: Sequence[dict[str, Any]],
    identity: CycleIdentity,
    *,
    reward_source: str,
    allowed_attempt_indices: set[int] | None = None,
) -> None:
    seen: set[tuple[str, int]] = set()
    for receipt in receipts:
        assert_receipt_identity(receipt, identity, reward_source=reward_source)
        problem_id = row_id(receipt)
        if "attempt_index" not in receipt:
            raise CycleContractError(f"{problem_id}: missing attempt_index")
        attempt = int(receipt["attempt_index"])
        if allowed_attempt_indices is not None and attempt not in allowed_attempt_indices:
            raise CycleContractError(f"{problem_id}: unexpected attempt_index {attempt}")
        key = (problem_id, attempt)
        if key in seen:
            raise CycleContractError(f"duplicate receipt {key}")
        seen.add(key)


def difficulty_weight(
    successes: int, attempts: int, config: DynamicCycleConfig | None = None
) -> float:
    """Aggressively suppress 3/8 and 4/8 while keeping sparse successes useful.

    For top-ups, the current cumulative rate is projected onto an equivalent
    pass@8 count.  Any non-zero rate below 1/8 receives the sparse 1/8 cap.
    """

    cfg = config or DynamicCycleConfig()
    cfg.validate()
    if attempts not in {8, 16, 32} or not 0 <= successes <= attempts:
        raise CycleContractError("success count must describe pass@8/16/32")
    if successes == 0:
        return 0.0
    equivalent = 8.0 * successes / attempts
    if equivalent >= cfg.mastered_successes_at_pass8:
        return 0.0
    anchors = {
        1.0: cfg.weight_success_1,
        2.0: cfg.weight_success_2,
        3.0: cfg.weight_success_3,
        4.0: cfg.weight_success_4,
        5.0: 0.0,
    }
    if equivalent <= 1.0:
        return cfg.weight_success_1
    lower = math.floor(equivalent)
    upper = math.ceil(equivalent)
    if lower == upper:
        return anchors[float(lower)]
    fraction = equivalent - lower
    return anchors[float(lower)] * (1.0 - fraction) + anchors[float(upper)] * fraction


def topup_mass_scale(attempts: int) -> float:
    if attempts not in {8, 16, 32}:
        raise CycleContractError("attempts must be 8, 16, or 32")
    return 8.0 / attempts


def effective_step_scale(
    effective_groups: int, config: DynamicCycleConfig | None = None
) -> float:
    cfg = config or DynamicCycleConfig()
    cfg.validate()
    if not 0 <= effective_groups <= cfg.problems_per_step:
        raise CycleContractError("effective group count is outside the step")
    if effective_groups == 0:
        return 0.0
    return min(
        cfg.effective_step_scale_cap,
        cfg.problems_per_step / effective_groups,
    )


def trajectory_rank_factors(values: Sequence[float], beta: float) -> list[float]:
    if not 0.0 <= beta < 1.0:
        raise CycleContractError("trajectory beta must be in [0, 1)")
    finite = [index for index, value in enumerate(values) if math.isfinite(value)]
    result = [1.0] * len(values)
    if not finite:
        return result
    order = sorted(finite, key=lambda index: (-values[index], index))
    for rank, index in enumerate(order):
        result[index] = 1.0 - beta * (len(order) - rank) / len(order)
    return result


def shape_current_cycle_group(
    binary_rewards: Sequence[float],
    old_policy_mean_logprobs: Sequence[float],
    *,
    config: DynamicCycleConfig | None = None,
) -> dict[str, Any]:
    """Compute a current-cycle group advantage without standard-deviation scaling."""

    cfg = config or DynamicCycleConfig()
    cfg.validate()
    if len(binary_rewards) not in {8, 16, 32}:
        raise CycleContractError("reward group must contain 8, 16, or 32 attempts")
    if len(binary_rewards) != len(old_policy_mean_logprobs):
        raise CycleContractError("one old-policy score is required per trajectory")
    successes = sum(float(value) > 0.5 for value in binary_rewards)
    weight = difficulty_weight(successes, len(binary_rewards), cfg)
    mass = topup_mass_scale(len(binary_rewards))
    factors = trajectory_rank_factors(old_policy_mean_logprobs, cfg.trajectory_rank_beta)
    shaped = [
        weight * mass * factor if float(reward) > 0.5 else 0.0
        for reward, factor in zip(binary_rewards, factors)
    ]
    mean = sum(shaped) / len(shaped)
    centered = [value - mean for value in shaped]
    return {
        "successes": successes,
        "attempts": len(binary_rewards),
        "difficulty_weight": weight,
        "topup_mass_scale": mass,
        "trajectory_factors": factors,
        "centered_advantages": centered,
        "effective": 0 < successes and weight > 0.0,
    }


def classify_screen_group(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(attempts, key=lambda row: int(row["attempt_index"]))
    indices = [int(row["attempt_index"]) for row in ordered]
    if len(indices) not in {8, 16, 32} or indices != list(range(len(indices))):
        raise CycleContractError("screen/top-up attempts must be canonical indices 0..n-1")
    initial_successes = sum(is_success(row) for row in ordered[:8])
    total_successes = sum(is_success(row) for row in ordered)
    total_attempts = len(ordered)
    if initial_successes >= 5:
        route = "mastered"
    elif initial_successes > 0:
        route = "active_pass8"
    elif total_successes > 0:
        route = "active_after_topup"
    elif total_attempts == 32:
        route = "hard_bank"
    else:
        route = "unresolved_zero"
    return {
        "problem_id": row_id(ordered[0]),
        "route": route,
        "initial_successes": initial_successes,
        "total_successes": total_successes,
        "total_attempts": total_attempts,
        "screen_success_rate": total_successes / total_attempts,
        "screen_difficulty_weight": difficulty_weight(total_successes, total_attempts),
    }


def _bucket(record: dict[str, Any]) -> str:
    if record["route"] == "active_after_topup":
        return "sparse"
    successes = int(record["initial_successes"])
    if successes <= 2:
        return "sparse"
    if successes == 3:
        return "middle"
    return "boundary"


def _select_active(
    records: list[dict[str, Any]], slots: int, seed: int
) -> list[dict[str, Any]]:
    quotas = {
        "sparse": round(slots * 0.50),
        "middle": round(slots * 0.30),
    }
    quotas["boundary"] = slots - sum(quotas.values())
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bucket[_bucket(record)].append(record)
    for name, values in by_bucket.items():
        values.sort(key=lambda row: deterministic_key(seed, name, row["problem_id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for name in ("sparse", "middle", "boundary"):
        for record in by_bucket[name][: quotas[name]]:
            selected.append(record)
            selected_ids.add(str(record["problem_id"]))
    remaining = [
        record
        for record in records
        if str(record["problem_id"]) not in selected_ids
    ]
    remaining.sort(
        key=lambda row: (
            {"sparse": 0, "middle": 1, "boundary": 2}[_bucket(row)],
            deterministic_key(seed, "remainder", row["problem_id"]),
        )
    )
    selected.extend(remaining[: max(0, slots - len(selected))])
    return selected[:slots]


def build_cycle_schedule(
    *,
    selected_source_rows: Sequence[dict[str, Any]],
    receipts: Sequence[dict[str, Any]],
    identity: CycleIdentity,
    config: DynamicCycleConfig | None = None,
    seed: int = 20260909,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create 16 balanced steps, filling shortages with zero-success problems."""

    cfg = config or DynamicCycleConfig()
    cfg.validate()
    if len(selected_source_rows) != cfg.screen_problems:
        raise CycleContractError(
            f"screen pool has {len(selected_source_rows)} rows, expected {cfg.screen_problems}"
        )
    source = {row_id(row): dict(row) for row in selected_source_rows}
    if len(source) != len(selected_source_rows):
        raise CycleContractError("screen pool contains duplicate problems")
    validate_receipt_set(receipts, identity, reward_source="screen_selection_only")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        problem_id = row_id(receipt)
        if problem_id not in source:
            raise CycleContractError(f"receipt problem {problem_id} is outside selection")
        grouped[problem_id].append(receipt)
    if set(grouped) != set(source):
        missing = sorted(set(source) - set(grouped))
        raise CycleContractError(f"{len(missing)} selected problems lack receipts")
    classified = [classify_screen_group(grouped[problem_id]) for problem_id in source]
    active = [record for record in classified if record["route"].startswith("active_")]
    zero = [
        record
        for record in classified
        if record["route"] in {"unresolved_zero", "hard_bank"}
    ]
    chosen_active = _select_active(active, cfg.training_slots, seed)
    chosen_ids = {str(record["problem_id"]) for record in chosen_active}
    zero.sort(
        key=lambda row: (
            row["route"] == "hard_bank",
            deterministic_key(seed, "zero-fill", row["problem_id"]),
        )
    )
    chosen_zero = zero[: cfg.training_slots - len(chosen_active)]
    chosen = [
        {**record, "training_role": "active"} for record in chosen_active
    ] + [
        {**record, "training_role": "exploration_filler"} for record in chosen_zero
    ]
    if len(chosen) != cfg.training_slots:
        raise CycleContractError(
            f"only {len(chosen)} unique train/filler problems for {cfg.training_slots} slots"
        )
    if len({str(row["problem_id"]) for row in chosen}) != len(chosen):
        raise CycleContractError("a problem would repeat inside the cycle")

    # Spread every difficulty bucket and every zero filler across all steps.
    steps: list[list[dict[str, Any]]] = [[] for _ in range(cfg.steps_per_cycle)]
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in chosen:
        key = _bucket(record) if record["training_role"] == "active" else "zero"
        partitions[key].append(record)
    cursor = 0
    for name in ("sparse", "middle", "boundary", "zero"):
        values = partitions[name]
        random.Random(int(deterministic_key(seed, name)[:16], 16)).shuffle(values)
        for record in values:
            candidates = [
                index
                for index, step in enumerate(steps)
                if len(step) < cfg.problems_per_step
            ]
            if not candidates:
                raise AssertionError("schedule overflow")
            index = min(candidates, key=lambda value: (len(steps[value]), (value - cursor) % len(steps)))
            steps[index].append(record)
            cursor = (index + 1) % len(steps)
    if any(len(step) != cfg.problems_per_step for step in steps):
        raise AssertionError("schedule is not rectangular")

    schedule: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        screen_effective = sum(record["training_role"] == "active" for record in step)
        planned_scale = effective_step_scale(screen_effective, cfg)
        for slot_index, record in enumerate(step):
            problem_id = str(record["problem_id"])
            training_row = dict(source[problem_id])
            training_row["repeat"] = 1
            training_row["dynamic_cycle"] = {
                "cycle_id": identity.cycle_id,
                "old_policy_sha256": identity.old_policy_sha256,
                "selection_manifest_sha256": identity.selection_manifest_sha256,
                "step_index": step_index,
                "slot_index": slot_index,
                "training_role": record["training_role"],
                "screen_route": record["route"],
                "screen_successes": record["total_successes"],
                "screen_attempts": record["total_attempts"],
                "planned_screen_effective_groups": screen_effective,
                "planned_effective_step_scale": planned_scale,
                "reward_source": "fresh_training_rollout_current_cycle_only",
            }
            schedule.append(training_row)
    summary = {
        "schema": "dynamic_grpo_cycle_schedule_v2",
        "cycle_identity": asdict(identity),
        "config": asdict(cfg),
        "screen_problem_count": len(source),
        "training_slots": len(schedule),
        "unique_training_problems": len({row_id(row) for row in schedule}),
        "active_training_problems": len(chosen_active),
        "exploration_fillers": len(chosen_zero),
        "mastered_excluded": sum(row["route"] == "mastered" for row in classified),
        "unused_active_reserve": len(active) - len(chosen_active),
        "route_counts": {
            route: sum(row["route"] == route for row in classified)
            for route in sorted({row["route"] for row in classified})
        },
        "reward_source": "fresh_training_rollout_current_cycle_only",
        "history_enters_reward": False,
        "screen_receipts_enter_reward": False,
        "cross_cycle_receipts_allowed": False,
        "strategy_drift_scaling": False,
    }
    return schedule, summary


def validate_training_reward_group(
    receipts: Sequence[dict[str, Any]],
    identity: CycleIdentity,
    *,
    expected_policy_step: int,
) -> None:
    """Fail if online training tries to consume screen or another cycle's result."""

    seen: set[tuple[str, int]] = set()
    for receipt in receipts:
        expected = {
            "rollout_cycle_id": identity.cycle_id,
            "screen_old_policy_sha256": identity.old_policy_sha256,
            "selection_manifest_sha256": identity.selection_manifest_sha256,
            "reward_source": "fresh_training_rollout_current_cycle_only",
            # Online GRPO generates immediately before this optimizer update.
            # It must not pretend that all 16 steps came from the cycle-start
            # screen policy, because the policy changes after every step.
            "rollout_policy_step": expected_policy_step,
        }
        mismatches = [
            f"{key}={receipt.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if receipt.get(key) != value
        ]
        if mismatches:
            raise CycleContractError("training receipt lineage mismatch: " + "; ".join(mismatches))
        attempt = int(receipt.get("attempt_index", -1))
        if attempt not in range(8):
            raise CycleContractError("training attempt index must be in 0..7")
        key = (row_id(receipt), attempt)
        if key in seen:
            raise CycleContractError(f"duplicate training receipt {key}")
        seen.add(key)
    problem_ids = {row_id(row) for row in receipts}
    if len(problem_ids) != 1 or len(receipts) != 8:
        raise CycleContractError("training reward must be one exact pass@8 problem group")


def write_cycle_artifacts(
    output_dir: Path,
    schedule: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = output_dir / "train_schedule.jsonl"
    write_jsonl(schedule_path, schedule)
    final_summary = {
        **summary,
        "train_schedule_sha256": sha256_file(schedule_path),
    }
    (output_dir / "CYCLE_SCHEDULE_FROZEN.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_summary
