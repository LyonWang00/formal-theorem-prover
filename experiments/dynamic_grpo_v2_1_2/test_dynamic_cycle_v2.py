from __future__ import annotations

import hashlib
import math

import pytest

from dynamic_cycle_v2 import (
    CycleContractError,
    CycleIdentity,
    DynamicCycleConfig,
    build_cycle_schedule,
    difficulty_weight,
    effective_step_scale,
    shape_current_cycle_group,
    validate_training_reward_group,
)


IDENTITY = CycleIdentity(
    cycle_id="cycle-0003",
    old_policy_sha256="a" * 64,
    selection_manifest_sha256="b" * 64,
)


def receipt(problem_id: str, attempt: int, success: bool, *, source: str = "screen_selection_only", cycle: str = "cycle-0003", policy: str = "a" * 64) -> dict:
    return {
        "problem_id": problem_id,
        "attempt_index": attempt,
        "success": success,
        "rollout_cycle_id": cycle,
        "old_policy_sha256": policy,
        "selection_manifest_sha256": "b" * 64,
        "reward_source": source,
    }


def group(problem_id: str, successes: int, attempts: int = 8) -> list[dict]:
    return [receipt(problem_id, index, index < successes) for index in range(attempts)]


def test_more_aggressive_difficulty_weights() -> None:
    assert difficulty_weight(1, 8) == 1.30
    assert difficulty_weight(2, 8) == 1.10
    assert difficulty_weight(3, 8) == 0.80
    assert difficulty_weight(4, 8) == 0.60
    assert difficulty_weight(5, 8) == 0.0
    assert difficulty_weight(1, 32) == 1.30


def test_topup_mass_is_not_four_times_larger() -> None:
    result = shape_current_cycle_group(
        [1.0] + [0.0] * 31,
        [-2.0] + [-0.1] * 31,
    )
    assert result["topup_mass_scale"] == 0.25
    assert result["difficulty_weight"] == 1.30
    assert math.isclose(sum(result["centered_advantages"]), 0.0, abs_tol=1e-12)


def test_effective_step_compensation_is_capped() -> None:
    assert effective_step_scale(16) == 1.0
    assert effective_step_scale(12) == 16 / 12
    assert effective_step_scale(8) == 1.5
    assert effective_step_scale(1) == 1.5
    assert effective_step_scale(0) == 0.0


def test_schedule_uses_unique_active_then_zero_fillers() -> None:
    source = [{"id": f"p{index:03d}", "prompt": "x", "repeat": 9} for index in range(512)]
    receipts = []
    # 220 useful, 32 mastered, 260 zero: schedule must use 220+36 fillers.
    for index in range(512):
        successes = 1 + index % 4 if index < 220 else 6 if index < 252 else 0
        receipts.extend(group(f"p{index:03d}", successes))
    schedule, summary = build_cycle_schedule(
        selected_source_rows=source,
        receipts=receipts,
        identity=IDENTITY,
    )
    assert len(schedule) == 256
    assert len({row["id"] for row in schedule}) == 256
    assert all(row["repeat"] == 1 for row in schedule)
    assert summary["active_training_problems"] == 220
    assert summary["exploration_fillers"] == 36
    assert summary["mastered_excluded"] == 32
    assert summary["screen_receipts_enter_reward"] is False
    step_counts = {}
    for row in schedule:
        meta = row["dynamic_cycle"]
        step_counts[meta["step_index"]] = step_counts.get(meta["step_index"], 0) + 1
        assert meta["reward_source"] == "fresh_training_rollout_current_cycle_only"
    assert step_counts == {index: 16 for index in range(16)}


def test_schedule_caps_active_at_256_without_repeating() -> None:
    source = [{"id": f"p{index:03d}"} for index in range(512)]
    receipts = []
    for index in range(512):
        receipts.extend(group(f"p{index:03d}", 1 + index % 4))
    schedule, summary = build_cycle_schedule(
        selected_source_rows=source,
        receipts=receipts,
        identity=IDENTITY,
    )
    assert len(schedule) == 256
    assert summary["active_training_problems"] == 256
    assert summary["unused_active_reserve"] == 256
    assert not any(row["dynamic_cycle"]["training_role"] == "exploration_filler" for row in schedule)


@pytest.mark.parametrize(
    "mutation",
    [
        {"rollout_cycle_id": "cycle-0002"},
        {"screen_old_policy_sha256": "c" * 64},
        {"selection_manifest_sha256": "d" * 64},
        {"reward_source": "screen_selection_only"},
        {"rollout_policy_step": 40},
    ],
)
def test_training_reward_rejects_wrong_round_or_source(mutation: dict) -> None:
    rows = group("p000", 1)
    rows = [
        {
            **row,
            "reward_source": "fresh_training_rollout_current_cycle_only",
            "screen_old_policy_sha256": "a" * 64,
            "rollout_policy_step": 41,
        }
        for row in rows
    ]
    rows[3].update(mutation)
    with pytest.raises(CycleContractError, match="lineage mismatch"):
        validate_training_reward_group(rows, IDENTITY, expected_policy_step=41)


def test_training_reward_accepts_one_exact_current_cycle_group() -> None:
    rows = group("p000", 1)
    rows = [
        {
            **row,
            "reward_source": "fresh_training_rollout_current_cycle_only",
            "screen_old_policy_sha256": "a" * 64,
            "rollout_policy_step": 41,
        }
        for row in rows
    ]
    validate_training_reward_group(rows, IDENTITY, expected_policy_step=41)


def test_screen_receipt_from_other_cycle_fails_closed() -> None:
    source = [{"id": f"p{index:03d}"} for index in range(512)]
    receipts = []
    for index in range(512):
        receipts.extend(group(f"p{index:03d}", 1))
    receipts[17]["rollout_cycle_id"] = "cycle-from-another-policy"
    with pytest.raises(CycleContractError, match="lineage mismatch"):
        build_cycle_schedule(
            selected_source_rows=source,
            receipts=receipts,
            identity=IDENTITY,
        )
