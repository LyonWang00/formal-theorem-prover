#!/usr/bin/env python3
"""End-to-end synthetic smoke test for one dynamic GRPO cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from dynamic_cycle_controller import bind_receipts, finalize, freeze_cycle, plan_topup
from dynamic_cycle_v2 import CycleContractError, read_jsonl, sha256_bytes, validate_training_reward_group


POLICY = "a" * 64


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def raw_stage(root: Path, pool: Path, attempts: int, success_ids: set[str]) -> tuple[Path, Path]:
    generations = []
    receipts = []
    for row in read_jsonl(pool):
        problem_id = str(row["id"])
        statement_hash = str(row["statement_hash"])
        for attempt in range(attempts):
            payload = sha256_bytes(f"{root.name}:{problem_id}:{attempt}".encode())
            generations.append(
                {
                    "problem_id": problem_id,
                    "attempt_index": attempt,
                    "model_sums_sha256": POLICY,
                    "generation_payload_sha256": payload,
                    "statement_hash": statement_hash,
                }
            )
            receipts.append(
                {
                    "problem_id": problem_id,
                    "attempt_index": attempt,
                    "success": problem_id in success_ids and attempt == 0,
                    "generation_payload_sha256": payload,
                    "statement_hash": statement_hash,
                }
            )
    generation_root = root / "generation"
    raw_receipts = root / "receipts.jsonl"
    dump(generation_root / "chunks" / "problem_000000.jsonl", generations)
    dump(raw_receipts, receipts)
    return generation_root, raw_receipts


def bind(cycle: Path, stage: str, pool: Path, generation: Path, raw: Path, output: Path) -> None:
    bind_receipts(
        SimpleNamespace(
            cycle_dir=cycle,
            stage=stage,
            stage_pool=pool,
            generation_root=generation,
            raw_receipts=raw,
            output=output,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    source = root / "source.jsonl"
    source_rows = [
        {
            "id": f"p{index:03d}",
            "prompt": "import Mathlib\n\ntheorem t : True :=\n",
            "statement_hash": sha256_bytes(f"statement-{index}".encode()),
            "repeat": 9,
        }
        for index in range(512)
    ]
    dump(source, source_rows)
    cycle = root / "cycle-0000"
    freeze_cycle(
        SimpleNamespace(
            source_dataset=source,
            history=None,
            cycle_id="cycle-0000",
            old_policy_sha256=POLICY,
            output_dir=cycle,
            seed=7,
        )
    )
    screen_pool = cycle / "screen_pool.jsonl"
    selected = read_jsonl(screen_pool)
    # 220 useful, 32 mastered, 260 zero.
    active = {str(row["id"]) for row in selected[:220]}
    mastered = {str(row["id"]) for row in selected[220:252]}
    generation = []
    receipts = []
    for row in selected:
        problem_id = str(row["id"])
        successes = 1 + int(problem_id[1:]) % 4 if problem_id in active else 6 if problem_id in mastered else 0
        for attempt in range(8):
            payload = sha256_bytes(f"pass8:{problem_id}:{attempt}".encode())
            generation.append(
                {
                    "problem_id": problem_id,
                    "attempt_index": attempt,
                    "model_sums_sha256": POLICY,
                    "generation_payload_sha256": payload,
                    "statement_hash": row["statement_hash"],
                }
            )
            receipts.append(
                {
                    "problem_id": problem_id,
                    "attempt_index": attempt,
                    "success": attempt < successes,
                    "generation_payload_sha256": payload,
                    "statement_hash": row["statement_hash"],
                }
            )
    pass8_root = root / "pass8"
    dump(pass8_root / "generation/chunks/problem_000000.jsonl", generation)
    dump(pass8_root / "receipts.jsonl", receipts)
    wrong_generation = [dict(row) for row in generation]
    wrong_generation[37]["model_sums_sha256"] = "c" * 64
    wrong_root = root / "pass8-wrong-policy"
    dump(wrong_root / "generation/chunks/problem_000000.jsonl", wrong_generation)
    try:
        bind(
            cycle,
            "pass8",
            screen_pool,
            wrong_root / "generation",
            pass8_root / "receipts.jsonl",
            cycle / "must-not-exist-wrong-policy.jsonl",
        )
    except CycleContractError:
        pass
    else:
        raise AssertionError("generation from a different old policy was accepted")
    pass8_bound = cycle / "pass8_bound.jsonl"
    bind(cycle, "pass8", screen_pool, pass8_root / "generation", pass8_root / "receipts.jsonl", pass8_bound)

    pass16_plan = cycle / "pass16_plan"
    plan_topup(
        SimpleNamespace(
            cycle_dir=cycle,
            stage="pass16",
            bound_receipts=[pass8_bound],
            output_dir=pass16_plan,
            seed=8,
        )
    )
    pass16_pool = pass16_plan / "pass16_pool.jsonl"
    pass16_rows = read_jsonl(pass16_pool)
    assert len(pass16_rows) == 128
    pass16_success = {str(row["id"]) for row in pass16_rows[:10]}
    generation16, raw16 = raw_stage(root / "pass16", pass16_pool, 8, pass16_success)
    pass16_bound = cycle / "pass16_bound.jsonl"
    bind(cycle, "pass16", pass16_pool, generation16, raw16, pass16_bound)

    pass32_plan = cycle / "pass32_plan"
    plan_topup(
        SimpleNamespace(
            cycle_dir=cycle,
            stage="pass32",
            bound_receipts=[pass8_bound, pass16_bound],
            output_dir=pass32_plan,
            seed=9,
        )
    )
    pass32_pool = pass32_plan / "pass32_pool.jsonl"
    pass32_rows = read_jsonl(pass32_pool)
    assert len(pass32_rows) == 64
    pass32_success = {str(row["id"]) for row in pass32_rows[:5]}
    generation32, raw32 = raw_stage(root / "pass32", pass32_pool, 16, pass32_success)
    pass32_bound = cycle / "pass32_bound.jsonl"
    bind(cycle, "pass32", pass32_pool, generation32, raw32, pass32_bound)

    schedule_dir = cycle / "schedule"
    finalize(
        SimpleNamespace(
            cycle_dir=cycle,
            bound_receipts=[pass8_bound, pass16_bound, pass32_bound],
            output_dir=schedule_dir,
            seed=10,
            history=None,
        )
    )
    schedule = read_jsonl(schedule_dir / "train_schedule.jsonl")
    frozen = json.loads((schedule_dir / "CYCLE_SCHEDULE_FROZEN.json").read_text())
    assert len(schedule) == 256
    assert len({row["id"] for row in schedule}) == 256
    assert all(row["repeat"] == 1 for row in schedule)
    assert frozen["active_training_problems"] == 235
    assert frozen["exploration_fillers"] == 21
    assert frozen["strategy_drift_scaling"] is False

    # Positive control: one exact online training group from this cycle.
    training_group = []
    for attempt in range(8):
        training_group.append(
            {
                "problem_id": schedule[0]["id"],
                "attempt_index": attempt,
                "success": attempt == 0,
                "rollout_cycle_id": "cycle-0000",
                "screen_old_policy_sha256": POLICY,
                "selection_manifest_sha256": frozen["cycle_identity"]["selection_manifest_sha256"],
                "reward_source": "fresh_training_rollout_current_cycle_only",
                "rollout_policy_step": 32,
            }
        )
    validate_training_reward_group(
        training_group,
        __import__("dynamic_cycle_v2").CycleIdentity(**frozen["cycle_identity"]),
        expected_policy_step=32,
    )

    # Negative controls: a stale cycle and a stale policy must fail closed.
    for field, wrong in (("rollout_cycle_id", "cycle-old"), ("screen_old_policy_sha256", "c" * 64), ("rollout_policy_step", 31)):
        corrupted = [dict(row) for row in training_group]
        corrupted[2][field] = wrong
        try:
            validate_training_reward_group(
                corrupted,
                __import__("dynamic_cycle_v2").CycleIdentity(**frozen["cycle_identity"]),
                expected_policy_step=32,
            )
        except CycleContractError:
            pass
        else:
            raise AssertionError(f"stale {field} was accepted")

    report = {
        "status": "PASS",
        "screen_problems": 512,
        "pass8_attempts": 4096,
        "pass16_topup_problems": 128,
        "pass32_topup_problems": 64,
        "training_slots": 256,
        "active_training_problems": 235,
        "exploration_fillers": 21,
        "cross_cycle_negative_tests": 3,
        "wrong_generation_policy_negative_tests": 1,
    }
    (root / "SMOKE_PASS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
