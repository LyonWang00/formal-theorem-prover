#!/usr/bin/env python3
"""Freeze the capped pass@32 selection from pass@16 zero-success problems."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from adaptive_grpo_core import history_rollout_priority


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in values),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--generation-staging-pool", type=Path, required=True)
    p.add_argument("--pass16-topup-receipts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260906)
    a = p.parse_args()

    pool = {str(row["id"]): row for row in rows(a.pool)}
    staging = {str(row["id"]): row for row in rows(a.generation_staging_pool)}
    if set(pool) != set(staging):
        raise ValueError("pass16 pool/staging IDs differ")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for receipt in rows(a.pass16_topup_receipts):
        grouped[str(receipt["problem_id"])].append(receipt)
    if set(grouped) != set(pool):
        raise ValueError("pass16 receipts do not cover the frozen pass16 selection")

    candidates = []
    for problem_id, receipts in grouped.items():
        if len(receipts) != 8 or {int(row["attempt_index"]) for row in receipts} != set(range(8)):
            raise ValueError(f"{problem_id}: expected exact pass16 topup attempts 0..7")
        if any(bool(row["success"]) for row in receipts):
            continue
        metadata = pool[problem_id]["selection_metadata"]
        hs = int(metadata["history_successes"])
        ha = int(metadata["history_attempts"])
        if hs <= 0:
            continue
        priority = history_rollout_priority(historical_successes=hs, historical_attempts=ha)
        tie = hashlib.sha256(f"{a.seed}:{problem_id}".encode()).hexdigest()
        candidates.append((priority, tie, problem_id))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[: a.limit]
    if len(selected) != a.limit:
        raise ValueError(f"only {len(selected)} eligible pass32 candidates for limit {a.limit}")
    ids = [item[2] for item in selected]
    selected_pool = [pool[problem_id] for problem_id in ids]
    selected_staging = [staging[problem_id] for problem_id in ids]
    if any(int(row.get("repeat", -1)) != 1 for row in selected_pool + selected_staging):
        raise ValueError("pass32 selection requires repeat=1")

    a.output_dir.mkdir(parents=True, exist_ok=False)
    pool_out = a.output_dir / "pass32_topup_pool_64.jsonl"
    staging_out = a.output_dir / "pass32_topup_generation_staging.jsonl"
    dump(pool_out, selected_pool)
    dump(staging_out, selected_staging)
    report = {
        "schema": "adaptive_e2_pass32_selection_v1",
        "status": "FROZEN",
        "eligible_zero_after_pass16": len(candidates),
        "selected_problem_count": len(ids),
        "new_attempts_per_problem": 16,
        "expected_new_attempt_count": len(ids) * 16,
        "selection_seed": a.seed,
        "ranking": "posterior_probability_of_next_pass8_successes_1_to_4_desc",
        "source_pool_sha256": sha(a.pool),
        "source_generation_staging_pool_sha256": sha(a.generation_staging_pool),
        "pass16_topup_receipts_sha256": sha(a.pass16_topup_receipts),
        "pool_path": str(pool_out),
        "pool_sha256": sha(pool_out),
        "generation_staging_path": str(staging_out),
        "generation_staging_sha256": sha(staging_out),
        "selected_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "priority_min": min(item[0] for item in selected),
        "priority_max": max(item[0] for item in selected),
    }
    (a.output_dir / "PASS32_SELECTION_FROZEN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
