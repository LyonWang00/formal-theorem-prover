#!/usr/bin/env python3
"""Freeze a capped, history-prioritized pass@16 top-up selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from adaptive_grpo_core import history_rollout_priority


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--generation-staging-pool", type=Path, required=True)
    p.add_argument("--pass8-receipts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260905)
    a = p.parse_args()

    pool_rows = read_jsonl(a.pool)
    staging_rows = read_jsonl(a.generation_staging_pool)
    by_id = {str(row["id"]): row for row in pool_rows}
    staging_by_id = {str(row["id"]): row for row in staging_rows}
    if len(by_id) != len(pool_rows) or set(by_id) != set(staging_by_id):
        raise ValueError("pool/staging ID contract mismatch")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for receipt in read_jsonl(a.pass8_receipts):
        grouped[str(receipt["problem_id"])].append(receipt)
    if set(grouped) != set(by_id):
        raise ValueError("pass@8 receipt problem IDs differ from frozen pool")

    candidates = []
    for problem_id, receipts in grouped.items():
        if len(receipts) != 8 or {int(row["attempt_index"]) for row in receipts} != set(range(8)):
            raise ValueError(f"{problem_id}: expected exact attempts 0..7")
        if any(bool(row["success"]) for row in receipts):
            continue
        row = by_id[problem_id]
        metadata = row["selection_metadata"]
        hs = int(metadata["history_successes"])
        ha = int(metadata["history_attempts"])
        priority = history_rollout_priority(
            historical_successes=hs,
            historical_attempts=ha,
        )
        tie = hashlib.sha256(f"{a.seed}:{problem_id}".encode()).hexdigest()
        candidates.append((priority, hs > 0, tie, problem_id))
    candidates.sort(key=lambda item: (-item[0], -int(item[1]), item[2], item[3]))
    selected = candidates[: a.limit]
    if len(selected) != a.limit:
        raise ValueError(f"only {len(selected)} zero-success candidates for limit {a.limit}")
    selected_ids = [item[3] for item in selected]
    selected_rows = [by_id[problem_id] for problem_id in selected_ids]
    selected_staging = [staging_by_id[problem_id] for problem_id in selected_ids]
    for row in selected_rows + selected_staging:
        if int(row.get("repeat", -1)) != 1:
            raise ValueError("top-up selection requires repeat=1")

    a.output_dir.mkdir(parents=True, exist_ok=False)
    pool_out = a.output_dir / "pass16_topup_pool_128.jsonl"
    staging_out = a.output_dir / "pass16_topup_generation_staging.jsonl"
    write_jsonl(pool_out, selected_rows)
    write_jsonl(staging_out, selected_staging)
    report = {
        "schema": "adaptive_e2_pass16_selection_v1",
        "status": "FROZEN",
        "source_pool": str(a.pool),
        "source_pool_sha256": sha(a.pool),
        "source_generation_staging_pool": str(a.generation_staging_pool),
        "source_generation_staging_pool_sha256": sha(a.generation_staging_pool),
        "pass8_receipts": str(a.pass8_receipts),
        "pass8_receipts_sha256": sha(a.pass8_receipts),
        "zero_success_candidate_count": len(candidates),
        "selected_problem_count": len(selected_rows),
        "selected_with_historical_success": sum(item[1] for item in selected),
        "new_attempts_per_problem": 8,
        "expected_new_attempt_count": len(selected_rows) * 8,
        "selection_seed": a.seed,
        "ranking": "posterior_probability_of_next_pass8_successes_1_to_4_desc",
        "pool_path": str(pool_out),
        "pool_sha256": sha(pool_out),
        "generation_staging_path": str(staging_out),
        "generation_staging_sha256": sha(staging_out),
        "selected_id_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "priority_min": min(item[0] for item in selected),
        "priority_max": max(item[0] for item in selected),
    }
    (a.output_dir / "PASS16_SELECTION_FROZEN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
