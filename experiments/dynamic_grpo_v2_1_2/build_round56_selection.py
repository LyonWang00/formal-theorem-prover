#!/usr/bin/env python3
"""Build the fixed 512-problem E2 rollout selection from round-5/6 evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from route_rollouts import read_jsonl, write_jsonl


QUOTAS = {
    "active_both": 128,
    "active_hard": 192,
    "active_mastered": 128,
    "hard_both": 64,
}


def status(successes: int) -> str:
    return "hard" if successes == 0 else "active" if successes <= 4 else "mastered"


def load_successes(paths: list[Path]) -> dict[str, int]:
    grouped: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    for path in paths:
        for row in read_jsonl(path):
            problem_id = str(row["problem_id"])
            trajectory_key = (int(row["rank"]), int(row["reward_index"]))
            if trajectory_key in grouped[problem_id]:
                raise ValueError(f"duplicate trajectory key {problem_id}/{trajectory_key}")
            grouped[problem_id][trajectory_key] = bool(
                row.get("success", float(row["reward"]) > 0.5)
            )
    bad = {problem_id: len(values) for problem_id, values in grouped.items() if len(values) != 8}
    if bad:
        raise ValueError(f"non-pass@8 historical groups: {list(bad.items())[:5]}")
    return {problem_id: sum(values.values()) for problem_id, values in grouped.items()}


def stratum(r5: int, r6: int) -> str | None:
    pair = {status(r5), status(r6)}
    if pair == {"active"}:
        return "active_both"
    if pair == {"active", "hard"}:
        return "active_hard"
    if pair == {"active", "mastered"}:
        return "active_mastered"
    if pair == {"hard"}:
        return "hard_both"
    return None


def tie(seed: int, problem_id: str) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--round5", type=Path, nargs="+", required=True)
    parser.add_argument("--round6", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    source = read_jsonl(args.source)
    source_by_id = {str(row["id"]): row for row in source}
    if len(source_by_id) != len(source):
        raise ValueError("source IDs are not unique")
    r5 = load_successes(args.round5)
    r6 = load_successes(args.round6)
    if set(r5) != set(source_by_id) or set(r6) != set(source_by_id):
        raise ValueError("round-5/6 histories do not exactly cover the source dataset")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    history_rows: list[dict[str, Any]] = []
    for problem_id in source_by_id:
        label = stratum(r5[problem_id], r6[problem_id])
        history = {
            "problem_id": problem_id,
            "round5_successes": r5[problem_id],
            "round6_successes": r6[problem_id],
            "round5_status": status(r5[problem_id]),
            "round6_status": status(r6[problem_id]),
            "selection_stratum": label or "excluded_non_boundary",
            "history_successes": r5[problem_id] + r6[problem_id],
            "history_attempts": 16,
        }
        history_rows.append(history)
        if label:
            buckets[label].append(history)

    selected: list[dict[str, Any]] = []
    selected_history: list[dict[str, Any]] = []
    for label, quota in QUOTAS.items():
        rows = sorted(
            buckets[label],
            key=lambda row: (
                abs((row["history_successes"] / 16) - 0.25),
                tie(args.seed, row["problem_id"]),
            ),
        )
        if len(rows) < quota:
            raise ValueError(f"stratum {label} has {len(rows)}, needs {quota}")
        for rank, history in enumerate(rows[:quota]):
            problem_id = history["problem_id"]
            row = dict(source_by_id[problem_id])
            row["repeat"] = 1
            row["selection_metadata"] = {
                **history,
                "stratum_rank": rank,
                "selection_seed": args.seed,
                "history_controls_selection_only": True,
            }
            selected.append(row)
            selected_history.append(history)

    selected.sort(key=lambda row: tie(args.seed + 1, str(row["id"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "rollout_pool_512.jsonl"
    write_jsonl(output, selected)
    write_jsonl(args.output_dir / "round56_history.jsonl", history_rows)
    write_jsonl(args.output_dir / "selected_history.jsonl", selected_history)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "schema": "adaptive_e2_round56_stratified_selection_v1",
        "selected": len(selected),
        "quotas": QUOTAS,
        "available": {key: len(value) for key, value in sorted(buckets.items())},
        "seed": args.seed,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "rollout_pool_sha256": digest,
        "repeat_is_one": all(row.get("repeat") == 1 for row in selected),
    }
    (args.output_dir / "SELECTION_FROZEN.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
