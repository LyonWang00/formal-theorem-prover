#!/usr/bin/env python3
"""Apply the mandatory C0 zero-step drift gate before any B/C training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_generation_drift import summarize


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-greedy", required=True, type=Path)
    parser.add_argument("--c0-greedy", required=True, type=Path)
    parser.add_argument("--m0-retention", required=True, type=Path)
    parser.add_argument("--c0-retention", required=True, type=Path)
    parser.add_argument("--m0-generations", required=True, type=Path)
    parser.add_argument("--c0-generations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    m0_greedy = read_jsonl(args.m0_greedy)
    c0_greedy = read_jsonl(args.c0_greedy)
    if len(m0_greedy) != 10 or len(c0_greedy) != 10:
        raise ValueError("C0 greedy gate requires exactly 10 prompts/model")
    first_token_matches = 0
    exact_matches = 0
    details = []
    for m0, c0 in zip(m0_greedy, c0_greedy, strict=True):
        if m0["record_id"] != c0["record_id"]:
            raise ValueError("C0 greedy prompt alignment mismatch")
        left = list(m0["token_ids"])
        right = list(c0["token_ids"])
        first_match = bool(left and right and left[0] == right[0])
        exact = left == right
        first_token_matches += int(first_match)
        exact_matches += int(exact)
        details.append(
            {
                "record_id": m0["record_id"],
                "first_token_match": first_match,
                "exact_match": exact,
                "m0_first_token": left[0] if left else None,
                "c0_first_token": right[0] if right else None,
            }
        )

    m0_retention = read_json(args.m0_retention)
    c0_retention = read_json(args.c0_retention)
    m0_solved = int(m0_retention["successes"])
    c0_solved = int(c0_retention["successes"])
    m0_behavior = summarize(read_jsonl(args.m0_generations))
    c0_behavior = summarize(read_jsonl(args.c0_generations))
    behavior_deltas = {
        key: c0_behavior[key] - m0_behavior[key]
        for key in (
            "extraction_success_rate",
            "length_finish_ratio",
            "repetitive_output_ratio",
            "multiple_proof_ratio",
        )
    }
    behavior_gate = all(abs(value) <= 0.10 for value in behavior_deltas.values())
    checks = {
        "first_token_10_of_10": first_token_matches == 10,
        "retention_solved_difference_at_most_2": abs(c0_solved - m0_solved) <= 2,
        "no_systematic_format_or_extraction_drift": behavior_gate,
    }
    report = {
        "success": all(checks.values()),
        "checks": checks,
        "greedy": {
            "prompts": 10,
            "first_token_matches": first_token_matches,
            "exact_matches": exact_matches,
            "details": details,
        },
        "retention": {
            "m0_solved": m0_solved,
            "c0_solved": c0_solved,
            "difference": c0_solved - m0_solved,
            "m0_pass_at_1": m0_retention["discovery_replay_pass_at_1"],
            "m0_pass_at_4": m0_retention["discovery_replay_pass_at_4"],
            "c0_pass_at_1": c0_retention["discovery_replay_pass_at_1"],
            "c0_pass_at_4": c0_retention["discovery_replay_pass_at_4"],
        },
        "generation_behavior": {
            "M0": m0_behavior,
            "C0": c0_behavior,
            "delta": behavior_deltas,
            "systematic_drift_threshold_absolute": 0.10,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "generation_behavior"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
