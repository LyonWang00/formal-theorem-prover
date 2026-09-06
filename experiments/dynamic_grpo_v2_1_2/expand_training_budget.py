#!/usr/bin/env python3
"""Assign balanced repeat counts for an exact adaptive optimizer-step budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=149)
    parser.add_argument("--problems-per-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("input pool is empty")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("input pool must contain one physical row per problem")
    target_tickets = args.target_steps * args.problems_per_step
    quotient, remainder = divmod(target_tickets, len(rows))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    boosted = set(order[:remainder])
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["repeat"] = quotient + int(index in boosted)
        if row["repeat"] <= 0:
            raise ValueError("target budget is smaller than the unique pool")
        output.append(row)
    if sum(int(row["repeat"]) for row in output) != target_tickets:
        raise AssertionError("repeat allocation does not equal the target ticket count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema": "adaptive_exact_step_budget_v1",
        "target_optimizer_steps": args.target_steps,
        "problems_per_optimizer_step": args.problems_per_step,
        "repeat_expanded_tickets": target_tickets,
        "unique_problems": len(output),
        "repeat_min": min(int(row["repeat"]) for row in output),
        "repeat_max": max(int(row["repeat"]) for row in output),
        "seed": args.seed,
        "source_sha256": sha(args.input),
        "output_sha256": sha(args.output),
        "r6_global_step_reference": 4752,
        "comparability_note": "149 E2 steps use 2384 problem groups and 19072 rollout attempts, within eight groups of the R6 rollout budget.",
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
