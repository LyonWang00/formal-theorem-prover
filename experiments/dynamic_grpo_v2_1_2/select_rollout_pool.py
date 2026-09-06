#!/usr/bin/env python3
"""Select the next pass@8 rollout pool using historical success evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_training_pool import row_id
from route_rollouts import read_jsonl, write_jsonl


def deterministic_tiebreak(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def select_rollouts(
    *,
    source_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    budget: int,
    id_field: str,
    seed: int,
    include_cooldown_banks: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history = {str(row["problem_id"]): row for row in history_rows}
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    deferred: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        problem_id = row_id(source, id_field)
        if problem_id in seen:
            raise ValueError(f"duplicate source problem {problem_id}")
        seen.add(problem_id)
        previous = history.get(problem_id)
        route = str(previous.get("last_route")) if previous else "unseen"
        if route in {"mastered", "hard_bank"} and not include_cooldown_banks:
            deferred.append({"problem_id": problem_id, "reason": route})
            continue
        # Unseen problems receive maximum initial priority.  Seen problems use
        # the posterior probability of landing in the useful 1/8-4/8 band.
        priority = 1.0 if previous is None else float(previous["rollout_priority"])
        row = dict(source)
        row["selection_metadata"] = {
            "history_priority": priority,
            "history_route": route,
            "history_visits": int(previous.get("history_visits", 0)) if previous else 0,
        }
        candidates.append((priority, deterministic_tiebreak(problem_id, seed), row))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [row for _, _, row in candidates[:budget]]
    for _, _, row in candidates[budget:]:
        deferred.append({"problem_id": row_id(row, id_field), "reason": "budget"})
    return selected, deferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--include-cooldown-banks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, deferred = select_rollouts(
        source_rows=read_jsonl(args.source_dataset),
        history_rows=read_jsonl(args.history),
        budget=args.budget,
        id_field=args.id_field,
        seed=args.seed,
        include_cooldown_banks=args.include_cooldown_banks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "rollout_pool.jsonl", selected)
    write_jsonl(args.output_dir / "deferred.jsonl", deferred)
    summary = {
        "schema": "adaptive_grpo_rollout_selection_v1",
        "selected": len(selected),
        "deferred": len(deferred),
        "history_controls_selection_only": True,
        "history_enters_reward": False,
        "seed": args.seed,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
