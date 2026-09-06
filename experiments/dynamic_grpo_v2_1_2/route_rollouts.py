#!/usr/bin/env python3
"""Route capped pass@8/pass@16/pass@32 receipts into adaptive GRPO banks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from adaptive_grpo_core import (
    RewardConfig,
    Route,
    history_rollout_priority,
    problem_weight,
    route_rollout,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def is_success(row: dict[str, Any]) -> bool:
    value = row.get("success", row.get("reward", False))
    if isinstance(value, bool):
        return value
    return float(value) > 0.5


def attempt_index(row: dict[str, Any]) -> int:
    for key in ("attempt_index", "sample_index", "completion_index"):
        if key in row:
            return int(row[key])
    raise KeyError("receipt needs attempt_index, sample_index, or completion_index")


def merge_history(
    old_rows: list[dict[str, Any]], snapshots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    history: dict[str, dict[str, Any]] = {
        str(row["problem_id"]): dict(row) for row in old_rows
    }
    for snapshot in snapshots:
        problem_id = str(snapshot["problem_id"])
        previous = history.get(problem_id, {})
        records = list(previous.get("rollout_history", []))
        records.append(snapshot)
        rates = [float(record["success_rate"]) for record in records]
        historical_successes = sum(int(record["successes"]) for record in records)
        historical_attempts = sum(int(record["attempts"]) for record in records)
        effective_visits = sum(
            0.0 < float(record["success_rate"]) < 1.0 for record in records
        )
        history[problem_id] = {
            "problem_id": problem_id,
            "rollout_history": records,
            "history_visits": len(records),
            "history_effective_visits": effective_visits,
            "history_mean_success_rate": sum(rates) / len(rates),
            "history_successes": historical_successes,
            "history_attempts": historical_attempts,
            "rollout_priority": history_rollout_priority(
                historical_successes=historical_successes,
                historical_attempts=historical_attempts,
            ),
            "last_route": snapshot["route"],
            "last_seen_at": snapshot["observed_at"],
        }
    return sorted(history.values(), key=lambda row: row["problem_id"])


def route_receipts(
    receipts: list[dict[str, Any]], *, run_id: str
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[str(row["problem_id"])].append(row)
    buckets = {route.value: [] for route in Route}
    snapshots: list[dict[str, Any]] = []
    observed_at = datetime.now(timezone.utc).isoformat()
    cfg = RewardConfig()

    for problem_id, attempts in grouped.items():
        attempts.sort(key=attempt_index)
        indices = [attempt_index(row) for row in attempts]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate attempt indices for {problem_id}")
        # Use sorted position rather than assuming attempt indices are 0-based;
        # existing evaluation artifacts contain both 0-based and 1-based forms.
        initial = attempts[:8]
        if len(initial) != 8:
            raise ValueError(f"{problem_id} has {len(initial)} initial attempts, expected 8")
        if len(attempts) > 32:
            raise ValueError(f"{problem_id} has more than 32 attempts")
        initial_successes = sum(is_success(row) for row in initial)
        total_successes = sum(is_success(row) for row in attempts)
        route = route_rollout(
            initial_successes=initial_successes,
            total_successes=total_successes,
            total_attempts=len(attempts),
        )
        current_attempts = 8 if route in (Route.ACTIVE_PASS8, Route.MASTERED) else len(attempts)
        current_successes = (
            initial_successes if current_attempts == 8 else total_successes
        )
        weight = problem_weight(
            successes=current_successes, attempts=current_attempts, config=cfg
        )
        exemplar = attempts[0]
        record = {
            "problem_id": problem_id,
            "route": route.value,
            "initial_successes": initial_successes,
            "initial_attempts": 8,
            "total_successes": total_successes,
            "total_attempts": len(attempts),
            "current_success_rate": current_successes / current_attempts,
            "problem_reward_weight": weight,
            "training_ticket_count": 1 if route in (Route.ACTIVE_PASS8, Route.ACTIVE_AFTER_TOPUP) else 0,
            "topup_attempts_needed": (
                8 if route is Route.NEED_PASS16_TOPUP else
                16 if route is Route.NEED_PASS32_TOPUP else 0
            ),
            "source": {
                key: exemplar[key]
                for key in ("prompt", "statement", "source", "source_path", "repeat")
                if key in exemplar
            },
        }
        buckets[route.value].append(record)
        snapshots.append(
            {
                "problem_id": problem_id,
                "run_id": run_id,
                "observed_at": observed_at,
                "route": route.value,
                "successes": current_successes,
                "attempts": current_attempts,
                "success_rate": current_successes / current_attempts,
                "problem_reward_weight": weight,
            }
        )
    for rows in buckets.values():
        rows.sort(key=lambda row: row["problem_id"])
    return buckets, snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buckets, snapshots = route_receipts(read_jsonl(args.receipts), run_id=args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for route_name, rows in buckets.items():
        write_jsonl(args.output_dir / f"{route_name}.jsonl", rows)
    prior_history = read_jsonl(args.history) if args.history else []
    history = merge_history(prior_history, snapshots)
    write_jsonl(args.output_dir / "rollout_history.jsonl", history)
    summary = {
        "run_id": args.run_id,
        "input_problems": sum(len(rows) for rows in buckets.values()),
        "problem_counts": {name: len(rows) for name, rows in buckets.items()},
        "active_training_tickets": sum(
            len(buckets[name])
            for name in (Route.ACTIVE_PASS8.value, Route.ACTIVE_AFTER_TOPUP.value)
        ),
        "topup_is_admission_only": True,
        "pass32_gradient_multiplier": 1,
    }
    (args.output_dir / "routing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
