#!/usr/bin/env python3
"""Build an aligned one-ticket-per-problem GRPO pool from routing manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from adaptive_grpo_core import make_memory_safe_layout
from route_rollouts import read_jsonl, write_jsonl


def row_id(row: dict[str, Any], id_field: str) -> str:
    for key in (id_field, "problem_id", "id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"row has no {id_field!r}, 'problem_id', or 'id'")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pool(
    *,
    source_rows: list[dict[str, Any]],
    active_ids: set[str],
    priority: dict[str, float],
    id_field: str,
    problems_per_update: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        problem_id = row_id(source, id_field)
        if problem_id not in active_ids:
            continue
        if problem_id in unique:
            raise ValueError(f"duplicate source problem {problem_id}")
        row = dict(source)
        # Top-up attempts are admission evidence only.  Every admitted problem
        # contributes exactly one prompt ticket to the live policy update.
        if "repeat" in row:
            row["repeat"] = 1
        unique[problem_id] = row
    missing = sorted(active_ids - set(unique))
    if missing:
        raise ValueError(f"{len(missing)} active IDs are absent from source data: {missing[:5]}")
    ranked = sorted(
        unique.items(),
        key=lambda item: (-priority.get(item[0], 0.0), item[0]),
    )
    aligned_count = len(ranked) // problems_per_update * problems_per_update
    if aligned_count == 0:
        raise ValueError("not enough active problems for one optimizer window")
    selected = [row for _, row in ranked[:aligned_count]]
    holdout = [row for _, row in ranked[aligned_count:]]
    random.Random(seed).shuffle(selected)
    return selected, holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--problems-per-update", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_routes = [
        *read_jsonl(args.routing_dir / "active_pass8.jsonl"),
        *read_jsonl(args.routing_dir / "active_after_topup.jsonl"),
    ]
    active_ids = {str(row["problem_id"]) for row in active_routes}
    history = read_jsonl(args.history)
    priorities = {
        str(row["problem_id"]): float(row.get("rollout_priority", 0.0))
        for row in history
    }
    selected, holdout = build_pool(
        source_rows=read_jsonl(args.source_dataset),
        active_ids=active_ids,
        priority=priorities,
        id_field=args.id_field,
        problems_per_update=args.problems_per_update,
        seed=args.seed,
    )
    layout = make_memory_safe_layout(
        problems_per_update=args.problems_per_update,
        num_iterations=args.num_iterations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "train.jsonl"
    holdout_path = args.output_dir / "alignment_holdout.jsonl"
    write_jsonl(pool_path, selected)
    write_jsonl(holdout_path, holdout)
    audit = {
        "schema": "adaptive_grpo_training_pool_v1",
        "source_dataset": str(args.source_dataset),
        "active_candidates": len(active_ids),
        "selected_problems": len(selected),
        "alignment_holdout_problems": len(holdout),
        "problems_per_update": layout.prompt_rows_per_update,
        "generation_batch_size": layout.generation_batch_size,
        "gradient_accumulation_steps": layout.gradient_accumulation_steps,
        "num_iterations": layout.num_iterations,
        "pass32_gradient_multiplier": 1,
        "seed": args.seed,
        "train_sha256": file_sha256(pool_path),
        "holdout_sha256": file_sha256(holdout_path),
    }
    (args.output_dir / "POOL_FROZEN.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
