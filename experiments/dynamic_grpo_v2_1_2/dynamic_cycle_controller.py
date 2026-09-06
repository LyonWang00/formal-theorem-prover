#!/usr/bin/env python3
"""Artifact-bound controller for one 512-screen / 16-step GRPO cycle."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from dynamic_cycle_v2 import (
    CycleContractError,
    CycleIdentity,
    DynamicCycleConfig,
    build_cycle_schedule,
    deterministic_key,
    is_success,
    read_jsonl,
    row_id,
    sha256_file,
    validate_receipt_set,
    write_cycle_artifacts,
    write_jsonl,
)
from adaptive_grpo_core import history_rollout_priority


def load_manifest(cycle_dir: Path) -> tuple[dict[str, Any], CycleIdentity]:
    manifest = json.loads((cycle_dir / "CYCLE_FROZEN.json").read_text(encoding="utf-8"))
    identity = CycleIdentity(
        cycle_id=str(manifest["cycle_id"]),
        old_policy_sha256=str(manifest["old_policy_sha256"]),
        selection_manifest_sha256=str(manifest["screen_pool_sha256"]),
    )
    identity.validate()
    if sha256_file(cycle_dir / "screen_pool.jsonl") != identity.selection_manifest_sha256:
        raise CycleContractError("screen pool changed after cycle freeze")
    return manifest, identity


def history_rank(row: dict[str, Any], history: dict[str, dict[str, Any]], seed: int) -> tuple:
    problem_id = row_id(row)
    previous = history.get(problem_id)
    if previous is None:
        return (0, 0.0, deterministic_key(seed, problem_id))
    route = str(previous.get("last_route", "active_pass8"))
    tier = 1 if route.startswith("active_") or route == "unresolved_zero" else 2
    return (
        tier,
        int(previous.get("history_visits", 0)) if tier == 2 else 0,
        -float(previous.get("rollout_priority", 0.0)),
        deterministic_key(seed, problem_id),
    )


def freeze_cycle(args: argparse.Namespace) -> None:
    cfg = DynamicCycleConfig()
    cfg.validate()
    source_rows = read_jsonl(args.source_dataset)
    unique = {row_id(row): row for row in source_rows}
    if len(unique) != len(source_rows):
        raise CycleContractError("source dataset contains duplicate problem IDs")
    history_rows = read_jsonl(args.history) if args.history else []
    history = {str(row["problem_id"]): row for row in history_rows}
    ranked = sorted(source_rows, key=lambda row: history_rank(row, history, args.seed))
    if len(ranked) < cfg.screen_problems:
        raise CycleContractError("source dataset is smaller than the 512-problem screen")
    selected = []
    for row in ranked[: cfg.screen_problems]:
        value = dict(row)
        value["repeat"] = 1
        value["dynamic_screen"] = {
            "cycle_id": args.cycle_id,
            "old_policy_sha256": args.old_policy_sha256,
            "reward_source": "screen_selection_only",
        }
        selected.append(value)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pool = args.output_dir / "screen_pool.jsonl"
    write_jsonl(pool, selected)
    manifest = {
        "schema": "dynamic_grpo_cycle_frozen_v2",
        "status": "FROZEN",
        "cycle_id": args.cycle_id,
        "old_policy_sha256": args.old_policy_sha256,
        "source_dataset": str(args.source_dataset),
        "source_dataset_sha256": sha256_file(args.source_dataset),
        "history_sha256": sha256_file(args.history) if args.history else None,
        "screen_pool_sha256": sha256_file(pool),
        "screen_problem_count": len(selected),
        "selection_seed": args.seed,
        "config": asdict(cfg),
        "screen_is_selection_only": True,
        "training_reward_source": "fresh_training_rollout_current_cycle_only",
        "strategy_drift_scaling": False,
    }
    (args.output_dir / "CYCLE_FROZEN.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def generation_rows(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.rglob("problem_*.jsonl")):
        for row in read_jsonl(path):
            key = (row_id(row), int(row["attempt_index"]))
            if key in result:
                raise CycleContractError(f"duplicate generated attempt {key}")
            result[key] = row
    return result


def bind_receipts(args: argparse.Namespace) -> None:
    _, identity = load_manifest(args.cycle_dir)
    pool_rows = read_jsonl(args.stage_pool)
    pool_ids = {row_id(row) for row in pool_rows}
    if len(pool_ids) != len(pool_rows):
        raise CycleContractError("stage pool contains duplicate problems")
    raw = read_jsonl(args.raw_receipts)
    generated = generation_rows(args.generation_root)
    expected = {"pass8": (0, 8), "pass16": (8, 8), "pass32": (16, 16)}
    offset, count = expected[args.stage]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in raw:
        problem_id = row_id(receipt)
        local_attempt = int(receipt["attempt_index"])
        if problem_id not in pool_ids:
            raise CycleContractError(f"raw receipt {problem_id} is outside stage pool")
        key = (problem_id, local_attempt)
        generation = generated.get(key)
        if generation is None:
            raise CycleContractError(f"raw receipt {key} has no generation payload")
        if generation.get("model_sums_sha256") != identity.old_policy_sha256:
            raise CycleContractError(f"generated attempt {key} came from another policy")
        for field in ("generation_payload_sha256", "statement_hash"):
            if receipt.get(field) != generation.get(field):
                raise CycleContractError(f"{key}: {field} differs between generation and receipt")
        value = dict(receipt)
        value["stage_local_attempt_index"] = local_attempt
        value["attempt_index"] = offset + local_attempt
        value["rollout_stage"] = args.stage
        value["rollout_cycle_id"] = identity.cycle_id
        value["old_policy_sha256"] = identity.old_policy_sha256
        value["selection_manifest_sha256"] = identity.selection_manifest_sha256
        value["reward_source"] = "screen_selection_only"
        grouped[problem_id].append(value)
    if set(grouped) != pool_ids:
        raise CycleContractError("receipts do not cover the exact stage pool")
    if any(
        len(rows) != count
        or {int(row["stage_local_attempt_index"]) for row in rows} != set(range(count))
        for rows in grouped.values()
    ):
        raise CycleContractError(f"{args.stage} needs exact local attempts 0..{count - 1}")
    bound = [row for problem_id in sorted(grouped) for row in sorted(grouped[problem_id], key=lambda row: int(row["attempt_index"]))]
    write_jsonl(args.output, bound)
    report = {
        "schema": "dynamic_grpo_bound_receipts_v2",
        "stage": args.stage,
        "cycle_id": identity.cycle_id,
        "problem_count": len(grouped),
        "receipt_count": len(bound),
        "old_policy_sha256": identity.old_policy_sha256,
        "selection_manifest_sha256": identity.selection_manifest_sha256,
        "raw_receipts_sha256": sha256_file(args.raw_receipts),
        "bound_receipts_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def plan_topup(args: argparse.Namespace) -> None:
    manifest, identity = load_manifest(args.cycle_dir)
    cfg = DynamicCycleConfig(**manifest["config"])
    source = {row_id(row): row for row in read_jsonl(args.cycle_dir / "screen_pool.jsonl")}
    receipts = []
    for path in args.bound_receipts:
        receipts.extend(read_jsonl(path))
    validate_receipt_set(receipts, identity, reward_source="screen_selection_only")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[row_id(row)].append(row)
    if args.stage == "pass16":
        required_indices = set(range(8))
        limit = cfg.pass16_topup_limit
    else:
        required_indices = set(range(16))
        limit = cfg.pass32_topup_limit
    eligible = []
    for problem_id, rows in grouped.items():
        indices = {int(row["attempt_index"]) for row in rows}
        if indices != required_indices:
            continue
        if not any(is_success(row) for row in rows):
            eligible.append(problem_id)
    eligible.sort(key=lambda problem_id: deterministic_key(args.seed, args.stage, problem_id))
    chosen = eligible[:limit]
    output_rows = []
    for problem_id in chosen:
        row = dict(source[problem_id])
        row["repeat"] = 1
        row["dynamic_topup"] = {
            "cycle_id": identity.cycle_id,
            "old_policy_sha256": identity.old_policy_sha256,
            "selection_manifest_sha256": identity.selection_manifest_sha256,
            "stage": args.stage,
        }
        output_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pool = args.output_dir / f"{args.stage}_pool.jsonl"
    write_jsonl(pool, output_rows)
    report = {
        "schema": "dynamic_grpo_topup_plan_v2",
        "cycle_id": identity.cycle_id,
        "stage": args.stage,
        "eligible_zero_problem_count": len(eligible),
        "selected_problem_count": len(chosen),
        "selection_limit": limit,
        "pool_sha256": sha256_file(pool),
        "bound_receipt_sha256": [sha256_file(path) for path in args.bound_receipts],
    }
    (args.output_dir / "TOPUP_FROZEN.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    manifest, identity = load_manifest(args.cycle_dir)
    cfg = DynamicCycleConfig(**manifest["config"])
    receipts = []
    for path in args.bound_receipts:
        receipts.extend(read_jsonl(path))
    schedule, summary = build_cycle_schedule(
        selected_source_rows=read_jsonl(args.cycle_dir / "screen_pool.jsonl"),
        receipts=receipts,
        identity=identity,
        config=cfg,
        seed=args.seed,
    )
    report = write_cycle_artifacts(args.output_dir, schedule, summary)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[row_id(row)].append(row)
    # History is an append-style selection signal only.  Its values are never
    # copied into the online reward fields frozen in the schedule.
    prior_rows = read_jsonl(args.history) if args.history else []
    history = {str(row["problem_id"]): dict(row) for row in prior_rows}
    for problem_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["attempt_index"]))
        successes = sum(is_success(row) for row in ordered)
        attempts = len(ordered)
        previous = history.get(problem_id, {})
        snapshots = list(previous.get("rollout_history", []))
        snapshots.append(
            {
                "cycle_id": identity.cycle_id,
                "old_policy_sha256": identity.old_policy_sha256,
                "successes": successes,
                "attempts": attempts,
                "success_rate": successes / attempts,
            }
        )
        historical_successes = sum(int(row["successes"]) for row in snapshots)
        historical_attempts = sum(int(row["attempts"]) for row in snapshots)
        route = next(
            row["dynamic_cycle"]["screen_route"]
            for row in schedule
            if row_id(row) == problem_id
        ) if any(row_id(row) == problem_id for row in schedule) else (
            "mastered" if sum(is_success(row) for row in ordered[:8]) >= 5
            else "hard_bank" if attempts == 32 and successes == 0
            else "unresolved_zero" if successes == 0
            else "active_after_topup"
        )
        history[problem_id] = {
            "problem_id": problem_id,
            "rollout_history": snapshots,
            "history_visits": len(snapshots),
            "history_successes": historical_successes,
            "history_attempts": historical_attempts,
            "history_mean_success_rate": sum(float(row["success_rate"]) for row in snapshots) / len(snapshots),
            "rollout_priority": history_rollout_priority(
                historical_successes=historical_successes,
                historical_attempts=historical_attempts,
            ),
            "last_route": route,
            "last_cycle_id": identity.cycle_id,
            "last_old_policy_sha256": identity.old_policy_sha256,
        }
    history_path = args.output_dir / "rollout_history.jsonl"
    write_jsonl(history_path, [history[key] for key in sorted(history)])
    report["rollout_history_sha256"] = sha256_file(history_path)
    (args.output_dir / "CYCLE_SCHEDULE_FROZEN.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source-dataset", type=Path, required=True)
    freeze.add_argument("--history", type=Path)
    freeze.add_argument("--cycle-id", required=True)
    freeze.add_argument("--old-policy-sha256", required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20260909)
    freeze.set_defaults(func=freeze_cycle)

    bind = commands.add_parser("bind-receipts")
    bind.add_argument("--cycle-dir", type=Path, required=True)
    bind.add_argument("--stage", choices=("pass8", "pass16", "pass32"), required=True)
    bind.add_argument("--stage-pool", type=Path, required=True)
    bind.add_argument("--generation-root", type=Path, required=True)
    bind.add_argument("--raw-receipts", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.set_defaults(func=bind_receipts)

    topup = commands.add_parser("plan-topup")
    topup.add_argument("--cycle-dir", type=Path, required=True)
    topup.add_argument("--stage", choices=("pass16", "pass32"), required=True)
    topup.add_argument("--bound-receipts", type=Path, nargs="+", required=True)
    topup.add_argument("--output-dir", type=Path, required=True)
    topup.add_argument("--seed", type=int, default=20260909)
    topup.set_defaults(func=plan_topup)

    final = commands.add_parser("finalize")
    final.add_argument("--cycle-dir", type=Path, required=True)
    final.add_argument("--bound-receipts", type=Path, nargs="+", required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--history", type=Path)
    final.add_argument("--seed", type=int, default=20260909)
    final.set_defaults(func=finalize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
