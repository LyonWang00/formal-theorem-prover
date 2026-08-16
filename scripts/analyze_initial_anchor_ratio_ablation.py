#!/usr/bin/env python3
"""Aggregate metrics and paired statistics for the initial-anchor ratio study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analyze_wb_ld_small_sft_ablation import (
    behavior,
    mcnemar,
    paired_bootstrap,
    solve_vectors,
)


MODELS = (
    "BASE-ZERO",
    "CLEAN-M0",
    "ANCHOR-A0",
    "ANCHOR-A5",
    "ANCHOR-A10",
    "ANCHOR-A20",
)
CORE_DATASETS = ("wb_gate150", "ld_easy_holdout64", "monitor64", "ld_hard128")
FULL_DATASETS = ("full500", "strict_unseen200")
COMPARISONS = (
    ("ANCHOR-A5", "ANCHOR-A0"),
    ("ANCHOR-A10", "ANCHOR-A0"),
    ("ANCHOR-A20", "ANCHOR-A0"),
    ("ANCHOR-A10", "ANCHOR-A5"),
    ("ANCHOR-A20", "ANCHOR-A10"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def metric_row(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    generations = read_jsonl(path / "generations.jsonl")
    attempts = read_jsonl(path / "attempts.jsonl")
    if not generations or not attempts:
        raise FileNotFoundError(f"incomplete evaluation: {path}")
    summary = read_json(path / "benchmark_summary.json")
    metrics = {
        "pass_at_1": float(summary["pass_at"]["pass@1"]),
        "pass_at_2": float(summary["pass_at"]["pass@2"]),
        "pass_at_4": float(summary["pass_at"]["pass@4"]),
        "solved": int(summary["successes"]),
        "statements": int(summary["num_benchmark_samples"]),
        "candidate_success_rate": sum(
            bool(row.get("success")) for row in attempts
        )
        / len(attempts),
        "generation_seconds": float(summary.get("generation_seconds") or 0),
        "verification_seconds": float(summary.get("verification_seconds") or 0),
        "worker_restarts": int(
            summary.get("pantograph_worker_restart_count") or 0
        ),
    }
    return metrics, behavior(generations, attempts), solve_vectors(attempts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/initial_anchor_ratio_ablation"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    metrics: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    behaviors: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in (*CORE_DATASETS, *FULL_DATASETS)
    }
    missing: list[str] = []

    for model in MODELS:
        eval_loss = root / "evaluation" / model / "teacher_forcing_eval160/eval_metrics.json"
        if eval_loss.is_file():
            metrics[model]["teacher_forcing_eval160"] = read_json(eval_loss)
        else:
            missing.append(f"{model}/teacher_forcing_eval160")
        for dataset in CORE_DATASETS:
            path = root / "evaluation" / model / dataset
            try:
                row, diagnostics, vector = metric_row(path)
            except FileNotFoundError:
                missing.append(f"{model}/{dataset}")
                continue
            metrics[model][dataset] = row
            behaviors[model][dataset] = diagnostics
            vectors[dataset][model] = vector
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"incomplete core evaluations: {missing}")

    paired: dict[str, Any] = {}
    for dataset in CORE_DATASETS:
        for candidate, baseline in COMPARISONS:
            if candidate not in vectors[dataset] or baseline not in vectors[dataset]:
                continue
            key = f"{dataset}:{candidate}_vs_{baseline}"
            paired[key] = {
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset][baseline],
                    seed=20261020 + len(paired),
                ),
                **mcnemar(vectors[dataset][candidate], vectors[dataset][baseline]),
            }

    decisions: dict[str, Any] = {}
    eligible: list[tuple[str, int, int, int]] = []
    if all(len(metrics[model]) == len(CORE_DATASETS) + 1 for model in MODELS):
        baseline = "ANCHOR-A0"
        for candidate in ("ANCHOR-A5", "ANCHOR-A10", "ANCHOR-A20"):
            easy_gain = (
                metrics[candidate]["ld_easy_holdout64"]["solved"]
                - metrics[baseline]["ld_easy_holdout64"]["solved"]
            )
            wb_drop = (
                metrics[baseline]["wb_gate150"]["solved"]
                - metrics[candidate]["wb_gate150"]["solved"]
            )
            monitor_drop = (
                metrics[baseline]["monitor64"]["solved"]
                - metrics[candidate]["monitor64"]["solved"]
            )
            checks = {
                "ld_easy_positive_effect": easy_gain >= 1,
                "wb_drop_at_most_2": wb_drop <= 2,
                "monitor_drop_at_most_1": monitor_drop <= 1,
                "repetition_not_clearly_worse": (
                    behaviors[candidate]["ld_easy_holdout64"]["repetition_ratio"]
                    <= behaviors[baseline]["ld_easy_holdout64"]["repetition_ratio"]
                    + 0.02
                ),
                "length_finish_not_clearly_worse": (
                    behaviors[candidate]["ld_easy_holdout64"]["length_finish_ratio"]
                    <= behaviors[baseline]["ld_easy_holdout64"]["length_finish_ratio"]
                    + 0.02
                ),
            }
            qualifies = all(checks.values())
            decisions[candidate] = {
                "ld_easy_solved_gain": easy_gain,
                "wb_solved_drop": wb_drop,
                "monitor_solved_drop": monitor_drop,
                "checks": checks,
                "qualifies_for_full_evaluation": qualifies,
                "selected_for_full_evaluation": False,
            }
            if qualifies:
                eligible.append((candidate, easy_gain, -wb_drop, -monitor_drop))
        for candidate, *_ in sorted(
            eligible, key=lambda item: item[1:], reverse=True
        )[:2]:
            decisions[candidate]["selected_for_full_evaluation"] = True

    full_metrics: dict[str, dict[str, Any]] = {}
    full_behaviors: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for dataset in FULL_DATASETS:
            path = root / "evaluation" / model / dataset
            if not (path / "benchmark_summary.json").is_file():
                continue
            row, diagnostics, vector = metric_row(path)
            full_metrics.setdefault(model, {})[dataset] = row
            full_behaviors.setdefault(model, {})[dataset] = diagnostics
            vectors[dataset][model] = vector

    checkpoint_hashes = {}
    training_runtime = {}
    for model, arm in {
        "ANCHOR-A0": "A0_WB3000_LD0",
        "ANCHOR-A5": "A5_WB2750_LD250",
        "ANCHOR-A10": "A10_WB2500_LD500",
        "ANCHOR-A20": "A20_WB2000_LD1000",
    }.items():
        checkpoint = root / "checkpoints" / arm / "best"
        if checkpoint.is_dir():
            checkpoint_hashes[model] = {
                "path": str(checkpoint),
                "sha256": directory_sha256(checkpoint),
            }
        summary = root / "training" / arm / "training_summary.json"
        if summary.is_file():
            payload = read_json(summary)
            training_runtime[model] = {
                "runtime_seconds": payload.get("runtime_seconds"),
                "gpu_peak_allocated_bytes": payload.get(
                    "gpu_peak_allocated_bytes"
                ),
                "gpu_peak_reserved_bytes": payload.get(
                    "gpu_peak_reserved_bytes"
                ),
                "process_max_rss_kib": payload.get("process_max_rss_kib"),
            }
    evaluated_rows = [
        row
        for model_metrics in metrics.values()
        for name, row in model_metrics.items()
        if name in CORE_DATASETS
    ] + [
        row
        for model_metrics in full_metrics.values()
        for row in model_metrics.values()
    ]
    runtime_summary = {
        "training": training_runtime,
        "training_seconds_total": sum(
            float(row.get("runtime_seconds") or 0)
            for row in training_runtime.values()
        ),
        "generation_seconds_total": sum(
            float(row.get("generation_seconds") or 0) for row in evaluated_rows
        ),
        "verification_seconds_total": sum(
            float(row.get("verification_seconds") or 0) for row in evaluated_rows
        ),
        "pantograph_worker_restarts_total": sum(
            int(row.get("worker_restarts") or 0) for row in evaluated_rows
        ),
        "gpu_peak_reserved_bytes": max(
            (
                int(row.get("gpu_peak_reserved_bytes") or 0)
                for row in training_runtime.values()
            ),
            default=0,
        ),
        "process_max_rss_kib": max(
            (
                int(row.get("process_max_rss_kib") or 0)
                for row in training_runtime.values()
            ),
            default=0,
        ),
    }

    write_json(root / "comparisons/core_metrics.json", metrics)
    write_json(root / "comparisons/core_behavior.json", behaviors)
    write_json(root / "comparisons/paired_statistics.json", paired)
    write_json(root / "comparisons/promotion_decisions.json", decisions)
    write_json(root / "comparisons/full_metrics.json", full_metrics)
    write_json(root / "comparisons/full_behavior.json", full_behaviors)
    write_json(root / "audit/checkpoint_hashes.json", checkpoint_hashes)
    write_json(root / "runtime/runtime_summary.json", runtime_summary)
    write_json(
        root / "comparisons/analysis_status.json",
        {
            "missing_core": missing,
            "core_complete": not missing,
            "selected_for_full_evaluation": [
                model
                for model, decision in decisions.items()
                if decision["selected_for_full_evaluation"]
            ],
        },
    )
    print(json.dumps({"missing": missing, "promotion_decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
