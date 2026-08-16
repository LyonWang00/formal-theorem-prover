#!/usr/bin/env python3
"""Aggregate the gated LD-easy pilot and make auditable promotion decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_wb_ld_small_sft_ablation import (
    behavior,
    mcnemar,
    paired_bootstrap,
    solve_vectors,
)


MODEL_MANIFESTS = {
    "LDE-A2-WB100": "A2_WB1000",
    "LDE-B2-LD10": "B2_WB900_LDE100",
    "LDE-C2-LD20": "C2_WB800_LDE200",
}
CORE_DATASETS = ("wb_gate150", "ld_easy_holdout64", "monitor64", "ld_hard128")
FULL_DATASETS = ("full500", "strict_unseen200")


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


def active_models(root: Path) -> tuple[str, ...]:
    inventory = read_json(root / "audit/input_inventory.json")
    return (
        "M0-ZERO",
        *(
            model
            for model, manifest in MODEL_MANIFESTS.items()
            if manifest in inventory
        ),
    )


def metric_row(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    generations = read_jsonl(path / "generations.jsonl")
    attempts = read_jsonl(path / "attempts.jsonl")
    if not generations or not attempts:
        raise FileNotFoundError(f"incomplete pilot evaluation: {path}")
    summary = read_json(path / "benchmark_summary.json")
    metrics = {
        "pass_at_1": float(summary["pass_at"]["pass@1"]),
        "pass_at_2": float(summary["pass_at"]["pass@2"]),
        "pass_at_4": float(summary["pass_at"]["pass@4"]),
        "solved": int(summary["successes"]),
        "statements": int(summary["num_benchmark_samples"]),
        "candidate_success_rate": (
            sum(bool(row.get("success")) for row in attempts) / len(attempts)
        ),
        "generation_seconds": float(summary.get("generation_seconds") or 0),
        "verification_seconds": float(summary.get("verification_seconds") or 0),
        "pantograph_worker_restart_count": int(
            summary.get("pantograph_worker_restart_count") or 0
        ),
    }
    return metrics, behavior(generations, attempts), solve_vectors(attempts)


def core_promotion(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    candidate_behavior: dict[str, Any],
    baseline_behavior: dict[str, Any],
) -> dict[str, Any]:
    easy_gain = (
        candidate_metrics["ld_easy_holdout64"]["solved"]
        - baseline_metrics["ld_easy_holdout64"]["solved"]
    )
    wb_drop = (
        baseline_metrics["wb_gate150"]["solved"]
        - candidate_metrics["wb_gate150"]["solved"]
    )
    monitor_drop = (
        baseline_metrics["monitor64"]["solved"]
        - candidate_metrics["monitor64"]["solved"]
    )
    checks = {
        "ld_easy_clear_gain": easy_gain >= 1,
        "wb_drop_at_most_2": wb_drop <= 2,
        "monitor_drop_at_most_1": monitor_drop <= 1,
        "repetition_not_clearly_worse": (
            candidate_behavior["ld_easy_holdout64"]["repetition_ratio"]
            <= baseline_behavior["ld_easy_holdout64"]["repetition_ratio"] + 0.02
        ),
        "length_finish_not_clearly_worse": (
            candidate_behavior["ld_easy_holdout64"]["length_finish_ratio"]
            <= baseline_behavior["ld_easy_holdout64"]["length_finish_ratio"] + 0.02
        ),
        "proof_extraction_not_worse": (
            candidate_behavior["ld_easy_holdout64"][
                "proof_extraction_success_rate"
            ]
            >= baseline_behavior["ld_easy_holdout64"][
                "proof_extraction_success_rate"
            ]
            - 0.01
        ),
        "format_validity_not_worse": (
            candidate_behavior["ld_easy_holdout64"]["format_validity_rate"]
            >= baseline_behavior["ld_easy_holdout64"]["format_validity_rate"] - 0.01
        ),
    }
    return {
        "easy_solved_gain": easy_gain,
        "wb_solved_drop": wb_drop,
        "monitor_solved_drop": monitor_drop,
        "checks": checks,
        "promoted": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/pilot_sft"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    models = active_models(root)
    all_metrics: dict[str, dict[str, Any]] = {}
    all_behavior: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in (*CORE_DATASETS, *FULL_DATASETS)
    }
    missing: list[str] = []
    for model in models:
        all_metrics[model] = {}
        all_behavior[model] = {}
        for dataset in CORE_DATASETS:
            path = root / "evaluation" / model / dataset
            try:
                metrics, diagnostics, vector = metric_row(path)
            except FileNotFoundError:
                missing.append(f"{model}/{dataset}")
                continue
            all_metrics[model][dataset] = metrics
            all_behavior[model][dataset] = diagnostics
            vectors[dataset][model] = vector
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"incomplete core pilot evaluations: {missing}")

    paired: dict[str, Any] = {}
    for dataset in CORE_DATASETS:
        for candidate in models[1:]:
            if candidate not in vectors[dataset] or "M0-ZERO" not in vectors[dataset]:
                continue
            key = f"{dataset}:{candidate}_vs_M0-ZERO"
            paired[key] = {
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset]["M0-ZERO"],
                    seed=20260803 + len(paired),
                ),
                **mcnemar(
                    vectors[dataset][candidate],
                    vectors[dataset]["M0-ZERO"],
                ),
            }
        for candidate in models[1:]:
            if candidate == "LDE-A2-WB100":
                continue
            if (
                candidate not in vectors[dataset]
                or "LDE-A2-WB100" not in vectors[dataset]
            ):
                continue
            key = f"{dataset}:{candidate}_vs_LDE-A2-WB100"
            paired[key] = {
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset]["LDE-A2-WB100"],
                    seed=20260803 + len(paired),
                ),
                **mcnemar(
                    vectors[dataset][candidate],
                    vectors[dataset]["LDE-A2-WB100"],
                ),
            }

    promotions: dict[str, Any] = {}
    if all(len(all_metrics.get(model, {})) == len(CORE_DATASETS) for model in models):
        for candidate in models[1:]:
            if candidate == "LDE-A2-WB100":
                continue
            promotions[candidate] = core_promotion(
                all_metrics[candidate],
                all_metrics["LDE-A2-WB100"],
                all_behavior[candidate],
                all_behavior["LDE-A2-WB100"],
            )

    full_metrics: dict[str, dict[str, Any]] = {}
    full_behavior: dict[str, dict[str, Any]] = {}
    for model in models:
        for dataset in FULL_DATASETS:
            path = root / "evaluation" / model / dataset
            if not (path / "benchmark_summary.json").is_file():
                continue
            metrics, diagnostics, vector = metric_row(path)
            full_metrics.setdefault(model, {})[dataset] = metrics
            full_behavior.setdefault(model, {})[dataset] = diagnostics
            vectors[dataset][model] = vector

    write_json(root / "comparisons/core_metrics.json", all_metrics)
    write_json(root / "comparisons/core_behavior.json", all_behavior)
    write_json(root / "comparisons/paired_statistics.json", paired)
    write_json(root / "comparisons/promotion_decisions.json", promotions)
    write_json(root / "comparisons/full_metrics.json", full_metrics)
    write_json(root / "comparisons/full_behavior.json", full_behavior)
    write_json(
        root / "comparisons/analysis_status.json",
        {
            "missing_core": missing,
            "core_complete": not missing,
            "promoted_models": [
                model
                for model, decision in promotions.items()
                if decision["promoted"]
            ],
        },
    )
    print(json.dumps({"missing": missing, "promotions": promotions}, indent=2))


if __name__ == "__main__":
    main()
