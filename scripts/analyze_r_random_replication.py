#!/usr/bin/env python3
"""Aggregate multi-seed replay metrics and make the pre-registered M0 decision."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.analyze_anchor_ratio_eos_fixed import metric_row
from scripts.analyze_wb_ld_small_sft_ablation import mcnemar, paired_bootstrap
from scripts.prepare_r_random_replication import NEW_MODELS, OUTPUT
from scripts.run_r_random_replication_evaluation import (
    ADDON_LABEL,
    MODELS,
    S1_LABEL,
    STRICT,
    read_json,
    write_json,
)


RANDOM_LABELS = (
    S1_LABEL,
    "R-RANDOM-S2-SEED-20261502",
    "R-RANDOM-S3-SEED-20261503",
)
CORE_DATASETS = (
    "wb_retention150",
    "wb_unseen150",
    "ld_easy64",
    "monitor64",
    "hard_ld128",
)
PAIRWISE = (
    (S1_LABEL, ADDON_LABEL),
    ("R-RANDOM-S2-SEED-20261502", ADDON_LABEL),
    ("R-RANDOM-S3-SEED-20261503", ADDON_LABEL),
    ("R-RANDOM-S2-SEED-20261502", S1_LABEL),
    ("R-RANDOM-S3-SEED-20261503", S1_LABEL),
    ("R-RANDOM-S3-SEED-20261503", "R-RANDOM-S2-SEED-20261502"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def eval_metrics(root: Path, label: str) -> dict[str, float]:
    row = read_json(root / "evaluation/eval160" / label / "eval_metrics.json")
    return {
        "eval_loss": float(row["eval_loss"]),
        "eval_token_accuracy": float(
            row.get("eval_mean_token_accuracy")
            or row.get("eval_token_accuracy")
            or 0
        ),
    }


def scalar_stats(values: dict[str, float], *, seed: int) -> dict[str, Any]:
    numbers = list(values.values())
    rng = random.Random(seed)
    draws = 20000
    bootstrap = sorted(
        statistics.mean(
            numbers[rng.randrange(len(numbers))] for _ in numbers
        )
        for _ in range(draws)
    )
    return {
        "values": values,
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "standard_deviation": statistics.pstdev(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
        "bootstrap_draws": draws,
        "bootstrap_95_ci": [
            bootstrap[round((draws - 1) * 0.025)],
            bootstrap[round((draws - 1) * 0.975)],
        ],
    }


def paired_statistics(
    metrics: dict[str, Any],
    vectors: dict[str, dict[str, dict[str, int]]],
    datasets: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_seed = seed
    for candidate, baseline in comparisons:
        for dataset in datasets:
            key = f"{dataset}:{candidate}_vs_{baseline}"
            result[key] = {
                "candidate": candidate,
                "baseline": baseline,
                "dataset": dataset,
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset][baseline],
                    seed=current_seed,
                ),
                **mcnemar(
                    vectors[dataset][candidate],
                    vectors[dataset][baseline],
                ),
                "candidate_solved": metrics[candidate][dataset]["solved"],
                "baseline_solved": metrics[baseline][dataset]["solved"],
                "delta_solved": (
                    metrics[candidate][dataset]["solved"]
                    - metrics[baseline][dataset]["solved"]
                ),
            }
            current_seed += 1
    return result


def load_metrics(
    root: Path, datasets: tuple[str, ...], *, allow_missing: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, int]]]]:
    metrics: dict[str, Any] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in datasets
    }
    canary = read_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json"
    )
    for label in MODELS:
        metrics[label] = {
            "eval160": eval_metrics(root, label),
            "behavior_canary": canary[label],
        }
        for dataset in datasets:
            evaluation_dir = root / "evaluation" / dataset / label
            if allow_missing and not (
                evaluation_dir / "benchmark_summary.json"
            ).is_file():
                metrics[label][dataset] = {
                    "status": (
                        "NOT_EVALUATED_CHECKPOINT_EQUIVALENCE_FAILED"
                    )
                }
                continue
            row, behavior, vector = metric_row(
                evaluation_dir
            )
            metrics[label][dataset] = {**row, "behavior": behavior}
            vectors[dataset][label] = vector
    return metrics, vectors


def write_runtime_files(root: Path, metrics: dict[str, Any]) -> None:
    training_rows = []
    for label in NEW_MODELS:
        summary = read_json(root / "training" / label / "training_summary.json")
        training_rows.append(
            {
                "model": label,
                "runtime_seconds": summary["runtime_seconds"],
                "optimizer_steps": summary["optimizer_steps"],
                "gpu_peak_allocated_bytes": summary[
                    "gpu_peak_allocated_bytes"
                ],
                "gpu_peak_reserved_bytes": summary["gpu_peak_reserved_bytes"],
                "process_max_rss_kib": summary["process_max_rss_kib"],
            }
        )
    generation_rows = []
    verification_rows = []
    for label in MODELS:
        for dataset in (*CORE_DATASETS, *STRICT):
            row = metrics[label][dataset]
            if "solved" not in row:
                continue
            generation_rows.append(
                {
                    "model": label,
                    "dataset": dataset,
                    "seconds": row["generation_seconds"],
                    "statements": row["statements"],
                    "candidates": row["behavior"]["candidates"],
                }
            )
            summary = read_json(
                root
                / "evaluation"
                / dataset
                / label
                / "benchmark_summary.json"
            )
            verification_rows.append(
                {
                    "model": label,
                    "dataset": dataset,
                    "seconds": row["verification_seconds"],
                    "worker_restarts": row["worker_restarts"],
                    "cache_hits": int(summary.get("cache_hits") or 0),
                    "cache_misses": int(summary.get("cache_misses") or 0),
                    "timeout_attempts": int(
                        summary.get("timeout_attempts") or 0
                    ),
                }
            )
    for filename, rows in (
        ("training_resources.jsonl", training_rows),
        ("generation_resources.jsonl", generation_rows),
        ("verification_resources.jsonl", verification_rows),
    ):
        path = root / "runtime" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )


def comparison_report(
    core: dict[str, Any],
    strict: dict[str, Any],
    cross_seed: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# R-Random replay multi-seed comparison",
        "",
        "## Core evaluation",
        "",
        "| Model | Retention150 | WB unseen150 | LD-easy64 | Monitor64 | Hard-LD128 | eval loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in MODELS:
        lines.append(
            f"| {label} | {core[label]['wb_retention150']['solved']}/150 | "
            f"{core[label]['wb_unseen150']['solved']}/150 | "
            f"{core[label]['ld_easy64']['solved']}/64 | "
            f"{core[label]['monitor64']['solved']}/64 | "
            f"{core[label]['hard_ld128']['solved']}/128 | "
            f"{core[label]['eval160']['eval_loss']:.5f} |"
        )
    lines += [
        "",
        "## Strict evaluation",
        "",
        "| Model | Full500 Pass@4 | Strict-unseen200 Pass@4 |",
        "|---|---:|---:|",
    ]
    for label in MODELS:
        full = strict[label]["full500"]
        unseen = strict[label]["strict_unseen200"]
        if "solved" not in full or "solved" not in unseen:
            lines.append(
                f"| {label} | not evaluated (checkpoint equivalence gate) | "
                "not evaluated (checkpoint equivalence gate) |"
            )
        else:
            lines.append(
                f"| {label} | {full['solved']}/500 "
                f"({full['pass_at_4']:.2%}) | "
                f"{unseen['solved']}/200 "
                f"({unseen['pass_at_4']:.2%}) |"
            )
    lines += [
        "",
        "## Cross-seed decision",
        "",
        f"- Core stability gate: `{decision['core_stability']['passed']}`",
        f"- Strict generalization gate: `{decision['strict_stability']['passed']}`",
        f"- Random Replay replication: `{decision['random_replay_passed']}`",
        f"- Frozen experimental M0: `{decision['frozen_model_name']}`",
        f"- Selected source model: `{decision['selected_source_model']}`",
        f"- Additional random seeds required: `{decision['additional_seed_required']}`",
        "",
        "The selection ranks strict unseen first, then Full500, WB unseen, "
        "LD/Monitor transfer, retention, and eval loss. It does not select a "
        "manifest from retention alone.",
        "",
        "Cross-seed aggregate details are stored in "
        "`comparisons/cross_seed_statistics.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()

    core, core_vectors = load_metrics(root, CORE_DATASETS)
    strict, strict_vectors = load_metrics(
        root, tuple(STRICT), allow_missing=True
    )
    core_paired = paired_statistics(
        core, core_vectors, CORE_DATASETS, PAIRWISE, seed=20261800
    )
    strict_comparisons = tuple(
        (label, ADDON_LABEL)
        for label in RANDOM_LABELS
        if all(label in strict_vectors[dataset] for dataset in STRICT)
    )
    strict_paired = paired_statistics(
        strict,
        strict_vectors,
        tuple(STRICT),
        strict_comparisons,
        seed=20261900,
    )

    canary = read_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json"
    )
    cross_values: dict[str, dict[str, float]] = {}
    for dataset in CORE_DATASETS:
        cross_values[f"{dataset}_solved"] = {
            label: float(core[label][dataset]["solved"])
            for label in RANDOM_LABELS
        }
    cross_values["eval_loss"] = {
        label: core[label]["eval160"]["eval_loss"] for label in RANDOM_LABELS
    }
    cross_values["mean_output_tokens"] = {
        label: canary[label]["output_tokens"]["mean"] for label in RANDOM_LABELS
    }
    cross_values["length_finish_ratio"] = {
        label: canary[label]["length_finish_ratio"] for label in RANDOM_LABELS
    }
    cross_values["repetition_ratio"] = {
        label: canary[label]["repetition_ratio"] for label in RANDOM_LABELS
    }
    cross_values["full500_solved"] = {
        label: float(strict[label]["full500"]["solved"])
        for label in RANDOM_LABELS
        if "solved" in strict[label]["full500"]
    }
    cross_values["strict_unseen200_solved"] = {
        label: float(strict[label]["strict_unseen200"]["solved"])
        for label in RANDOM_LABELS
        if "solved" in strict[label]["strict_unseen200"]
    }
    cross_seed = {
        name: scalar_stats(values, seed=20262000 + index)
        for index, (name, values) in enumerate(cross_values.items())
    }

    addon = {
        dataset: int(core[ADDON_LABEL][dataset]["solved"])
        for dataset in CORE_DATASETS
    }
    unseen = list(cross_values["wb_unseen150_solved"].values())
    ld_easy = list(cross_values["ld_easy64_solved"].values())
    monitor = list(cross_values["monitor64_solved"].values())
    retention = list(cross_values["wb_retention150_solved"].values())
    behavior_pass = all(canary[label]["passed"] for label in RANDOM_LABELS)
    core_checks = {
        "wb_unseen_at_least_two_ge_addon_minus_2": (
            sum(value >= addon["wb_unseen150"] - 2 for value in unseen) >= 2
        ),
        "wb_unseen_mean_ge_addon_minus_1": (
            statistics.mean(unseen) >= addon["wb_unseen150"] - 1
        ),
        "ld_easy_at_least_two_ge_addon": (
            sum(value >= addon["ld_easy64"] for value in ld_easy) >= 2
        ),
        "ld_easy_mean_ge_addon": (
            statistics.mean(ld_easy) >= addon["ld_easy64"]
        ),
        "monitor_at_least_two_ge_addon_minus_1": (
            sum(value >= addon["monitor64"] - 1 for value in monitor) >= 2
        ),
        "retention_mean_gt_addon": (
            statistics.mean(retention) > addon["wb_retention150"]
        ),
        "all_behavior_gates_passed": behavior_pass,
    }
    core_stability = {
        "passed": all(core_checks.values()),
        "checks": core_checks,
        "addon_reference": addon,
    }

    addon_full = int(strict[ADDON_LABEL]["full500"]["solved"])
    addon_strict = int(strict[ADDON_LABEL]["strict_unseen200"]["solved"])
    full_values = list(cross_values["full500_solved"].values())
    strict_values = list(cross_values["strict_unseen200_solved"].values())
    strict_checks = {
        "all_three_random_seeds_evaluated": (
            len(full_values) == 3 and len(strict_values) == 3
        ),
        "full500_majority_within_5": (
            sum(value >= addon_full - 5 for value in full_values) >= 2
        ),
        "full500_mean_within_5": (
            statistics.mean(full_values) >= addon_full - 5
        ),
        "strict200_majority_within_3": (
            sum(value >= addon_strict - 3 for value in strict_values) >= 2
        ),
        "strict200_mean_within_3": (
            statistics.mean(strict_values) >= addon_strict - 3
        ),
    }
    strict_stability = {
        "passed": all(strict_checks.values()),
        "checks": strict_checks,
        "addon_reference": {
            "full500": addon_full,
            "strict_unseen200": addon_strict,
        },
    }

    manifest_audit = read_json(root / "audit/manifest_overlap_audit.json")
    extreme_token_confound = bool(
        manifest_audit["token_exposure_confound"]["flagged"]
    )
    random_passed = (
        core_stability["passed"]
        and strict_stability["passed"]
        and not extreme_token_confound
    )
    if random_passed:
        selected = max(
            RANDOM_LABELS,
            key=lambda label: (
                strict[label]["strict_unseen200"]["solved"],
                strict[label]["full500"]["solved"],
                core[label]["wb_unseen150"]["solved"],
                core[label]["ld_easy64"]["solved"]
                + core[label]["monitor64"]["solved"],
                core[label]["wb_retention150"]["solved"],
                -core[label]["eval160"]["eval_loss"],
            ),
        )
        frozen_name = "M0-RANDOM-REPLAY-FROZEN"
    else:
        selected = ADDON_LABEL
        frozen_name = "M0-ADDON-B-FROZEN"
    decision = {
        "status": "FINAL_SELECTION_FROZEN",
        "random_replay_passed": random_passed,
        "core_stability": core_stability,
        "strict_stability": strict_stability,
        "extreme_token_exposure_confound": extreme_token_confound,
        "selected_source_model": selected,
        "frozen_model_name": frozen_name,
        "selection_priority": [
            "strict_unseen200",
            "full500",
            "wb_unseen150",
            "ld_easy64_plus_monitor64",
            "wb_retention150",
            "eval_loss",
        ],
        "additional_seed_required": False,
        "allow_second_stage_sft_ratio_experiment": True,
        "clean_m0_remains_production_default": True,
        "experimental_m0_does_not_overwrite_clean_m0": True,
    }

    write_json(root / "comparisons/core_metrics.json", core)
    write_json(root / "comparisons/paired_statistics.json", core_paired)
    write_json(root / "comparisons/cross_seed_statistics.json", cross_seed)
    write_json(root / "comparisons/strict_evaluation_metrics.json", strict)
    write_json(root / "comparisons/strict_paired_statistics.json", strict_paired)
    write_json(root / "comparisons/behavior_analysis.json", canary)
    write_json(root / "comparisons/promotion_decision.json", decision)
    report = comparison_report(core, strict, cross_seed, decision)
    (root / "comparisons/comparison_report.md").write_text(
        report, encoding="utf-8"
    )
    pointer = root / "checkpoints" / frozen_name / "FROZEN_POINTER.json"
    write_json(
        pointer,
        {
            "frozen_model_name": frozen_name,
            "selected_source_model": selected,
            "selection_report": str(
                root / "comparisons/promotion_decision.json"
            ),
            "does_not_overwrite_clean_m0": True,
        },
    )
    runtime_metrics = {
        label: {**core[label], **strict[label]} for label in MODELS
    }
    write_runtime_files(root, runtime_metrics)
    (root / "final_report.md").write_text(
        report
        + "\n## Interpretation\n\n"
        + (
            "Random Replay passed the pre-registered cross-seed core and strict "
            "generalization gates. The selected manifest is frozen as an "
            "experimental M0; CLEAN-M0 remains the production default.\n"
            if random_passed
            else
            "Random Replay did not pass every pre-registered stability gate. "
            "The experiment therefore falls back to ADDON-B as the frozen "
            "experimental M0; CLEAN-M0 remains the production default.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
