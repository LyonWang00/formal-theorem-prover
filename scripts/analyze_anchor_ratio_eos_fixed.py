#!/usr/bin/env python3
"""Aggregate EOS-fixed anchor-ratio metrics, paired tests, and gates."""

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


NEW_MODELS = (
    "ANCHOR-A0-EOS-FIXED",
    "ANCHOR-A5-EOS-FIXED",
    "ANCHOR-A10-EOS-FIXED",
    "ANCHOR-A20-EOS-FIXED",
)
REFERENCES = (
    "CLEAN-M0",
    "R0-HISTORICAL",
    "R2-CURRENT-DATA-HIST-PIPELINE",
)
CORE = ("wb_gate150", "ld_easy64", "monitor64", "hard_ld128")
FULL = ("full500", "strict_unseen200")
COMPARISONS = (
    ("ANCHOR-A5-EOS-FIXED", "ANCHOR-A0-EOS-FIXED"),
    ("ANCHOR-A10-EOS-FIXED", "ANCHOR-A0-EOS-FIXED"),
    ("ANCHOR-A20-EOS-FIXED", "ANCHOR-A0-EOS-FIXED"),
    ("ANCHOR-A10-EOS-FIXED", "ANCHOR-A5-EOS-FIXED"),
    ("ANCHOR-A20-EOS-FIXED", "ANCHOR-A10-EOS-FIXED"),
    ("ANCHOR-A0-EOS-FIXED", "CLEAN-M0"),
    ("ANCHOR-A0-EOS-FIXED", "R0-HISTORICAL"),
    ("ANCHOR-A0-EOS-FIXED", "R2-CURRENT-DATA-HIST-PIPELINE"),
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


def metric_row(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    generations = read_jsonl(path / "generations.jsonl")
    attempts = read_jsonl(path / "attempts.jsonl")
    if not generations or not attempts:
        raise FileNotFoundError(path)
    summary = read_json(path / "benchmark_summary.json")
    pass_at = summary["pass_at"]
    metrics = {
        "pass_at_1": float(pass_at["pass@1"]),
        "pass_at_2": float(pass_at["pass@2"]),
        "pass_at_4": float(pass_at["pass@4"]),
        "solved": int(summary["successes"]),
        "statements": int(summary["num_benchmark_samples"]),
        "candidate_success_rate": sum(bool(row.get("success")) for row in attempts)
        / len(attempts),
        "generation_seconds": float(summary.get("generation_seconds") or 0),
        "verification_seconds": float(summary.get("verification_seconds") or 0),
        "worker_restarts": int(
            summary.get("pantograph_worker_restart_count") or 0
        ),
    }
    return metrics, behavior(generations, attempts), solve_vectors(attempts)


def reference_path(project: Path, name: str, dataset: str) -> Path | None:
    if name == "CLEAN-M0":
        mapping = {
            "wb_gate150": "wb_gate150",
            "ld_easy64": "ld_easy_holdout64",
            "monitor64": "monitor64",
            "hard_ld128": "ld_hard128",
        }
        return (
            project
            / "outputs/initial_anchor_ratio_ablation/evaluation/CLEAN-M0"
            / mapping[dataset]
        )
    if dataset == "hard_ld128":
        return None
    folder = {
        "R0-HISTORICAL": "R0-HISTORICAL",
        "R2-CURRENT-DATA-HIST-PIPELINE": (
            "R2-CURRENT-DATA-HIST-PIPELINE"
        ),
    }[name]
    mapping = {
        "wb_gate150": "wb_gate150",
        "ld_easy64": "ld_easy_holdout64",
        "monitor64": "monitor64",
    }
    return (
        project
        / "outputs/clean_m0_a0_reproduction_audit/evaluation/full"
        / folder
        / mapping[dataset]
    )


def eval_loss_path(project: Path, root: Path, name: str) -> Path | None:
    if name in NEW_MODELS:
        return root / "evaluation/eval160" / name / "eval_metrics.json"
    if name == "CLEAN-M0":
        return (
            project
            / "outputs/initial_anchor_ratio_ablation/evaluation/CLEAN-M0/"
            "teacher_forcing_eval160/eval_metrics.json"
        )
    folder = {
        "R0-HISTORICAL": "R0-HISTORICAL",
        "R2-CURRENT-DATA-HIST-PIPELINE": (
            "R2-CURRENT-DATA-HIST-PIPELINE"
        ),
    }[name]
    return (
        project
        / "outputs/clean_m0_a0_reproduction_audit/evaluation/full"
        / folder
        / "teacher_forcing_eval160/eval_metrics.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/anchor_ratio_eos_fixed"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()

    metrics: dict[str, dict[str, Any]] = {}
    behaviors: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in (*CORE, *FULL)
    }
    missing: list[str] = []
    for name in (*REFERENCES, *NEW_MODELS):
        metrics[name] = {}
        behaviors[name] = {}
        loss = eval_loss_path(project, root, name)
        if loss and loss.is_file():
            metrics[name]["eval160"] = read_json(loss)
        else:
            missing.append(f"{name}/eval160")
        for dataset in CORE:
            path = (
                root / "evaluation/core" / name / dataset
                if name in NEW_MODELS
                else reference_path(project, name, dataset)
            )
            if path is None:
                continue
            try:
                row, diagnostics, vector = metric_row(path)
            except FileNotFoundError:
                missing.append(f"{name}/{dataset}")
                continue
            metrics[name][dataset] = row
            behaviors[name][dataset] = diagnostics
            vectors[dataset][name] = vector
    new_missing = [
        item for item in missing if item.split("/", 1)[0] in NEW_MODELS
    ]
    if new_missing and not args.allow_incomplete:
        raise FileNotFoundError(f"incomplete EOS-fixed core: {new_missing}")

    paired: dict[str, Any] = {}
    for dataset in CORE:
        for candidate, baseline in COMPARISONS:
            if (
                candidate not in vectors[dataset]
                or baseline not in vectors[dataset]
            ):
                continue
            key = f"{dataset}:{candidate}_vs_{baseline}"
            paired[key] = {
                **paired_bootstrap(
                    vectors[dataset][candidate],
                    vectors[dataset][baseline],
                    seed=20261220 + len(paired),
                ),
                **mcnemar(vectors[dataset][candidate], vectors[dataset][baseline]),
            }

    a0 = "ANCHOR-A0-EOS-FIXED"
    references = ("R0-HISTORICAL", "R2-CURRENT-DATA-HIST-PIPELINE")
    reproduction: dict[str, Any] = {"status": "INCOMPLETE", "passed": False}
    if all(
        dataset in metrics[a0]
        for dataset in ("wb_gate150", "ld_easy64", "monitor64")
    ) and all(
        all(dataset in metrics[name] for dataset in ("wb_gate150", "monitor64"))
        for name in references
    ):
        wb_distance = min(
            abs(
                metrics[a0]["wb_gate150"]["solved"]
                - metrics[name]["wb_gate150"]["solved"]
            )
            for name in references
        )
        monitor_distance = min(
            abs(
                metrics[a0]["monitor64"]["solved"]
                - metrics[name]["monitor64"]["solved"]
            )
            for name in references
        )
        combined_generations: list[dict[str, Any]] = []
        combined_attempts: list[dict[str, Any]] = []
        for dataset in ("wb_gate150", "ld_easy64", "monitor64"):
            path = root / "evaluation/core" / a0 / dataset
            combined_generations.extend(read_jsonl(path / "generations.jsonl"))
            combined_attempts.extend(read_jsonl(path / "attempts.jsonl"))
        combined = behavior(combined_generations, combined_attempts)
        checks = {
            "wb_within_4_solved_of_r0_or_r2": wb_distance <= 4,
            "monitor_within_2_solved_of_r0_or_r2": monitor_distance <= 2,
            "ld_easy_nonzero": metrics[a0]["ld_easy64"]["solved"] > 0,
            "mean_tokens_below_80": combined["output_tokens"]["mean"] < 80,
            "length_finish_below_15pct": combined["length_finish_ratio"] < 0.15,
            "repetition_below_20pct": combined["repetition_ratio"] < 0.20,
        }
        reproduction = {
            "status": (
                "A0_REPRODUCTION_GATE_PASSED"
                if all(checks.values())
                else "A0_REPRODUCTION_GATE_FAILED"
            ),
            "passed": all(checks.values()),
            "checks": checks,
            "wb_nearest_reference_distance_solved": wb_distance,
            "monitor_nearest_reference_distance_solved": monitor_distance,
            "combined_behavior": combined,
        }

    decisions: dict[str, Any] = {}
    eligible: list[tuple[str, int, int, int]] = []
    if reproduction["passed"]:
        for candidate in NEW_MODELS[1:]:
            easy_gain = (
                metrics[candidate]["ld_easy64"]["solved"]
                - metrics[a0]["ld_easy64"]["solved"]
            )
            wb_drop = (
                metrics[a0]["wb_gate150"]["solved"]
                - metrics[candidate]["wb_gate150"]["solved"]
            )
            monitor_drop = (
                metrics[a0]["monitor64"]["solved"]
                - metrics[candidate]["monitor64"]["solved"]
            )
            checks = {
                "ld_easy_positive_effect": easy_gain >= 1,
                "wb_drop_at_most_2": wb_drop <= 2,
                "monitor_drop_at_most_1": monitor_drop <= 1,
                "repetition_not_clearly_worse": (
                    behaviors[candidate]["ld_easy64"]["repetition_ratio"]
                    <= behaviors[a0]["ld_easy64"]["repetition_ratio"] + 0.02
                ),
                "length_finish_not_clearly_worse": (
                    behaviors[candidate]["ld_easy64"]["length_finish_ratio"]
                    <= behaviors[a0]["ld_easy64"]["length_finish_ratio"] + 0.02
                ),
                "extraction_not_worse": (
                    behaviors[candidate]["ld_easy64"][
                        "proof_extraction_success_rate"
                    ]
                    >= behaviors[a0]["ld_easy64"]["proof_extraction_success_rate"]
                    - 0.02
                ),
                "format_not_worse": (
                    behaviors[candidate]["ld_easy64"]["format_validity_rate"]
                    >= behaviors[a0]["ld_easy64"]["format_validity_rate"] - 0.02
                ),
            }
            qualifies = all(checks.values())
            decisions[candidate] = {
                "ld_easy_rows": {
                    "ANCHOR-A5-EOS-FIXED": 250,
                    "ANCHOR-A10-EOS-FIXED": 500,
                    "ANCHOR-A20-EOS-FIXED": 1000,
                }[candidate],
                "actual_ld_row_share": {
                    "ANCHOR-A5-EOS-FIXED": 250 / 3000,
                    "ANCHOR-A10-EOS-FIXED": 500 / 3000,
                    "ANCHOR-A20-EOS-FIXED": 1000 / 3000,
                }[candidate],
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
    else:
        for candidate in NEW_MODELS[1:]:
            decisions[candidate] = {
                "qualifies_for_full_evaluation": False,
                "selected_for_full_evaluation": False,
                "blocked_by": "A0 reproduction gate",
            }

    full_metrics: dict[str, dict[str, Any]] = {}
    full_behaviors: dict[str, dict[str, Any]] = {}
    for name in NEW_MODELS:
        for dataset in FULL:
            path = root / "evaluation/promoted" / name / dataset
            if not (path / "benchmark_summary.json").is_file():
                continue
            row, diagnostics, vector = metric_row(path)
            full_metrics.setdefault(name, {})[dataset] = row
            full_behaviors.setdefault(name, {})[dataset] = diagnostics
            vectors[dataset][name] = vector
    for dataset in FULL:
        for candidate, baseline in COMPARISONS[:3]:
            if candidate in vectors[dataset] and baseline in vectors[dataset]:
                key = f"{dataset}:{candidate}_vs_{baseline}"
                paired[key] = {
                    **paired_bootstrap(
                        vectors[dataset][candidate],
                        vectors[dataset][baseline],
                        seed=20261320 + len(paired),
                    ),
                    **mcnemar(
                        vectors[dataset][candidate], vectors[dataset][baseline]
                    ),
                }

    write_json(root / "comparisons/core_metrics.json", metrics)
    write_json(root / "comparisons/behavior_analysis.json", behaviors)
    write_json(root / "comparisons/paired_statistics.json", paired)
    write_json(root / "comparisons/reproduction_gate.json", reproduction)
    write_json(root / "comparisons/promotion_decisions.json", decisions)
    write_json(root / "comparisons/full_metrics.json", full_metrics)
    write_json(root / "comparisons/full_behavior.json", full_behaviors)
    write_json(
        root / "comparisons/analysis_status.json",
        {
            "missing": missing,
            "new_model_core_complete": not new_missing,
            "a0_reproduction": reproduction,
            "selected_for_full_evaluation": [
                name
                for name, row in decisions.items()
                if row.get("selected_for_full_evaluation")
            ],
            "a20_token_budget_confound": {
                "label_token_budget_vs_a0": "-5.21% (pre-existing)",
                "total_token_budget_vs_a0": "-18.71% (pre-existing)",
                "interpretation": "aggressive boundary control, not clean ratio causality",
            },
        },
    )
    print(
        json.dumps(
            {
                "missing": missing,
                "reproduction": reproduction,
                "promotion_decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
