#!/usr/bin/env python3
"""Aggregate core metrics, paired tests, promotions, and replay trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_anchor_ratio_eos_fixed import metric_row
from scripts.analyze_wb_ld_small_sft_ablation import mcnemar, paired_bootstrap


MODELS = (
    "BUDGET-A-WB2000",
    "BUDGET-A-WB1500-LD500",
    "BUDGET-A-WB1000-LD1000",
    "ADDON-B-WB2000-LD1000",
    "SUPPORT-C-LD1000",
)
DATASETS = (
    "wb_train_retention150",
    "wb_unseen_holdout150",
    "ld_easy64",
    "monitor64",
    "hard_ld128",
)
COMPARISONS = {
    "fixed_budget": (
        ("BUDGET-A-WB1500-LD500", "BUDGET-A-WB2000"),
        ("BUDGET-A-WB1000-LD1000", "BUDGET-A-WB2000"),
        ("BUDGET-A-WB1000-LD1000", "BUDGET-A-WB1500-LD500"),
    ),
    "fixed_wb": (
        ("ADDON-B-WB2000-LD1000", "BUDGET-A-WB2000"),
    ),
    "fixed_ld": (
        ("SUPPORT-C-LD1000", "BUDGET-A-WB1000-LD1000"),
        ("BUDGET-A-WB1000-LD1000", "ADDON-B-WB2000-LD1000"),
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/wb_ld_budget_support_replay_ablation"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()

    metrics: dict[str, dict[str, Any]] = {}
    behaviors: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in DATASETS
    }
    training: dict[str, Any] = {}
    for name in MODELS:
        summary = root / "training" / name / "training_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(summary)
        training[name] = read_json(summary)
        loss = root / "evaluation/eval160" / name / "eval_metrics.json"
        if not loss.is_file():
            raise FileNotFoundError(loss)
        metrics[name] = {"eval160": read_json(loss)}
        behaviors[name] = {}
        for dataset in DATASETS:
            path = root / "evaluation/core" / name / dataset
            row, diagnostics, vector = metric_row(path)
            metrics[name][dataset] = row
            behaviors[name][dataset] = diagnostics
            vectors[dataset][name] = vector

    paired: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    seed = 20261400
    for family, comparisons in COMPARISONS.items():
        tables[family] = {}
        for candidate, baseline in comparisons:
            comparison = f"{candidate}_vs_{baseline}"
            tables[family][comparison] = {}
            for dataset in DATASETS:
                candidate_row = metrics[candidate][dataset]
                baseline_row = metrics[baseline][dataset]
                result = {
                    **paired_bootstrap(
                        vectors[dataset][candidate],
                        vectors[dataset][baseline],
                        seed=seed,
                    ),
                    **mcnemar(
                        vectors[dataset][candidate],
                        vectors[dataset][baseline],
                    ),
                }
                seed += 1
                result.update(
                    {
                        "candidate_solved": candidate_row["solved"],
                        "baseline_solved": baseline_row["solved"],
                        "delta_solved": (
                            candidate_row["solved"] - baseline_row["solved"]
                        ),
                        "candidate_pass_at_4": candidate_row["pass_at_4"],
                        "baseline_pass_at_4": baseline_row["pass_at_4"],
                    }
                )
                key = f"{dataset}:{comparison}"
                paired[key] = result
                tables[family][comparison][dataset] = result

    control = "BUDGET-A-WB2000"
    promotions: dict[str, Any] = {}
    ranked: list[tuple[str, int, int, int, int]] = []
    for candidate in MODELS[1:]:
        ld_gain = (
            metrics[candidate]["ld_easy64"]["solved"]
            - metrics[control]["ld_easy64"]["solved"]
        )
        retention_drop = (
            metrics[control]["wb_train_retention150"]["solved"]
            - metrics[candidate]["wb_train_retention150"]["solved"]
        )
        monitor_drop = (
            metrics[control]["monitor64"]["solved"]
            - metrics[candidate]["monitor64"]["solved"]
        )
        unseen_drop = (
            metrics[control]["wb_unseen_holdout150"]["solved"]
            - metrics[candidate]["wb_unseen_holdout150"]["solved"]
        )
        behavior_stable = all(
            behaviors[candidate][dataset]["output_tokens"]["mean"] <= 100
            and behaviors[candidate][dataset]["length_finish_ratio"] <= 0.30
            and behaviors[candidate][dataset]["repetition_ratio"] <= 0.30
            for dataset in DATASETS
        )
        qualifies = (
            ld_gain >= 1
            and retention_drop <= 2
            and monitor_drop <= 1
            and behavior_stable
        )
        promotions[candidate] = {
            "ld_easy_solved_gain_vs_WB2000": ld_gain,
            "wb_retention_solved_drop_vs_WB2000": retention_drop,
            "wb_unseen_solved_drop_vs_WB2000": unseen_drop,
            "monitor_solved_drop_vs_WB2000": monitor_drop,
            "behavior_stable": behavior_stable,
            "qualifies_for_full_evaluation": qualifies,
            "selected_for_full_evaluation": False,
        }
        if qualifies:
            ranked.append(
                (candidate, ld_gain, -retention_drop, -monitor_drop, -unseen_drop)
            )
    for candidate, *_ in sorted(
        ranked, key=lambda item: item[1:], reverse=True
    )[:2]:
        promotions[candidate]["selected_for_full_evaluation"] = True

    # The preregistered replay experiment is conditional. A >=1 LD solved
    # gain paired with a >=3/150 retention loss is treated as a meaningful
    # transfer-retention trade-off, without using the new unseen holdout to
    # select training examples.
    replay_evidence = []
    for candidate in (
        "BUDGET-A-WB1500-LD500",
        "BUDGET-A-WB1000-LD1000",
        "ADDON-B-WB2000-LD1000",
    ):
        ld_gain = (
            metrics[candidate]["ld_easy64"]["solved"]
            - metrics[control]["ld_easy64"]["solved"]
        )
        retention_drop = (
            metrics[control]["wb_train_retention150"]["solved"]
            - metrics[candidate]["wb_train_retention150"]["solved"]
        )
        replay_evidence.append(
            {
                "candidate": candidate,
                "ld_easy_solved_gain": ld_gain,
                "wb_retention_solved_drop": retention_drop,
                "triggered": ld_gain >= 1 and retention_drop >= 3,
            }
        )
    replay_triggered = any(row["triggered"] for row in replay_evidence)
    replay = {
        "status": "TRIGGERED" if replay_triggered else "NOT_TRIGGERED",
        "rule": "LD-easy gain >=1 solved and WB retention loss >=3/150",
        "evidence": replay_evidence,
    }

    write_json(root / "comparisons/core_metrics.json", metrics)
    write_json(root / "comparisons/behavior_analysis.json", behaviors)
    write_json(root / "comparisons/paired_statistics.json", paired)
    write_json(root / "comparisons/fixed_budget_statistics.json", tables["fixed_budget"])
    write_json(root / "comparisons/fixed_wb_statistics.json", tables["fixed_wb"])
    write_json(root / "comparisons/fixed_ld_statistics.json", tables["fixed_ld"])
    write_json(root / "comparisons/promotion_decisions.json", promotions)
    write_json(root / "comparisons/replay_trigger.json", replay)
    print(json.dumps(
        {
            "promotion_decisions": promotions,
            "replay_trigger": replay,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
