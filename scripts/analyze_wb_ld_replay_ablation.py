#!/usr/bin/env python3
"""Compare triggered Core/Random/Curated WB replay selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_anchor_ratio_eos_fixed import metric_row
from scripts.analyze_wb_ld_small_sft_ablation import mcnemar, paired_bootstrap


MODELS = {
    "R-Core": "ADDON-B-WB2000-LD1000",
    "R-Random": "REPLAY-R-RANDOM-WB2000-LD1000",
    "R-Curated": "REPLAY-R-CURATED-WB2000-LD1000",
}
DATASETS = (
    "wb_train_retention150",
    "wb_unseen_holdout150",
    "ld_easy64",
    "monitor64",
    "hard_ld128",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/wb_ld_budget_support_replay_ablation"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    metrics: dict[str, Any] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = {
        dataset: {} for dataset in DATASETS
    }
    for label, model in MODELS.items():
        metrics[label] = {"model": model}
        for dataset in DATASETS:
            row, behavior, vector = metric_row(
                root / "evaluation/core" / model / dataset
            )
            metrics[label][dataset] = {**row, "behavior": behavior}
            vectors[dataset][label] = vector
    paired: dict[str, Any] = {}
    seed = 20261600
    for candidate in ("R-Random", "R-Curated"):
        for baseline in ("R-Core",):
            for dataset in DATASETS:
                key = f"{dataset}:{candidate}_vs_{baseline}"
                paired[key] = {
                    **paired_bootstrap(
                        vectors[dataset][candidate],
                        vectors[dataset][baseline],
                        seed=seed,
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
                seed += 1
    for dataset in DATASETS:
        key = f"{dataset}:R-Curated_vs_R-Random"
        paired[key] = {
            **paired_bootstrap(
                vectors[dataset]["R-Curated"],
                vectors[dataset]["R-Random"],
                seed=seed,
            ),
            **mcnemar(
                vectors[dataset]["R-Curated"],
                vectors[dataset]["R-Random"],
            ),
            "candidate_solved": metrics["R-Curated"][dataset]["solved"],
            "baseline_solved": metrics["R-Random"][dataset]["solved"],
            "delta_solved": (
                metrics["R-Curated"][dataset]["solved"]
                - metrics["R-Random"][dataset]["solved"]
            ),
        }
        seed += 1
    write_json(root / "comparisons/replay_metrics.json", metrics)
    write_json(root / "comparisons/replay_paired_statistics.json", paired)
    print(json.dumps({"metrics": metrics, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
