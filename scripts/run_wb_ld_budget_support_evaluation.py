#!/usr/bin/env python3
"""Run the adapted retention/unseen WB/LD budget-support evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import scripts.run_anchor_ratio_eos_fixed_evaluation as runner


MODELS = {
    "BUDGET-A-WB2000": "BUDGET-A-WB2000",
    "BUDGET-A-WB1500-LD500": "BUDGET-A-WB1500-LD500",
    "BUDGET-A-WB1000-LD1000": "BUDGET-A-WB1000-LD1000",
    "ADDON-B-WB2000-LD1000": "ADDON-B-WB2000-LD1000",
    "SUPPORT-C-LD1000": "SUPPORT-C-LD1000",
}
CORE = {
    "wb_train_retention150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay",
        20261001,
    ),
    "wb_unseen_holdout150": (
        "outputs/wb_ld_budget_support_replay_ablation/datasets/"
        "wb_unseen_holdout_150.jsonl",
        "benchmark",
        20261007,
    ),
    "ld_easy64": (
        "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
        "ld_easy_holdout_64.jsonl",
        "ld_holdout",
        20261002,
    ),
    "monitor64": (
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "monitor",
        20261003,
    ),
    "hard_ld128": (
        "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        "ld_holdout",
        20261004,
    ),
}
CANARY = {
    "wb_retention_canary30": (
        "wb_train_retention150",
        30,
        "discovery_replay",
        20261201,
    ),
    "ld_easy_canary20": ("ld_easy64", 20, "ld_holdout", 20261202),
    "monitor_canary20": ("monitor64", 20, "monitor", 20261203),
}


def configure() -> None:
    runner.MODELS = MODELS
    runner.CORE = CORE
    runner.CANARY = CANARY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare-canary", "canary", "core", "promoted"),
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/wb_ld_budget_support_replay_ablation"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    configure()
    if args.phase == "prepare-canary":
        runner.prepare_canary(project, root)
    elif args.phase == "canary":
        runner.run_canary(project, root)
    elif args.phase == "core":
        runner.run_core(project, root)
    else:
        runner.run_promoted(project, root)
    print(root / "evaluation")


if __name__ == "__main__":
    main()
