#!/usr/bin/env python3
"""Run canary and core evaluation for the two newly trained replay arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import scripts.run_anchor_ratio_eos_fixed_evaluation as runner
from scripts.run_wb_ld_budget_support_evaluation import CANARY, CORE


MODELS = {
    "REPLAY-R-RANDOM-WB2000-LD1000": "REPLAY-R-RANDOM-WB2000-LD1000",
    "REPLAY-R-CURATED-WB2000-LD1000": "REPLAY-R-CURATED-WB2000-LD1000",
}


def configure() -> None:
    runner.MODELS = MODELS
    runner.CORE = CORE
    runner.CANARY = CANARY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare-canary", "canary", "core"))
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
        path = root / "evaluation/behavior_canary/behavior_gate_decisions.json"
        existing = runner.read_json(path) if path.is_file() else {}
        runner.run_canary(project, root)
        replay = runner.read_json(path)
        runner.write_json(path, {**existing, **replay})
    else:
        runner.run_core(project, root)
    print(root / "evaluation")


if __name__ == "__main__":
    main()
