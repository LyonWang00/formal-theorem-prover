#!/usr/bin/env python3
"""Run matched-seed Core evaluation for four difficulty ablation arms."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SEED = 20261815
MODELS = {
    "A_medium_hard": "A-MERGED", "B_medium_only": "B-MERGED",
    "C_add_repair": "C-MERGED", "D_add_replay": "D-MERGED",
}
DATASETS = {
    "wb_unseen_holdout150": ("benchmark", "outputs/wb_ld_budget_support_replay_ablation/datasets/wb_unseen_holdout_150.jsonl", 150),
    "ld_easy64": ("ld_holdout", "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/ld_easy_holdout_64.jsonl", 64),
    "monitor64": ("monitor", "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl", 64),
    "wb_train_retention150": ("discovery_replay", "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl", 150),
}


def lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8-sig"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def complete(output: Path, expected: int, source_faithful: bool) -> bool:
    summary_path = output / "benchmark_summary.json"
    if not summary_path.is_file() or lines(output / "attempts.jsonl") != expected * 2 or lines(output / "generations.jsonl") != expected * 2 or lines(output / "verifications.jsonl") != expected * 2:
        return False
    if not source_faithful:
        return True
    summary = read_json(summary_path)
    return summary.get("data_role") == "ld_holdout" and summary.get("execution_mode") == "staged_sequential_grouped_source_faithful" and int(summary.get("import_group_count") or 0) > 1


def assert_ld_clean(output: Path) -> None:
    forbidden = ("already been declared", "Already exists entry for", "not allowed to be imported by this file")
    contaminated = 0
    for line in (output / "verifications.jsonl").open(encoding="utf-8-sig"):
        row = json.loads(line)
        errors = "\n".join(row.get("metadata", {}).get("compile_errors", []))
        contaminated += any(marker in errors for marker in forbidden)
    if contaminated:
        raise RuntimeError(f"LD source-faithful contamination: {contaminated} in {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/difficulty_ablation"
    completed = []
    for model, merged_name in MODELS.items():
        checkpoint = root / model / "checkpoint" / merged_name
        model_eval = root / model / "evaluation"
        config = model_eval / "runtime_config.json"
        if not config.is_file() or not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"evaluation prerequisites missing for {model}")
        for slug, (role, dataset_relative, expected) in DATASETS.items():
            output = model_eval / "core" / slug
            source_faithful = role == "ld_holdout"
            if complete(output, expected, source_faithful):
                completed.append(f"{model}:{slug}:already_complete")
                continue
            runner = "scripts/run_wb_ld_ablation_evaluation.py" if source_faithful else "scripts/run_ablation_evaluation.py"
            command = [
                sys.executable, runner, "--config", str(config), "--role", role,
                "--model", str(checkpoint), "--dataset", str(project / dataset_relative),
                "--output", str(output), "--seed", str(SEED), "--samples-per-statement", "2",
            ]
            subprocess.run(command, cwd=project, check=True)
            if not complete(output, expected, source_faithful):
                raise RuntimeError(f"evaluation incomplete: {model}:{slug}")
            if source_faithful:
                assert_ld_clean(output)
            completed.append(f"{model}:{slug}:completed")
    print(json.dumps({"status": "completed", "evaluation_seed": SEED, "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
