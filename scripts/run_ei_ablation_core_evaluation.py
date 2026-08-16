#!/usr/bin/env python3
"""Run matched-seed Core evaluation for H0 and the three EI ablation arms."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SEED = 20261815
MODELS = {
    "H0-No-Hard": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED",
    "EI-A_success_only": "outputs/expert_iteration/ei_ablation/EI-A_success_only/checkpoint/EI-A-MERGED",
    "EI-B_success_frontier": "outputs/expert_iteration/ei_ablation/EI-B_success_frontier/checkpoint/EI-B-MERGED",
    "EI-C_success_frontier_replay": "outputs/expert_iteration/ei_ablation/EI-C_success_frontier_replay/checkpoint/EI-C-MERGED",
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
    with path.open(encoding="utf-8-sig") as handle:
        return sum(bool(line.strip()) for line in handle)


def eval_root(root: Path, model: str) -> Path:
    return root / ("comparisons/H0_matched_seed/evaluation" if model == "H0-No-Hard" else f"{model}/evaluation")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_faithful_complete(output: Path, expected: int) -> bool:
    summary_path = output / "benchmark_summary.json"
    if not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    return (
        lines(output / "attempts.jsonl") == expected * 2
        and lines(output / "generations.jsonl") == expected * 2
        and lines(output / "verifications.jsonl") == expected * 2
        and summary.get("data_role") == "ld_holdout"
        and summary.get("execution_mode")
        == "staged_sequential_grouped_source_faithful"
        and int(summary.get("import_group_count") or 0) > 1
    )


def ordinary_complete(output: Path, expected: int) -> bool:
    return (
        lines(output / "attempts.jsonl") == expected * 2
        and lines(output / "generations.jsonl") == expected * 2
        and lines(output / "verifications.jsonl") == expected * 2
        and (output / "benchmark_summary.json").is_file()
    )


def assert_source_faithful_errors_clean(output: Path) -> None:
    forbidden = (
        "already been declared",
        "Already exists entry for",
        "not allowed to be imported by this file",
    )
    contaminated = 0
    with (output / "verifications.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            errors = "\n".join(row.get("metadata", {}).get("compile_errors", []))
            if any(marker in errors for marker in forbidden):
                contaminated += 1
    if contaminated:
        raise RuntimeError(
            f"source-faithful LD verification contaminated by preloaded declarations: "
            f"{contaminated} attempts in {output}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/ei_ablation"
    completed: list[str] = []
    for model, checkpoint_relative in MODELS.items():
        checkpoint = project / checkpoint_relative
        model_eval = eval_root(root, model)
        config = model_eval / "runtime_config.json"
        if not config.is_file() or not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"evaluation prerequisites missing for {model}")
        for slug, (role, dataset_relative, expected) in DATASETS.items():
            output = model_eval / "core" / slug
            complete = (
                source_faithful_complete(output, expected)
                if role == "ld_holdout"
                else ordinary_complete(output, expected)
            )
            if complete:
                completed.append(f"{model}:{slug}:already_complete")
                continue
            runner = (
                "scripts/run_wb_ld_ablation_evaluation.py"
                if role == "ld_holdout"
                else "scripts/run_ablation_evaluation.py"
            )
            command = [
                sys.executable, runner,
                "--config", str(config), "--role", role,
                "--model", str(checkpoint), "--dataset", str(project / dataset_relative),
                "--output", str(output), "--seed", str(SEED), "--samples-per-statement", "2",
            ]
            subprocess.run(command, cwd=project, check=True)
            complete = (
                source_faithful_complete(output, expected)
                if role == "ld_holdout"
                else ordinary_complete(output, expected)
            )
            if not complete:
                raise RuntimeError(f"evaluation incomplete after successful process: {model}:{slug}")
            if role == "ld_holdout":
                assert_source_faithful_errors_clean(output)
            completed.append(f"{model}:{slug}:completed")
    print(json.dumps({"status": "completed", "evaluation_seed": SEED, "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
