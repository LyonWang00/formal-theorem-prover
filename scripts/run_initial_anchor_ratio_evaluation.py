#!/usr/bin/env python3
"""Run the frozen initial-anchor ratio evaluation and gated full replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODELS = {
    "BASE-ZERO": ("models/Qwen2.5-1.5B-Instruct", None),
    "CLEAN-M0": (
        "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor",
        None,
    ),
    "ANCHOR-A0": (
        "models/Qwen2.5-1.5B-Instruct",
        "checkpoints/A0_WB3000_LD0/best",
    ),
    "ANCHOR-A5": (
        "models/Qwen2.5-1.5B-Instruct",
        "checkpoints/A5_WB2750_LD250/best",
    ),
    "ANCHOR-A10": (
        "models/Qwen2.5-1.5B-Instruct",
        "checkpoints/A10_WB2500_LD500/best",
    ),
    "ANCHOR-A20": (
        "models/Qwen2.5-1.5B-Instruct",
        "checkpoints/A20_WB2000_LD1000/best",
    ),
}
CORE_DATASETS = {
    "wb_gate150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay",
        20261001,
    ),
    "ld_easy_holdout64": (
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
    "ld_hard128": (
        "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        "ld_holdout",
        20261004,
    ),
}
FULL_DATASETS = {
    "full500": (
        "outputs/b2_expanded_validation/datasets/full500.jsonl",
        "discovery_replay",
        20261005,
    ),
    "strict_unseen200": (
        "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
        "discovery_replay",
        20261006,
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}; inspect {log}")


def model_paths(project: Path, root: Path, name: str) -> tuple[Path, Path | None]:
    model_rel, adapter_rel = MODELS[name]
    model = project / model_rel
    adapter = root / adapter_rel if adapter_rel else None
    if not model.exists():
        raise FileNotFoundError(f"missing frozen model: {model}")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing trained adapter: {adapter}")
    return model, adapter


def evaluate_generation(
    *,
    project: Path,
    root: Path,
    config: Path,
    model_name: str,
    dataset_name: str,
    dataset: Path,
    role: str,
    seed: int,
) -> None:
    output = root / "evaluation" / model_name / dataset_name
    if (output / "benchmark_summary.json").is_file():
        return
    model, adapter = model_paths(project, root, model_name)
    command = [
        sys.executable,
        "-m",
        "scripts.run_wb_ld_ablation_evaluation",
        "--config",
        str(config),
        "--role",
        role,
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--samples-per-statement",
        "4",
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    run_logged(
        command,
        cwd=project,
        log=root / "runtime/logs" / f"eval_{model_name}_{dataset_name}.log",
    )


def evaluate_loss(
    *, project: Path, root: Path, config: Path, model_name: str
) -> None:
    output = root / "evaluation" / model_name / "teacher_forcing_eval160"
    if (output / "eval_metrics.json").is_file():
        return
    model, adapter = model_paths(project, root, model_name)
    command = [
        sys.executable,
        "-m",
        "scripts.run_round1_post_eval",
        "--config",
        str(config),
        "eval-loss",
        "--model",
        str(model),
        "--output",
        str(output),
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    run_logged(
        command,
        cwd=project,
        log=root / "runtime/logs" / f"eval_{model_name}_eval160.log",
    )


def run_analysis(project: Path, root: Path, *, allow_incomplete: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.analyze_initial_anchor_ratio_ablation",
        "--root",
        str(root),
    ]
    if allow_incomplete:
        command.append("--allow-incomplete")
    run_logged(
        command,
        cwd=project,
        log=root / "runtime/logs/analyze.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/initial_anchor_ratio_ablation"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )
    if not config.is_file():
        raise FileNotFoundError(f"frozen generation config missing: {config}")

    for model_name in MODELS:
        evaluate_loss(project=project, root=root, config=config, model_name=model_name)
        for dataset_name, (relative, role, seed) in CORE_DATASETS.items():
            dataset = project / relative
            if not dataset.is_file():
                raise FileNotFoundError(f"frozen evaluation set missing: {dataset}")
            evaluate_generation(
                project=project,
                root=root,
                config=config,
                model_name=model_name,
                dataset_name=dataset_name,
                dataset=dataset,
                role=role,
                seed=seed,
            )

    run_analysis(project, root)
    promotions = read_json(root / "comparisons/promotion_decisions.json")
    promoted = [
        name
        for name, decision in promotions.items()
        if decision.get("selected_for_full_evaluation")
    ]
    for model_name in promoted:
        for dataset_name, (relative, role, seed) in FULL_DATASETS.items():
            evaluate_generation(
                project=project,
                root=root,
                config=config,
                model_name=model_name,
                dataset_name=dataset_name,
                dataset=project / relative,
                role=role,
                seed=seed,
            )
    run_analysis(project, root)
    run_logged(
        [
            sys.executable,
            "-m",
            "scripts.report_initial_anchor_ratio_ablation",
            "--root",
            str(root),
        ],
        cwd=project,
        log=root / "runtime/logs/report.log",
    )
    print(json.dumps({"promoted_models": promoted, "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
