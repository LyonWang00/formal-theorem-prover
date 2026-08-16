#!/usr/bin/env python3
"""Run gated LD-easy training and sequential paired evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TRAIN_ARM_MANIFESTS = {
    "LDE-A2-WB100": "A2_WB1000",
    "LDE-B2-LD10": "B2_WB900_LDE100",
    "LDE-C2-LD20": "C2_WB800_LDE200",
}
CORE_DATASETS = {
    "wb_gate150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay",
        20260831,
    ),
    "ld_easy_holdout64": (
        "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
        "ld_easy_holdout_64.jsonl",
        "ld_holdout",
        20260832,
    ),
    "monitor64": (
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "monitor",
        20260833,
    ),
    "ld_hard128": (
        "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        "ld_holdout",
        20260834,
    ),
}
FULL_DATASETS = {
    "full500": (
        "outputs/b2_expanded_validation/datasets/full500.jsonl",
        "discovery_replay",
        20260835,
    ),
    "strict_unseen200": (
        "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
        "discovery_replay",
        20260836,
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def active_train_arms(output: Path) -> tuple[str, ...]:
    inventory = read_json(output / "audit/input_inventory.json")
    return tuple(
        arm
        for arm, manifest in TRAIN_ARM_MANIFESTS.items()
        if manifest in inventory
    )


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
        raise RuntimeError(
            f"command failed with {completed.returncode}; inspect {log}"
        )


def evaluate(
    *,
    project: Path,
    output_root: Path,
    config: Path,
    model_name: str,
    dataset_name: str,
    dataset: Path,
    role: str,
    seed: int,
) -> None:
    output = output_root / "evaluation" / model_name / dataset_name
    if (output / "benchmark_summary.json").is_file():
        return
    base = (
        project
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
    )
    adapter = None
    if model_name != "M0-ZERO":
        adapter = output_root / "checkpoints" / model_name / "best"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing frozen adapter: {adapter}")
    command = [
        sys.executable,
        "-m",
        "scripts.run_wb_ld_ablation_evaluation",
        "--config",
        str(config),
        "--role",
        role,
        "--model",
        str(base),
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
        log=output_root / "runtime/logs" / f"eval_{model_name}_{dataset_name}.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/pilot_sft"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (project / args.root).resolve()
    gate = read_json(
        project
        / "outputs/ld_length_difficulty_pipeline/difficulty/pilot_gate.json"
    )
    if not gate.get("pilot_gate_passed"):
        raise RuntimeError("manual classification gate did not pass")
    policy = read_json(
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/"
        "recommended_generation_policy.json"
    )
    length = int(policy["default_max_new_tokens"])
    config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests"
        / f"config_L{length}.json"
    )
    if not config.is_file():
        raise FileNotFoundError(f"recommended frozen generation config missing: {config}")

    train_arms = active_train_arms(output)
    if "LDE-A2-WB100" not in train_arms or "LDE-B2-LD10" not in train_arms:
        raise RuntimeError("frozen pilot inventory does not support required A2/B2 arms")
    models = ("M0-ZERO", *train_arms)
    for arm in train_arms:
        summary = output / "training" / arm / "training_summary.json"
        if summary.is_file():
            continue
        run_logged(
            [
                sys.executable,
                "-m",
                "scripts.train_ld_easy_pilot_arm",
                "--arm",
                arm,
                "--root",
                str(output),
            ],
            cwd=project,
            log=output / "runtime/logs" / f"train_{arm}.log",
        )

    for model in models:
        for dataset_name, (relative, role, seed) in CORE_DATASETS.items():
            dataset = project / relative
            if not dataset.is_file():
                raise FileNotFoundError(f"frozen evaluation dataset missing: {dataset}")
            evaluate(
                project=project,
                output_root=output,
                config=config,
                model_name=model,
                dataset_name=dataset_name,
                dataset=dataset,
                role=role,
                seed=seed,
            )

    run_logged(
        [
            sys.executable,
            "-m",
            "scripts.analyze_ld_easy_pilot",
            "--root",
            str(output),
        ],
        cwd=project,
        log=output / "runtime/logs/analyze_core.log",
    )
    promotions = read_json(output / "comparisons/promotion_decisions.json")
    promoted = [
        model for model, decision in promotions.items() if decision.get("promoted")
    ]
    if promoted:
        for model in ("LDE-A2-WB100", *promoted):
            for dataset_name, (relative, role, seed) in FULL_DATASETS.items():
                evaluate(
                    project=project,
                    output_root=output,
                    config=config,
                    model_name=model,
                    dataset_name=dataset_name,
                    dataset=project / relative,
                    role=role,
                    seed=seed,
                )
        run_logged(
            [
                sys.executable,
                "-m",
                "scripts.analyze_ld_easy_pilot",
                "--root",
                str(output),
            ],
            cwd=project,
            log=output / "runtime/logs/analyze_full.log",
        )
    run_logged(
        [
            sys.executable,
            "-m",
            "scripts.report_ld_easy_pilot",
            "--root",
            str(output),
        ],
        cwd=project,
        log=output / "runtime/logs/report.log",
    )
    print(
        json.dumps(
            {
                "generation_policy": policy,
                "promoted_models": promoted,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
