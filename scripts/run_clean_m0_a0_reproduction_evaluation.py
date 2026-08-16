#!/usr/bin/env python3
"""Run frozen canary or full evaluations for the CLEAN-M0/A0 reproductions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "CLEAN-M0": (
        "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor",
        None,
    ),
    "A0-EXISTING": (
        "models/Qwen2.5-1.5B-Instruct",
        "outputs/initial_anchor_ratio_ablation/checkpoints/A0_WB3000_LD0/best",
    ),
    "R0-HISTORICAL": (
        "models/Qwen2.5-1.5B-Instruct",
        (
            "outputs/clean_m0_a0_reproduction_audit/reproductions/"
            "R0_HISTORICAL/checkpoint/best"
        ),
    ),
    "R1-HIST-DATA-CURRENT-PIPELINE": (
        "models/Qwen2.5-1.5B-Instruct",
        (
            "outputs/clean_m0_a0_reproduction_audit/reproductions/"
            "R1_HIST_DATA_CURRENT_PIPELINE/checkpoint/best"
        ),
    ),
    "R2-CURRENT-DATA-HIST-PIPELINE": (
        "models/Qwen2.5-1.5B-Instruct",
        (
            "outputs/clean_m0_a0_reproduction_audit/reproductions/"
            "R2_CURRENT_DATA_HIST_PIPELINE/checkpoint/best"
        ),
    ),
}
FULL_DATASETS = {
    "wb_gate150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay",
        20261001,
    ),
    "ld_easy_holdout64": (
        (
            "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
            "ld_easy_holdout_64.jsonl"
        ),
        "ld_holdout",
        20261002,
    ),
    "monitor64": (
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "monitor",
        20261003,
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


def run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log}")


def paths(project: Path, name: str) -> tuple[Path, Path | None]:
    model_rel, adapter_rel = MODEL_SPECS[name]
    model = project / model_rel
    adapter = project / adapter_rel if adapter_rel else None
    if not model.is_dir():
        raise FileNotFoundError(model)
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter)
    return model, adapter


def run_canary(project: Path, root: Path) -> None:
    config = root / "evaluation/canary/config_greedy.json"
    dataset = root / "evaluation/canary/wb_canary_24.jsonl"
    for name in MODEL_SPECS:
        output = root / "evaluation/canary/results" / name
        if (output / "wb/metrics.json").is_file():
            continue
        model, adapter = paths(project, name)
        command = [
            sys.executable,
            "-m",
            "scripts.run_ld_length_canary",
            "--config",
            str(config),
            "--model-name",
            name,
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--seed",
            "20261101",
            "--max-new-tokens",
            "256",
            "--samples-per-statement",
            "1",
            "--expected-wb",
            "24",
            "--expected-ld",
            "0",
        ]
        if adapter:
            command.extend(["--adapter", str(adapter)])
        run_logged(
            command,
            cwd=project,
            log=root / f"evaluation/logs/canary_{name}.log",
        )
    decisions = {}
    for name in MODEL_SPECS:
        metrics = read_json(
            root / f"evaluation/canary/results/{name}/wb/metrics.json"
        )
        length_rate = float(metrics["length_finish"]["rate"])
        repetition_rate = float(metrics["repetition_ratio"])
        decisions[name] = {
            "length_finish_rate": length_rate,
            "repetition_rate": repetition_rate,
            "requires_eos_merge_audit_before_full": (
                length_rate > 0.40 or repetition_rate > 0.40
            ),
        }
    decisions["R3-CURRENT-A0"] = {
        **decisions["A0-EXISTING"],
        "artifact": "exact reference alias of A0-EXISTING",
    }
    write_json(root / "evaluation/canary/canary_gate_decisions.json", decisions)


def run_full(project: Path, root: Path) -> None:
    decisions_path = root / "evaluation/canary/canary_gate_decisions.json"
    if not decisions_path.is_file():
        raise FileNotFoundError("run canary phase first")
    decisions = read_json(decisions_path)
    for name, decision in decisions.items():
        if not decision.get("requires_eos_merge_audit_before_full"):
            continue
        if name == "R3-CURRENT-A0":
            continue
        report = root / f"checkpoint_audit/equivalence/{name}/report.json"
        if not report.is_file():
            raise FileNotFoundError(
                f"{name} crossed the 40% drift gate; merge audit is required"
            )
    config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )
    for name in (
        "R0-HISTORICAL",
        "R1-HIST-DATA-CURRENT-PIPELINE",
        "R2-CURRENT-DATA-HIST-PIPELINE",
    ):
        model, adapter = paths(project, name)
        model_root = root / "evaluation/full" / name
        eval_output = model_root / "teacher_forcing_eval160"
        if not (eval_output / "eval_metrics.json").is_file():
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
                str(eval_output),
            ]
            if adapter:
                command.extend(["--adapter", str(adapter)])
            run_logged(
                command,
                cwd=project,
                log=root / f"evaluation/logs/full_{name}_eval160.log",
            )
        for dataset_name, (relative, role, seed) in FULL_DATASETS.items():
            output = model_root / dataset_name
            if (output / "benchmark_summary.json").is_file():
                continue
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
                str(project / relative),
                "--output",
                str(output),
                "--seed",
                str(seed),
                "--samples-per-statement",
                "4",
            ]
            if adapter:
                command.extend(["--adapter", str(adapter)])
            run_logged(
                command,
                cwd=project,
                log=root / f"evaluation/logs/full_{name}_{dataset_name}.log",
            )
    write_json(
        root / "evaluation/reference_models.json",
        {
            "CLEAN-M0": {
                "evaluation": str(
                    project
                    / "outputs/initial_anchor_ratio_ablation/evaluation/CLEAN-M0"
                ),
                "reuse_reason": "same frozen manifests and generation settings",
            },
            "A0-EXISTING": {
                "evaluation": str(
                    project
                    / "outputs/initial_anchor_ratio_ablation/evaluation/ANCHOR-A0"
                ),
                "reuse_reason": "existing complete artifact; no retraining",
            },
            "R3-CURRENT-A0": {
                "evaluation": str(
                    project
                    / "outputs/initial_anchor_ratio_ablation/evaluation/ANCHOR-A0"
                ),
                "checkpoint": str(
                    project
                    / "outputs/initial_anchor_ratio_ablation/checkpoints/"
                    "A0_WB3000_LD0/best"
                ),
                "reuse_reason": "R3 is the exact existing A0 reference",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("canary", "full"))
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/clean_m0_a0_reproduction_audit"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    if args.phase == "canary":
        run_canary(project, root)
    else:
        run_full(project, root)
    print(root / "evaluation")


if __name__ == "__main__":
    main()
