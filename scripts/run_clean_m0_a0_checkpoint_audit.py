#!/usr/bin/env python3
"""Merge and compare A0/R0/R1/R2 checkpoints on 24 fixed greedy prompts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lean_prover.lean_training.expert_iteration.isolated_stage import (
    expose_environment_cuda_toolkit,
)


ADAPTERS = {
    "A0-EXISTING": (
        "outputs/initial_anchor_ratio_ablation/checkpoints/A0_WB3000_LD0/best"
    ),
    "R0-HISTORICAL": (
        "outputs/clean_m0_a0_reproduction_audit/reproductions/"
        "R0_HISTORICAL/checkpoint/best"
    ),
    "R1-HIST-DATA-CURRENT-PIPELINE": (
        "outputs/clean_m0_a0_reproduction_audit/reproductions/"
        "R1_HIST_DATA_CURRENT_PIPELINE/checkpoint/best"
    ),
    "R2-CURRENT-DATA-HIST-PIPELINE": (
        "outputs/clean_m0_a0_reproduction_audit/reproductions/"
        "R2_CURRENT_DATA_HIST_PIPELINE/checkpoint/best"
    ),
}


def run(command: list[str], *, cwd: Path, log: Path) -> None:
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


def main() -> None:
    expose_environment_cuda_toolkit(os.environ)
    project = Path(".").resolve()
    root = project / "outputs/clean_m0_a0_reproduction_audit/checkpoint_audit"
    base = project / "models/Qwen2.5-1.5B-Instruct"
    dataset = project / "data/processed/lean_workbook_verified_v2/eval.jsonl"
    common = root / "equivalence/common"
    base_output = common / "base.jsonl"
    if not base_output.is_file():
        run(
            [
                sys.executable,
                "-m",
                "scripts.checkpoint_equivalence",
                "transformers",
                "--model",
                str(base),
                "--dataset",
                str(dataset),
                "--output",
                str(base_output),
                "--label",
                "base",
                "--count",
                "24",
                "--seed",
                "42",
                "--max-new-tokens",
                "64",
            ],
            cwd=project,
            log=root / "logs/base_transformers.log",
        )
    for name, relative in ADAPTERS.items():
        adapter = project / relative
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter)
        merged = root / "merged" / name
        if not (merged / "model.safetensors").is_file():
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.merge_lora_adapter",
                    "--base-model",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(merged),
                ],
                cwd=project,
                log=root / f"logs/{name}_merge.log",
            )
        output = root / "equivalence" / name
        adapter_output = output / "adapter.jsonl"
        merged_output = output / "merged.jsonl"
        vllm_output = output / "vllm.jsonl"
        for label, model, adapter_arg, target in (
            ("adapter", base, adapter, adapter_output),
            ("merged", merged, None, merged_output),
        ):
            if target.is_file():
                continue
            command = [
                sys.executable,
                "-m",
                "scripts.checkpoint_equivalence",
                "transformers",
                "--model",
                str(model),
                "--dataset",
                str(dataset),
                "--output",
                str(target),
                "--label",
                label,
                "--count",
                "24",
                "--seed",
                "42",
                "--max-new-tokens",
                "64",
            ]
            if adapter_arg:
                command.extend(["--adapter", str(adapter_arg)])
            run(
                command,
                cwd=project,
                log=root / f"logs/{name}_{label}_transformers.log",
            )
        if not vllm_output.is_file():
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.checkpoint_equivalence",
                    "vllm",
                    "--model",
                    str(merged),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(vllm_output),
                    "--label",
                    "vllm",
                    "--count",
                    "24",
                    "--seed",
                    "42",
                    "--max-new-tokens",
                    "64",
                    "--max-model-len",
                    "1152",
                    "--gpu-memory-utilization",
                    "0.65",
                ],
                cwd=project,
                log=root / f"logs/{name}_vllm.log",
            )
        report = output / "report.json"
        if not report.is_file():
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.compare_clean_m0_a0_checkpoint_equivalence",
                    "--base",
                    str(base_output),
                    "--adapter",
                    str(adapter_output),
                    "--merged",
                    str(merged_output),
                    "--vllm",
                    str(vllm_output),
                    "--output",
                    str(report),
                ],
                cwd=project,
                log=root / f"logs/{name}_compare.log",
            )
    print(root)


if __name__ == "__main__":
    main()
