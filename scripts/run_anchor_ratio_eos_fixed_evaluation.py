#!/usr/bin/env python3
"""Run frozen canary, core, and promoted evaluations for EOS-fixed arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.analyze_wb_ld_small_sft_ablation import behavior


MODELS = {
    "ANCHOR-A0-EOS-FIXED": "ANCHOR-A0-EOS-FIXED",
    "ANCHOR-A5-EOS-FIXED": "ANCHOR-A5-EOS-FIXED",
    "ANCHOR-A10-EOS-FIXED": "ANCHOR-A10-EOS-FIXED",
    "ANCHOR-A20-EOS-FIXED": "ANCHOR-A20-EOS-FIXED",
}
CORE = {
    "wb_gate150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay",
        20261001,
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
FULL = {
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
CANARY = {
    "wb_canary30": ("wb_gate150", 30, "discovery_replay", 20261201),
    "ld_easy_canary20": ("ld_easy64", 20, "ld_holdout", 20261202),
    "monitor_canary20": ("monitor64", 20, "monitor", 20261203),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def model_paths(project: Path, root: Path, name: str) -> tuple[Path, Path]:
    model = project / "models/Qwen2.5-1.5B-Instruct"
    adapter = root / "checkpoints" / MODELS[name] / "best"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(adapter)
    return model, adapter


def run_generation(
    *,
    project: Path,
    root: Path,
    name: str,
    dataset_name: str,
    dataset: Path,
    role: str,
    seed: int,
    stage: str,
) -> None:
    output = root / "evaluation" / stage / name / dataset_name
    if (output / "benchmark_summary.json").is_file():
        return
    model, adapter = model_paths(project, root, name)
    config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )
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
        "--adapter",
        str(adapter),
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--samples-per-statement",
        "4",
    ]
    run_logged(
        command,
        cwd=project,
        log=root / "runtime/logs" / f"{stage}_{name}_{dataset_name}.log",
    )


def prepare_canary(project: Path, root: Path) -> None:
    inventory: dict[str, Any] = {}
    for output_name, (core_name, count, role, seed) in CANARY.items():
        relative = CORE[core_name][0]
        source = project / relative
        rows = read_jsonl(source)
        if len(rows) < count:
            raise ValueError(f"{source} has fewer than {count} rows")
        destination = root / "evaluation/behavior_canary/manifests" / (
            output_name + ".jsonl"
        )
        write_jsonl(destination, rows[:count])
        inventory[output_name] = {
            "source_path": str(source),
            "source_sha256": sha256(source),
            "subset_rule": f"first_{count}_in_frozen_order",
            "rows": count,
            "subset_sha256": sha256(destination),
            "role": role,
            "seed": seed,
            "samples_per_statement": 4,
        }
    write_json(
        root / "evaluation/behavior_canary/canary_manifest_identity.json",
        inventory,
    )


def run_canary(project: Path, root: Path) -> None:
    identity = root / "evaluation/behavior_canary/canary_manifest_identity.json"
    if not identity.is_file():
        prepare_canary(project, root)
    for name in MODELS:
        for dataset_name, (_core, _count, role, seed) in CANARY.items():
            run_generation(
                project=project,
                root=root,
                name=name,
                dataset_name=dataset_name,
                dataset=(
                    root
                    / "evaluation/behavior_canary/manifests"
                    / f"{dataset_name}.jsonl"
                ),
                role=role,
                seed=seed,
                stage="behavior_canary/results",
            )
    decisions: dict[str, Any] = {}
    for name in MODELS:
        generations: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        successes = 0
        for dataset_name in CANARY:
            path = (
                root
                / "evaluation/behavior_canary/results"
                / name
                / dataset_name
            )
            generations.extend(read_jsonl(path / "generations.jsonl"))
            attempts.extend(read_jsonl(path / "attempts.jsonl"))
            successes += int(read_json(path / "benchmark_summary.json")["successes"])
        metrics = behavior(generations, attempts)
        passed = (
            metrics["output_tokens"]["mean"] <= 100
            and metrics["length_finish_ratio"] <= 0.30
            and metrics["repetition_ratio"] <= 0.30
        )
        decisions[name] = {
            "status": "BEHAVIOR_GATE_PASSED" if passed else "BEHAVIOR_GATE_FAILED",
            "passed": passed,
            "statements": 70,
            "candidates": 280,
            "solved_across_subsets": successes,
            **metrics,
        }
    write_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json",
        decisions,
    )


def run_eval_loss(project: Path, root: Path, name: str) -> None:
    output = root / "evaluation/eval160" / name
    if (output / "eval_metrics.json").is_file():
        return
    model, adapter = model_paths(project, root, name)
    config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )
    run_logged(
        [
            sys.executable,
            "-m",
            "scripts.run_round1_post_eval",
            "--config",
            str(config),
            "eval-loss",
            "--model",
            str(model),
            "--adapter",
            str(adapter),
            "--train-file",
            str(
                root
                / "training"
                / MODELS[name]
                / "input/effective_train.jsonl"
            ),
            "--output",
            str(output),
        ],
        cwd=project,
        log=root / "runtime/logs" / f"core_{name}_eval160.log",
    )


def run_core(project: Path, root: Path) -> None:
    decisions = read_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json"
    )
    for name in MODELS:
        if not decisions[name]["passed"]:
            continue
        run_eval_loss(project, root, name)
        for dataset_name, (relative, role, seed) in CORE.items():
            run_generation(
                project=project,
                root=root,
                name=name,
                dataset_name=dataset_name,
                dataset=project / relative,
                role=role,
                seed=seed,
                stage="core",
            )


def run_promoted(project: Path, root: Path) -> None:
    decisions = read_json(root / "comparisons/promotion_decisions.json")
    promoted = [
        name
        for name, row in decisions.items()
        if row.get("selected_for_full_evaluation")
    ]
    if len(promoted) > 2:
        raise RuntimeError("promotion contract permits at most two models")
    for name in promoted:
        for dataset_name, (relative, role, seed) in FULL.items():
            run_generation(
                project=project,
                root=root,
                name=name,
                dataset_name=dataset_name,
                dataset=project / relative,
                role=role,
                seed=seed,
                stage="promoted",
            )


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
        default=Path("outputs/anchor_ratio_eos_fixed"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    if args.phase == "prepare-canary":
        prepare_canary(project, root)
    elif args.phase == "canary":
        run_canary(project, root)
    elif args.phase == "core":
        run_core(project, root)
    else:
        run_promoted(project, root)
    print(root / "evaluation")


if __name__ == "__main__":
    main()
