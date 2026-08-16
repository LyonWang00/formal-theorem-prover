#!/usr/bin/env python3
"""Run frozen canary, core, and strict R-Random evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.analyze_wb_ld_small_sft_ablation import behavior
from scripts.prepare_r_random_replication import (
    ADDON_NAME,
    BASE,
    NEW_MODELS,
    OUTPUT,
    PRIOR_ROOT,
    S1_NAME,
)

ADDON_LABEL = "ADDON-B-WB2000-LD1000"
S1_LABEL = "R-RANDOM-S1-SEED-20261501"
MODELS = (
    ADDON_LABEL,
    S1_LABEL,
    "R-RANDOM-S2-SEED-20261502",
    "R-RANDOM-S3-SEED-20261503",
)
CORE = {
    "wb_retention150": (
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        "discovery_replay", 20261001, "wb_train_retention150",
    ),
    "wb_unseen150": (
        "outputs/wb_ld_budget_support_replay_ablation/datasets/"
        "wb_unseen_holdout_150.jsonl",
        "benchmark", 20261007, "wb_unseen_holdout150",
    ),
    "ld_easy64": (
        "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
        "ld_easy_holdout_64.jsonl",
        "ld_holdout", 20261002, "ld_easy64",
    ),
    "monitor64": (
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "monitor", 20261003, "monitor64",
    ),
    "hard_ld128": (
        "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        "ld_holdout", 20261004, "hard_ld128",
    ),
}
STRICT = {
    "full500": (
        "outputs/b2_expanded_validation/datasets/full500.jsonl",
        "discovery_replay", 20261005,
    ),
    "strict_unseen200": (
        "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
        "discovery_replay", 20261006,
    ),
}
CANARY = {
    "wb_retention_canary30": ("wb_retention150", 30, "discovery_replay", 20261201),
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
            command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log}")


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def materialize_reused_tree(
    source: Path, destination: Path, *, expected_statements: int | None = None
) -> None:
    if destination.exists():
        return
    if expected_statements is not None:
        summary = read_json(source / "benchmark_summary.json")
        generations = read_jsonl(source / "generations.jsonl")
        if (
            int(summary["num_benchmark_samples"]) != expected_statements
            or len(generations) != expected_statements * 4
            or not all(
                float(row["temperature"]) == 0.8
                and float(row["top_p"]) == 0.95
                and int(row["max_new_tokens"]) == 256
                for row in generations
            )
        ):
            raise RuntimeError(f"reused generation contract changed at {source}")
    shutil.copytree(source, destination, copy_function=hardlink_or_copy)


def checkpoint_paths(project: Path, root: Path, label: str) -> tuple[Path, Path]:
    base = project / BASE
    if label == ADDON_LABEL:
        return base, project / PRIOR_ROOT / "checkpoints" / ADDON_NAME / "best"
    if label == S1_LABEL:
        return base, project / PRIOR_ROOT / "checkpoints" / S1_NAME / "best"
    return base, root / "checkpoints" / label / "best"


def effective_train_path(project: Path, root: Path, label: str) -> Path:
    if label in {ADDON_LABEL, S1_LABEL}:
        old = ADDON_NAME if label == ADDON_LABEL else S1_NAME
        return project / PRIOR_ROOT / "training" / old / "input/effective_train.jsonl"
    return root / "training" / label / "input/effective_train.jsonl"


def evaluation_config(project: Path) -> Path:
    return (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )


def run_generation(
    *, project: Path, root: Path, label: str, dataset_name: str,
    dataset: Path, role: str, seed: int, output: Path,
) -> None:
    if (output / "benchmark_summary.json").is_file():
        return
    model, adapter = checkpoint_paths(project, root, label)
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter)
    run_logged(
        [
            sys.executable, "-m", "scripts.run_wb_ld_ablation_evaluation",
            "--config", str(evaluation_config(project)),
            "--role", role, "--model", str(model), "--adapter", str(adapter),
            "--dataset", str(dataset), "--output", str(output),
            "--seed", str(seed), "--samples-per-statement", "4",
        ],
        cwd=project,
        log=root / "runtime/logs" / f"{dataset_name}_{label}.log",
    )


def prepare_canary(project: Path, root: Path) -> None:
    inventory: dict[str, Any] = {}
    for output_name, (core_name, count, role, seed) in CANARY.items():
        source = project / CORE[core_name][0]
        rows = read_jsonl(source)
        destination = (
            root / "evaluation/behavior_canary/manifests" / f"{output_name}.jsonl"
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


def reuse_prior(project: Path, root: Path) -> None:
    prior = project / PRIOR_ROOT
    reuse: dict[str, Any] = {}
    for label, old_model in ((ADDON_LABEL, ADDON_NAME), (S1_LABEL, S1_NAME)):
        source = prior / "evaluation/eval160" / old_model
        materialize_reused_tree(source, root / "evaluation/eval160" / label)
        reuse[f"{label}:eval160"] = {
            "source": str(source),
            "eval_metrics_sha256": sha256(source / "eval_metrics.json"),
        }
        for name, (relative, _role, _seed, old_name) in CORE.items():
            source = prior / "evaluation/core" / old_model / old_name
            destination = root / "evaluation" / name / label
            count = len(read_jsonl(project / relative))
            materialize_reused_tree(
                source, destination, expected_statements=count
            )
            reuse[f"{label}:{name}"] = {
                "source": str(source),
                "summary_sha256": sha256(source / "benchmark_summary.json"),
                "generations_sha256": sha256(source / "generations.jsonl"),
                "verifications_sha256": sha256(source / "verifications.jsonl"),
            }
        for name, (_core, count, _role, _seed) in CANARY.items():
            source = (
                prior / "evaluation/behavior_canary/results" / old_model / name
            )
            destination = (
                root / "evaluation/behavior_canary/results" / label / name
            )
            materialize_reused_tree(
                source, destination, expected_statements=count
            )
            reuse[f"{label}:{name}"] = {
                "source": str(source),
                "summary_sha256": sha256(source / "benchmark_summary.json"),
            }
    for name in STRICT:
        source = prior / "evaluation/promoted" / ADDON_NAME / name
        destination = root / "evaluation" / name / ADDON_LABEL
        materialize_reused_tree(
            source,
            destination,
            expected_statements=500 if name == "full500" else 200,
        )
        reuse[f"{ADDON_LABEL}:{name}"] = {
            "source": str(source),
            "summary_sha256": sha256(source / "benchmark_summary.json"),
            "generations_sha256": sha256(source / "generations.jsonl"),
            "verifications_sha256": sha256(source / "verifications.jsonl"),
        }
    write_json(root / "audit/reused_evaluation_identity.json", reuse)


def run_canary(project: Path, root: Path) -> None:
    prepare_canary(project, root)
    reuse_prior(project, root)
    for label in NEW_MODELS:
        for name, (_core, _count, role, seed) in CANARY.items():
            run_generation(
                project=project,
                root=root,
                label=label,
                dataset_name=f"canary_{name}",
                dataset=(
                    root / "evaluation/behavior_canary/manifests" / f"{name}.jsonl"
                ),
                role=role,
                seed=seed,
                output=(
                    root / "evaluation/behavior_canary/results" / label / name
                ),
            )
    decisions: dict[str, Any] = {}
    for label in MODELS:
        generations: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        solved = 0
        for name in CANARY:
            path = root / "evaluation/behavior_canary/results" / label / name
            generations.extend(read_jsonl(path / "generations.jsonl"))
            attempts.extend(read_jsonl(path / "attempts.jsonl"))
            solved += int(read_json(path / "benchmark_summary.json")["successes"])
        metrics = behavior(generations, attempts)
        passed = (
            metrics["output_tokens"]["mean"] <= 100
            and metrics["length_finish_ratio"] <= 0.30
            and metrics["repetition_ratio"] <= 0.30
            and metrics["proof_extraction_success_rate"] >= 0.70
            and metrics["format_validity_rate"] >= 0.70
        )
        decisions[label] = {
            "status": "BEHAVIOR_GATE_PASSED" if passed else "BEHAVIOR_GATE_FAILED",
            "passed": passed,
            "statements": 70,
            "candidates": 280,
            "solved": solved,
            "pantograph_candidate_success_rate": sum(
                bool(row.get("success")) for row in attempts
            )
            / max(1, len(attempts)),
            "eos_or_stop_finish_rate": 1.0 - metrics["length_finish_ratio"],
            **metrics,
        }
    write_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json",
        decisions,
    )


def run_eval_loss(project: Path, root: Path, label: str) -> None:
    output = root / "evaluation/eval160" / label
    if (output / "eval_metrics.json").is_file():
        return
    model, adapter = checkpoint_paths(project, root, label)
    run_logged(
        [
            sys.executable,
            "-m",
            "scripts.run_round1_post_eval",
            "--config",
            str(evaluation_config(project)),
            "eval-loss",
            "--model",
            str(model),
            "--adapter",
            str(adapter),
            "--train-file",
            str(effective_train_path(project, root, label)),
            "--output",
            str(output),
        ],
        cwd=project,
        log=root / "runtime/logs" / f"eval160_{label}.log",
    )


def run_core(project: Path, root: Path) -> None:
    reuse_prior(project, root)
    decisions = read_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json"
    )
    for label in MODELS:
        if not decisions[label]["passed"]:
            continue
        run_eval_loss(project, root, label)
        if label in {ADDON_LABEL, S1_LABEL}:
            continue
        for name, (relative, role, seed, _old_name) in CORE.items():
            run_generation(
                project=project,
                root=root,
                label=label,
                dataset_name=name,
                dataset=project / relative,
                role=role,
                seed=seed,
                output=root / "evaluation" / name / label,
            )


def run_strict(project: Path, root: Path) -> None:
    reuse_prior(project, root)
    decisions = read_json(
        root / "evaluation/behavior_canary/behavior_gate_decisions.json"
    )
    equivalence = read_json(
        root / "evaluation/checkpoint_equivalence/equivalence_summary.json"
    )
    for label in MODELS:
        if not decisions[label]["passed"] or label == ADDON_LABEL:
            continue
        if label in NEW_MODELS and not equivalence[label]["passed"]:
            continue
        for name, (relative, role, seed) in STRICT.items():
            run_generation(
                project=project,
                root=root,
                label=label,
                dataset_name=name,
                dataset=project / relative,
                role=role,
                seed=seed,
                output=root / "evaluation" / name / label,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "canary", "core", "strict"))
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    if args.phase == "prepare":
        prepare_canary(project, root)
        reuse_prior(project, root)
    elif args.phase == "canary":
        run_canary(project, root)
    elif args.phase == "core":
        run_core(project, root)
    else:
        run_strict(project, root)
    print(root / "evaluation")


if __name__ == "__main__":
    main()
