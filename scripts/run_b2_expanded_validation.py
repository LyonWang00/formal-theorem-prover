"""Run the real, gated B2 expanded validation with 100-statement stage lifecycles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import psutil

from lean_prover.lean_training.evaluation.expanded_validation import (
    PRIMARY_SEED,
    REPLICATION_SEED,
    benchmark_gate,
    metric_summary,
    primary_gate,
    read_jsonl,
    sha256_file,
    statement_id,
    write_json_atomic,
    write_jsonl_atomic,
)


MODELS = {
    "M0": {
        "base": Path("outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"),
        "adapter": None,
    },
    "B2": {
        "base": Path("outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"),
        "adapter": Path(
            "outputs/expert_sft_no_replacement_ablation/checkpoints/"
            "B2_anchor_expert_1000/step_63"
        ),
    },
}


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def complete_batch(path: Path, statements: int, role: str) -> bool:
    expected = statements * 4
    return (
        count_jsonl(path / "generations.jsonl") == expected
        and count_jsonl(path / "verifications.jsonl") == expected
        and count_jsonl(path / "attempts.jsonl") == expected
        and (path / f"{role}_metrics.json").is_file()
    )


def gpu_memory_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return 0
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return max(values, default=0)


def suspicious_processes() -> list[dict[str, Any]]:
    needles = ("VLLM::EngineCore", "pantograph-repl", "sft_pipeline.trainer")
    found = []
    for process in psutil.process_iter(("pid", "cmdline", "name")):
        try:
            command = (
                str(process.info.get("name") or "")
                + " "
                + " ".join(process.info.get("cmdline") or ())
            ).strip()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(needle in command for needle in needles):
            found.append({"pid": process.pid, "command": command})
    return found


def monitored_run(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    coordinator = psutil.Process()
    peak_system_used = 0
    peak_coordinator = 0
    peak_tree = 0
    peak_gpu = 0
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        while process.poll() is None:
            peak_system_used = max(peak_system_used, psutil.virtual_memory().used)
            try:
                peak_coordinator = max(peak_coordinator, coordinator.memory_info().rss)
                children = psutil.Process(process.pid).children(recursive=True)
                tree_rss = psutil.Process(process.pid).memory_info().rss + sum(
                    child.memory_info().rss for child in children if child.is_running()
                )
                peak_tree = max(peak_tree, tree_rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            peak_gpu = max(peak_gpu, gpu_memory_used_mib())
            time.sleep(2)
        return_code = process.wait()
    return {
        "command": command,
        "exit_code": return_code,
        "wall_seconds": round(time.monotonic() - started, 4),
        "system_ram_peak_used_bytes": peak_system_used,
        "coordinator_rss_peak_bytes": peak_coordinator,
        "evaluation_process_tree_rss_peak_bytes": peak_tree,
        "gpu_peak_memory_mib": peak_gpu,
    }


def shard_dataset(dataset: Path, shard_root: Path, size: int = 100) -> list[Path]:
    rows = read_jsonl(dataset)
    shards = []
    for index, offset in enumerate(range(0, len(rows), size)):
        path = shard_root / f"batch_{index:03d}.jsonl"
        expected = rows[offset : offset + size]
        if not path.is_file() or read_jsonl(path) != expected:
            write_jsonl_atomic(path, expected)
        shards.append(path)
    return shards


def aggregate_model_output(model_root: Path, role: str, expected_statements: int) -> dict[str, Any]:
    batch_dirs = sorted((model_root / "batches").glob("batch_*"))
    generations: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for directory in batch_dirs:
        generations.extend(read_jsonl(directory / "generations.jsonl"))
        verifications.extend(read_jsonl(directory / "verifications.jsonl"))
        attempts.extend(read_jsonl(directory / "attempts.jsonl"))
        batch_metrics.append(
            json.loads((directory / f"{role}_metrics.json").read_text(encoding="utf-8"))
        )
        runtime_rows.append(json.loads((directory / "batch_runtime.json").read_text(encoding="utf-8")))
    expected_candidates = expected_statements * 4
    if not (
        len(generations) == len(verifications) == len(attempts) == expected_candidates
    ):
        raise ValueError("aggregate candidate count mismatch")
    generation_ids = [str(row["generation_id"]) for row in generations]
    verification_ids = [str(row["generation_id"]) for row in verifications]
    if len(set(generation_ids)) != expected_candidates:
        raise ValueError("generation IDs are not unique")
    if Counter(generation_ids) != Counter(verification_ids):
        raise ValueError("generation and verification IDs are not one-to-one")
    write_jsonl_atomic(model_root / "generations.jsonl", generations)
    write_jsonl_atomic(model_root / "verifications.jsonl", verifications)
    write_jsonl_atomic(model_root / "attempts.jsonl", attempts)
    metrics = metric_summary(attempts)
    metrics.update(
        {
            "generation_seconds": sum(float(row.get("generation_seconds") or 0) for row in batch_metrics),
            "verification_seconds": sum(float(row.get("verification_seconds") or 0) for row in batch_metrics),
            "wall_seconds": sum(float(row["wall_seconds"]) for row in runtime_rows),
            "system_ram_peak_used_bytes": max(row["system_ram_peak_used_bytes"] for row in runtime_rows),
            "coordinator_rss_peak_bytes": max(row["coordinator_rss_peak_bytes"] for row in runtime_rows),
            "evaluation_process_tree_rss_peak_bytes": max(row["evaluation_process_tree_rss_peak_bytes"] for row in runtime_rows),
            "gpu_peak_memory_mib": max(row["gpu_peak_memory_mib"] for row in runtime_rows),
            "pantograph_worker_startup_seconds": [
                float(metric.get("warmup_seconds") or 0) for metric in batch_metrics
            ],
            "worker_restart_count": sum(int(metric.get("pantograph_worker_restart_count") or 0) for metric in batch_metrics),
            "timeout_count": sum(int(metric.get("timeout_attempts") or 0) for metric in batch_metrics),
            "cache_hits": sum(int(metric.get("cache_hits") or 0) for metric in batch_metrics),
            "cache_misses": sum(int(metric.get("cache_misses") or 0) for metric in batch_metrics),
            "generation_process_exit_codes": [row["exit_code"] for row in runtime_rows],
            "verification_process_exit_codes": [0 for _ in runtime_rows],
            "batch_count": len(batch_dirs),
            "residual_process_count": len(suspicious_processes()),
        }
    )
    write_json_atomic(model_root / "metrics.json", metrics)
    return metrics


def run_model_dataset(
    *,
    root: Path,
    config: Path,
    dataset: Path,
    stage: str,
    model_name: str,
    role: str,
    seed: int,
    max_attempts: int,
) -> dict[str, Any]:
    rows = read_jsonl(dataset)
    shards = shard_dataset(
        dataset,
        root / "datasets/shards" / stage,
    )
    model_root = root / "evaluations" / stage / model_name
    model = MODELS[model_name]
    for index, shard in enumerate(shards, start=1):
        batch_rows = read_jsonl(shard)
        output = model_root / "batches" / f"batch_{index - 1:03d}"
        if complete_batch(output, len(batch_rows), role):
            print(f"RESUME_SKIP stage={stage} model={model_name} batch={index}/{len(shards)}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/run_ablation_evaluation.py",
            "--config", str(config),
            "--role", role,
            "--model", str(model["base"]),
            "--dataset", str(shard),
            "--output", str(output),
            "--seed", str(seed),
            "--samples-per-statement", "4",
        ]
        if model["adapter"] is not None:
            command.extend(("--adapter", str(model["adapter"])))
        for attempt in range(1, max_attempts + 1):
            print(f"RUN stage={stage} model={model_name} batch={index}/{len(shards)} attempt={attempt}", flush=True)
            runtime = monitored_run(command, output / "evaluation.log")
            write_json_atomic(output / "batch_runtime.json", runtime)
            if runtime["exit_code"] == 0 and complete_batch(output, len(batch_rows), role):
                break
            if attempt == max_attempts:
                raise RuntimeError(
                    f"evaluation failed stage={stage} model={model_name} batch={index}; "
                    f"see {output / 'evaluation.log'}"
                )
            time.sleep(5)
    return aggregate_model_output(model_root, role, len(rows))


def subset_metrics(root: Path, stage: str, model: str, dataset: Path) -> dict[str, Any]:
    allowed = {statement_id(row) for row in read_jsonl(dataset)}
    attempts = [
        row
        for row in read_jsonl(root / f"evaluations/{stage}/{model}/attempts.jsonl")
        if str(row["problem_id"]) in allowed
    ]
    return metric_summary(attempts)


def run_stage_pair(
    *, root: Path, config: Path, dataset: Path, stage: str, role: str, seed: int, max_attempts: int
) -> dict[str, Any]:
    result = {}
    for model_name in ("M0", "B2"):
        result[model_name] = run_model_dataset(
            root=root,
            config=config,
            dataset=dataset,
            stage=stage,
            model_name=model_name,
            role=role,
            seed=seed,
            max_attempts=max_attempts,
        )
    write_json_atomic(root / f"evaluations/{stage}/stage_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/b2_expanded_validation"))
    parser.add_argument("--config", type=Path, default=Path("configs/expert_iteration.round1.yaml"))
    parser.add_argument("--phase", choices=("preflight", "primary", "replication", "benchmark", "all"), default="all")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    datasets = args.root / "datasets"
    if not (args.root / "run_manifest.json").is_file():
        raise FileNotFoundError("run_manifest.json missing; build datasets first")
    if suspicious_processes():
        raise RuntimeError(f"pre-existing GPU/Pantograph/trainer processes: {suspicious_processes()}")

    if args.phase in ("preflight", "all"):
        strict_rows = read_jsonl(datasets / "strict_unseen_discovery.jsonl")[:5]
        smoke = datasets / "preflight_smoke5.jsonl"
        write_jsonl_atomic(smoke, strict_rows)
        result = run_stage_pair(
            root=args.root, config=args.config, dataset=smoke, stage="preflight_smoke5",
            role="discovery_replay", seed=PRIMARY_SEED, max_attempts=args.max_attempts,
        )
        if any(row["candidate_count"] != 20 for row in result.values()):
            raise RuntimeError("preflight did not produce 20 candidates per model")
        write_json_atomic(args.root / "preflight.json", {"passed": True, "metrics": result})
        if args.phase == "preflight":
            return

    if args.phase in ("primary", "all"):
        full = run_stage_pair(
            root=args.root, config=args.config, dataset=datasets / "full500.jsonl",
            stage="full500", role="discovery_replay", seed=PRIMARY_SEED,
            max_attempts=args.max_attempts,
        )
        strict = run_stage_pair(
            root=args.root, config=args.config,
            dataset=datasets / "strict_unseen_discovery.jsonl",
            stage="strict_unseen_primary", role="discovery_replay", seed=PRIMARY_SEED,
            max_attempts=args.max_attempts,
        )
        monitor = run_stage_pair(
            root=args.root, config=args.config,
            dataset=datasets / "monitor_minif2f_valid_64.jsonl",
            stage="monitor_primary", role="monitor", seed=PRIMARY_SEED,
            max_attempts=args.max_attempts,
        )
        nonexpert = {
            model: subset_metrics(
                args.root, "full500", model, datasets / "full500_non_expert.jsonl"
            )
            for model in ("M0", "B2")
        }
        gate = primary_gate(
            strict_m0_solved=strict["M0"]["solved_statement_count"],
            strict_b2_solved=strict["B2"]["solved_statement_count"],
            nonexpert_m0_solved=nonexpert["M0"]["solved_statement_count"],
            nonexpert_b2_solved=nonexpert["B2"]["solved_statement_count"],
            monitor_m0_solved=monitor["M0"]["solved_statement_count"],
            monitor_b2_solved=monitor["B2"]["solved_statement_count"],
        )
        gate["nonexpert"] = nonexpert
        write_json_atomic(args.root / "primary_decision.json", gate)
        if args.phase == "primary":
            return
        if not gate["run_replication"]:
            write_json_atomic(
                args.root / "replication_decision.json",
                {"status": "not_run", "reason": "Primary gate failed", "primary_gate": gate},
            )
            write_json_atomic(
                args.root / "benchmark_decision.json",
                {"status": "not_run", "reason": "Replication was not permitted"},
            )
            return

    if args.phase in ("replication", "all"):
        strict_rep = run_stage_pair(
            root=args.root, config=args.config,
            dataset=datasets / "strict_unseen_discovery.jsonl",
            stage="strict_unseen_replication", role="discovery_replay",
            seed=REPLICATION_SEED, max_attempts=args.max_attempts,
        )
        monitor_rep = run_stage_pair(
            root=args.root, config=args.config,
            dataset=datasets / "monitor_minif2f_valid_64.jsonl",
            stage="monitor_replication", role="monitor", seed=REPLICATION_SEED,
            max_attempts=args.max_attempts,
        )
        primary_strict = {
            model: json.loads((args.root / f"evaluations/strict_unseen_primary/{model}/metrics.json").read_text(encoding="utf-8"))
            for model in ("M0", "B2")
        }
        primary_monitor = {
            model: json.loads((args.root / f"evaluations/monitor_primary/{model}/metrics.json").read_text(encoding="utf-8"))
            for model in ("M0", "B2")
        }
        gate = benchmark_gate(
            strict_primary_m0=primary_strict["M0"]["solved_statement_count"],
            strict_primary_b2=primary_strict["B2"]["solved_statement_count"],
            strict_replication_m0=strict_rep["M0"]["solved_statement_count"],
            strict_replication_b2=strict_rep["B2"]["solved_statement_count"],
            monitor_primary_m0=primary_monitor["M0"]["solved_statement_count"],
            monitor_primary_b2=primary_monitor["B2"]["solved_statement_count"],
            monitor_replication_m0=monitor_rep["M0"]["solved_statement_count"],
            monitor_replication_b2=monitor_rep["B2"]["solved_statement_count"],
        )
        gate["status"] = "completed"
        write_json_atomic(args.root / "replication_decision.json", gate)
        if args.phase == "replication":
            return
        if not gate["run_benchmark"]:
            write_json_atomic(
                args.root / "benchmark_decision.json",
                {"status": "not_run", "reason": "Replication/monitor benchmark gate failed", "gate": gate},
            )
            return

    if args.phase in ("benchmark", "all"):
        result = run_stage_pair(
            root=args.root, config=args.config,
            dataset=datasets / "benchmark_minif2f_test_96.jsonl",
            stage="benchmark", role="benchmark", seed=PRIMARY_SEED,
            max_attempts=args.max_attempts,
        )
        write_json_atomic(
            args.root / "benchmark_decision.json",
            {"status": "completed", "reason": "All prerequisite gates passed", "metrics": result},
        )


if __name__ == "__main__":
    main()
