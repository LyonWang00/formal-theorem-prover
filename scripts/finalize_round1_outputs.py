"""Assemble the requested Round-1 output view and reproducible metrics report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_view(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_finished_processes(root: Path) -> list[dict[str, Any]]:
    path = root / "runtime" / "isolated_processes.jsonl"
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if row.get("event") == "finished"]


def discovery_generation_seconds(root: Path) -> float:
    requests = root / "runtime" / "isolated_requests"
    wanted: set[str] = set()
    for path in requests.glob("generation-*.request.json"):
        payload = read_json(path).get("payload", {})
        output = str(payload.get("output_path") or "").replace("\\", "/")
        if output.endswith("iteration_000/discovery/generations.jsonl"):
            wanted.add(path.name.removesuffix(".request.json"))
    return round(
        sum(
            float(row.get("duration_seconds") or 0.0)
            for row in load_finished_processes(root)
            if row.get("run_id") in wanted and int(row.get("return_code") or 0) == 0
        ),
        4,
    )


def runtime_metrics(root: Path) -> dict[str, Any]:
    memory_path = root / "runtime" / "memory.jsonl"
    memory = read_jsonl(memory_path) if memory_path.exists() else []
    coordinator_peak = max(
        (int(row.get("process_rss_bytes") or 0) for row in memory), default=0
    )
    ram_peak = max((int(row.get("system_used_bytes") or 0) for row in memory), default=0)
    gpu_peak = max(
        (
            int(gpu.get("used_mib") or 0)
            for row in memory
            for gpu in row.get("gpu", [])
        ),
        default=0,
    )
    lifecycle_path = root / "runtime" / "lifecycle.jsonl"
    lifecycle = read_jsonl(lifecycle_path) if lifecycle_path.exists() else []
    timestamps = [float(row["timestamp"]) for row in lifecycle if row.get("timestamp")]
    actual_wall = round(max(timestamps) - min(timestamps), 4) if timestamps else None
    total_wall = round(time.time() - min(timestamps), 4) if timestamps else None
    training_rows = [
        row
        for row in load_finished_processes(root)
        if row.get("stage") == "training" and int(row.get("return_code") or 0) == 0
    ]
    return {
        "ram_peak_bytes": ram_peak,
        "ram_peak_gib": round(ram_peak / 2**30, 4),
        "coordinator_rss_peak_bytes": coordinator_peak,
        "coordinator_rss_peak_gib": round(coordinator_peak / 2**30, 4),
        "gpu_peak_memory_mib": gpu_peak,
        "discovery_generation_seconds": discovery_generation_seconds(root),
        "successful_training_seconds": (
            round(float(training_rows[-1].get("duration_seconds") or 0.0), 4)
            if training_rows
            else None
        ),
        "observed_coordinator_wall_seconds": actual_wall,
        "observed_total_experiment_wall_seconds": total_wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    iteration = root / "iteration_000"
    discovery = iteration / "discovery"

    generations = read_jsonl(discovery / "generations.jsonl")
    verifications = read_jsonl(discovery / "verifications.jsonl")
    selected = read_jsonl(discovery / "selected_pool.jsonl")
    proof_bank = read_jsonl(root / "banks" / "proof_bank.jsonl")
    failure_bank = read_jsonl(root / "banks" / "failure_bank.jsonl")
    train_rows = read_jsonl(iteration / "train" / "train_dataset.jsonl")
    verification_by_id = {row["generation_id"]: row for row in verifications}
    successful = [
        row for row in generations if verification_by_id[row["generation_id"]].get("verified")
    ]

    success_by_statement: dict[str, int] = Counter(
        row["statement_id"] for row in successful
    )
    success_distribution = Counter(
        min(4, success_by_statement.get(row["statement_id"], 0)) for row in selected
    )
    proofs = [str(row.get("normalized_proof") or row.get("extracted_proof") or "").strip() for row in successful]
    proof_frequency = Counter(proofs)
    lengths = [
        int(row.get("metadata", {}).get("completion_tokens") or len(proof.split()))
        for row, proof in zip(successful, proofs, strict=True)
    ]
    tactics = ["simp", "norm_num", "linarith", "aesop", "omega", "ring"]
    tactic_counts = {
        tactic: sum(bool(re.search(rf"\b{re.escape(tactic)}\b", proof)) for proof in proofs)
        for tactic in tactics
    }
    tactic_counts["other"] = sum(
        not any(re.search(rf"\b{re.escape(tactic)}\b", proof) for tactic in tactics)
        for proof in proofs
    )

    proof_by_statement: dict[str, int] = Counter(
        row["statement_id"] for row in proof_bank
    )
    failure_types = Counter()
    for row in failure_bank:
        if row.get("timed_out"):
            failure_types["timeout"] += 1
        else:
            failure_types[str(row.get("error_type") or row.get("status") or "other")] += 1
    for required_type in (
        "elaboration_error",
        "syntax_error",
        "tactic_error",
        "unsolved_goals",
        "timeout",
    ):
        failure_types.setdefault(required_type, 0)

    train_sources = Counter(str(row.get("expert_source") or "unknown") for row in train_rows)
    expert_rows = [
        row for row in train_rows if "expert" in str(row.get("expert_source") or "")
    ]
    train_manifest = {
        "dataset": str(iteration / "train" / "train_dataset.jsonl"),
        "sha256": sha256(iteration / "train" / "train_dataset.jsonl"),
        "total_rows": len(train_rows),
        "source_counts": dict(train_sources),
        "expert_training_proofs": len(expert_rows),
        "requested_mixture": {"original_verified_train": 0.85, "expert_proof": 0.15},
        "all_rows_pantograph_verified": all(bool(row.get("pantograph_verified")) for row in train_rows),
    }

    m0_eval = read_json(root / "evaluation" / "eval_loss" / "M0" / "eval_metrics.json")
    m1_train_eval = read_json(iteration / "eval" / "eval_metrics.json")
    m0_monitor = read_json(root / "monitor" / "M0" / "monitor_metrics.json")
    m1_monitor = read_json(iteration / "monitor" / "monitor_metrics.json")
    m1_replay = read_json(
        root
        / "evaluation"
        / "discovery_replay"
        / "M1"
        / "discovery_replay_metrics.json"
    )

    m0_monitor_p1 = float(m0_monitor["monitor_pass_at_1"])
    m0_monitor_p4 = float(m0_monitor["monitor_pass_at_4"])
    m1_monitor_p1 = float(m1_monitor["monitor_pass_at_1"])
    m1_monitor_p4 = float(m1_monitor["monitor_pass_at_4"])
    m0_discovery_p1 = sum(
        verification_by_id[row["generation_id"]].get("verified")
        for row in generations
        if int(row.get("sample_index") or 0) == 0
    ) / max(1, len(selected))
    m0_discovery_p4 = sum(value > 0 for value in success_by_statement.values()) / max(1, len(selected))
    m1_replay_p1 = float(m1_replay["discovery_replay_pass_at_1"])
    m1_replay_p4 = float(m1_replay["discovery_replay_pass_at_4"])

    discovery_metrics = {
        "total_statements": len(selected),
        "total_candidates": len(generations),
        "verified_candidates": len(successful),
        "candidate_success_rate": len(successful) / max(1, len(generations)),
        "success_at_4_distribution": {
            f"{index}/4": success_distribution.get(index, 0) for index in range(5)
        },
    }
    proof_bank_metrics = {
        "proof_bank_size": len(proof_bank),
        "unique_solved_statements": len(proof_by_statement),
        "multiple_proof_statements": sum(value > 1 for value in proof_by_statement.values()),
        "training_proof_count": len(expert_rows),
    }
    proof_quality = {
        "mean_proof_tokens": sum(lengths) / max(1, len(lengths)),
        "p50_proof_tokens": percentile(lengths, 0.50),
        "p90_proof_tokens": percentile(lengths, 0.90),
        "p95_proof_tokens": percentile(lengths, 0.95),
        "max_proof_tokens": max(lengths, default=0),
        "duplicate_proof_ratio": 1.0 - len(proof_frequency) / max(1, len(proofs)),
        "top10_proof_coverage": sum(count for _, count in proof_frequency.most_common(10)) / max(1, len(proofs)),
        "top50_proof_coverage": sum(count for _, count in proof_frequency.most_common(50)) / max(1, len(proofs)),
        "tactic_presence_counts": tactic_counts,
    }
    evaluation = {
        "eval_loss": {
            "eval_size": 160,
            "M0": m0_eval,
            "M1": {
                "eval_loss": m1_train_eval.get("final_eval_loss"),
                "eval_token_accuracy": m1_train_eval.get("final_eval_token_accuracy"),
            },
        },
        "monitor": {
            "M0": {"pass_at_1": m0_monitor_p1, "pass_at_4": m0_monitor_p4},
            "M1": {"pass_at_1": m1_monitor_p1, "pass_at_4": m1_monitor_p4},
            "delta_pass_at_1": m1_monitor_p1 - m0_monitor_p1,
            "delta_pass_at_4": m1_monitor_p4 - m0_monitor_p4,
        },
        "discovery_replay": {
            "M0": {
                "pass_at_1": m0_discovery_p1,
                "success_at_4": m0_discovery_p4,
                "solved_statements": len(success_by_statement),
                "candidate_success_rate": len(successful) / max(1, len(generations)),
            },
            "M1": {
                "pass_at_1": m1_replay_p1,
                "success_at_4": m1_replay_p4,
                "solved_statements": int(m1_replay.get("successes") or 0),
                "candidate_success_rate": float(m1_replay.get("discovery_replay_compile_success_rate") or 0),
            },
            "delta_success_at_4": m1_replay_p4 - m0_discovery_p4,
        },
    }
    runtime = runtime_metrics(root)
    runtime.update(
        {
            "discovery_compile_time_seconds": round(
                sum(float(row.get("compile_time_ms") or 0) for row in verifications) / 1000,
                4,
            ),
            "cache_hits": sum(bool(row.get("metadata", {}).get("cache_hit")) for row in verifications),
            "cache_misses": sum(not bool(row.get("metadata", {}).get("cache_hit")) for row in verifications),
            "pantograph_worker_restart_count": sum(
                max(
                    (
                        int(row.get("metadata", {}).get("worker_restart_count") or 0)
                        for row in verifications
                        if int(row.get("metadata", {}).get("worker_id") or -1) == worker
                    ),
                    default=0,
                )
                for worker in range(2)
            ),
            "m0_monitor_generation_seconds": m0_monitor.get("generation_seconds"),
            "m0_monitor_verification_seconds": m0_monitor.get("verification_seconds"),
            "m1_monitor_generation_seconds": m1_monitor.get("generation_seconds"),
            "m1_monitor_verification_seconds": m1_monitor.get("verification_seconds"),
            "m1_discovery_replay_generation_seconds": m1_replay.get("generation_seconds"),
            "m1_discovery_replay_verification_seconds": m1_replay.get("verification_seconds"),
            "all_evaluation_cache_hits": sum(
                int(row.get("cache_hits") or 0)
                for row in (m0_monitor, m1_monitor, m1_replay)
            ),
            "all_evaluation_cache_misses": sum(
                int(row.get("cache_misses") or 0)
                for row in (m0_monitor, m1_monitor, m1_replay)
            ),
            "all_evaluation_worker_restarts": sum(
                int(row.get("pantograph_worker_restart_count") or 0)
                for row in (m0_monitor, m1_monitor, m1_replay)
            ),
        }
    )
    metrics = {
        "experiment": "expert_iteration_round1",
        "completed_at_unix": time.time(),
        "discovery": discovery_metrics,
        "proof_bank": proof_bank_metrics,
        "failure_bank": {
            "failure_bank_size": len(failure_bank),
            "error_distribution": dict(failure_types),
        },
        "proof_quality": proof_quality,
        "expert_sft": {
            "started": len(expert_rows) >= 50,
            "expert_training_proofs": len(expert_rows),
            "checkpoint": str(root / "checkpoint" / "M1"),
            "training_metrics": m1_train_eval,
        },
        "evaluation": evaluation,
        "expert_gain_pass_at_4": m1_monitor_p4 - m0_monitor_p4,
        "runtime": runtime,
    }

    copy_view(discovery / "generations.jsonl", root / "discovery" / "generations.jsonl")
    copy_view(discovery / "verifications.jsonl", root / "discovery" / "verification.jsonl")
    copy_view(root / "banks" / "proof_bank.jsonl", root / "proof_bank" / "proof_bank.jsonl")
    copy_view(root / "banks" / "failure_bank.jsonl", root / "failure_bank" / "failure_bank.jsonl")
    checkpoint_view = root / "checkpoint" / "M1"
    checkpoint_view.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_view.exists():
        checkpoint_view.symlink_to(Path("../iteration_000/checkpoint"), target_is_directory=True)
    write_json(root / "train_manifest.json", train_manifest)
    write_json(root / "evaluation" / "eval_loss.json", evaluation["eval_loss"])
    write_json(root / "evaluation" / "monitor.json", evaluation["monitor"])
    write_json(root / "evaluation" / "discovery_replay.json", evaluation["discovery_replay"])
    write_json(root / "metrics.json", metrics)

    report = f"""# Expert Iteration Round-1 Report

- Discovery: {len(selected)} statements, {len(generations)} candidates, {len(successful)} verified ({len(successful) / max(1, len(generations)):.2%}).
- Proof Bank: {len(proof_bank)} records over {len(proof_by_statement)} solved statements; {len(expert_rows)} proofs selected for SFT.
- Failure Bank: {len(failure_bank)} records.
- M0 monitor: Pass@1 {m0_monitor_p1:.2%}, Pass@4 {m0_monitor_p4:.2%}.
- M1 monitor: Pass@1 {m1_monitor_p1:.2%}, Pass@4 {m1_monitor_p4:.2%}.
- Expert Gain (monitor Pass@4): {m1_monitor_p4 - m0_monitor_p4:+.2%}.
- Discovery replay success@4: M0 {m0_discovery_p4:.2%}, M1 {m1_replay_p4:.2%}, delta {m1_replay_p4 - m0_discovery_p4:+.2%}.
- Eval loss: M0 {m0_eval['eval_loss']:.6f}, M1 {float(m1_train_eval['final_eval_loss']):.6f}.

Machine-readable details are in `metrics.json`.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
