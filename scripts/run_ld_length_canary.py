#!/usr/bin/env python3
"""Run one model/length pair over the frozen WB and LD canaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from lean_prover.lean_training.expert_iteration.config import (
    load_expert_iteration_config,
)
from lean_prover.lean_training.expert_iteration.datasets import (
    index_statements,
    load_statements,
)
from lean_prover.lean_training.expert_iteration.discovery_verifier import (
    verify_candidates,
)
from lean_prover.lean_training.expert_iteration.isolated_stage import (
    IsolatedStageRunner,
    expose_environment_cuda_toolkit,
)
from lean_prover.lean_training.expert_iteration.schemas import (
    DataRole,
    DiscoveryStatementState,
    GenerationRecord,
    VerificationRecord,
)
from lean_prover.lean_training.expert_iteration.utils import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def repetitive(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 12:
        return False
    for width in (2, 3, 4):
        grams = Counter(
            tuple(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        )
        if max(grams.values(), default=0) >= 4:
            return True
    return False


def _ram_used_bytes() -> int:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
    return values.get("MemTotal", 0) - values.get("MemAvailable", 0)


def _gpu_used_bytes() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    used_mib = 0
    for line in result.stdout.splitlines():
        try:
            used_mib += int(line.strip())
        except ValueError:
            continue
    return used_mib * 1024 * 1024


@contextmanager
def resource_monitor(interval_seconds: float = 0.5) -> Iterator[dict[str, Any]]:
    stop = threading.Event()
    metrics: dict[str, Any] = {
        "ram_peak_bytes": 0,
        "gpu_peak_bytes": 0,
        "samples": 0,
    }

    def poll() -> None:
        while not stop.is_set():
            metrics["ram_peak_bytes"] = max(
                int(metrics["ram_peak_bytes"]), _ram_used_bytes()
            )
            metrics["gpu_peak_bytes"] = max(
                int(metrics["gpu_peak_bytes"]), _gpu_used_bytes()
            )
            metrics["samples"] = int(metrics["samples"]) + 1
            stop.wait(interval_seconds)

    thread = threading.Thread(target=poll, name="length-resource-monitor", daemon=True)
    thread.start()
    try:
        yield metrics
    finally:
        stop.set()
        thread.join(timeout=10)


def dataset_name(statement) -> str:
    nested_metadata = statement.metadata.get("metadata") or {}
    value = str(
        statement.metadata.get("length_ablation_dataset")
        or nested_metadata.get("length_ablation_dataset")
        or ""
    )
    if value not in {"wb", "ld"}:
        raise ValueError(
            f"statement {statement.statement_id} lacks length_ablation_dataset"
        )
    return value


def metrics_for(
    *,
    dataset: str,
    statements: list,
    generations: list[GenerationRecord],
    verifications: list[VerificationRecord],
    generation_seconds: float,
    verification_seconds: float,
    resource: dict[str, Any],
    group_runtime: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    statement_ids = {
        statement.statement_id
        for statement in statements
        if dataset_name(statement) == dataset
    }
    selected_generations = [
        row for row in generations if row.statement_id in statement_ids
    ]
    selected_verifications = [
        row for row in verifications if row.statement_id in statement_ids
    ]
    verification_by_id = {
        row.generation_id: row for row in selected_verifications
    }
    successes: dict[str, set[int]] = {
        statement_id: set() for statement_id in statement_ids
    }
    attempts: list[dict[str, Any]] = []
    for generation in selected_generations:
        result = verification_by_id.get(generation.generation_id)
        success = bool(result and result.verified)
        if success:
            successes[generation.statement_id].add(generation.sample_index)
        attempts.append(
            {
                "problem_id": generation.statement_id,
                "attempt_index": generation.sample_index,
                "generation_id": generation.generation_id,
                "raw_output": generation.raw_output,
                "generated_proof": generation.extracted_proof,
                "finish_reason": generation.finish_reason,
                "stop_reason": generation.metadata.get("stop_reason"),
                "completion_tokens": generation.metadata.get("completion_tokens"),
                "success": success,
                "status": result.status.value if result else "missing",
                "timed_out": bool(result and result.timed_out),
                "cache_hit": bool(
                    result and result.metadata.get("cache_hit")
                ),
                "compile_time_ms": result.compile_time_ms if result else None,
            }
        )
    lengths = sorted(
        int(row.metadata.get("completion_tokens") or 0)
        for row in selected_generations
    )
    extracted = [
        str(row.extracted_proof or "").strip() for row in selected_generations
    ]
    by_statement: dict[str, list[str]] = defaultdict(list)
    for row, proof in zip(selected_generations, extracted, strict=True):
        by_statement[row.statement_id].append(proof)
    duplicate_count = sum(
        len(values) - len(set(values)) for values in by_statement.values()
    )
    finish = Counter(
        str(row.finish_reason or "unknown") for row in selected_generations
    )
    stop_reasons = Counter(
        str(row.metadata.get("stop_reason") or "none")
        for row in selected_generations
    )
    statuses = Counter(row["status"] for row in attempts)
    candidate_count = max(1, len(selected_generations))
    solved_count = sum(bool(value) for value in successes.values())
    pass_at = {
        f"pass@{k}": sum(bool(value & set(range(k))) for value in successes.values())
        / max(1, len(successes))
        for k in (1, 2, 4)
    }
    worker_restarts = sum(
        int(worker.get("restart_count") or 0)
        for group in group_runtime
        for worker in group["runtime"].get("workers", [])
    )
    eos_count = sum(
        row.finish_reason == "eos"
        or (
            row.finish_reason == "stop"
            and str(row.metadata.get("stop_reason") or "").isdigit()
        )
        for row in selected_generations
    )
    stop_string_count = sum(
        row.finish_reason == "stop"
        and bool(row.metadata.get("stop_reason"))
        and not str(row.metadata.get("stop_reason") or "").isdigit()
        for row in selected_generations
    )
    summary = {
        "dataset": dataset,
        "statements": len(statement_ids),
        "candidates": len(selected_generations),
        "solved_count": solved_count,
        "pass_at": pass_at,
        "candidate_success_rate": sum(row["success"] for row in attempts)
        / candidate_count,
        "output_tokens": {
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "total": sum(lengths),
        },
        "length_finish": {
            "count": finish["length"],
            "rate": finish["length"] / candidate_count,
        },
        "eos_finish": {"count": eos_count, "rate": eos_count / candidate_count},
        "stop_string_finish": {
            "count": stop_string_count,
            "rate": stop_string_count / candidate_count,
        },
        "finish_reason_distribution": dict(finish),
        "stop_reason_distribution": dict(stop_reasons),
        "proof_extraction_success": {
            "count": sum(bool(value) for value in extracted),
            "rate": sum(bool(value) for value in extracted) / candidate_count,
        },
        "format_validity": {
            "count": sum(
                bool(value) and ":=" not in value.splitlines()[0]
                for value in extracted
            ),
            "rate": sum(
                bool(value) and ":=" not in value.splitlines()[0]
                for value in extracted
            )
            / candidate_count,
        },
        "duplicate_candidate_ratio": duplicate_count / candidate_count,
        "repetition_ratio": sum(repetitive(value) for value in extracted)
        / candidate_count,
        "unique_proof_ratio": sum(
            len(set(values)) for values in by_statement.values()
        )
        / candidate_count,
        "pantograph_compile_success": {
            "count": sum(row["success"] for row in attempts),
            "rate": sum(row["success"] for row in attempts) / candidate_count,
        },
        "pantograph_failure_taxonomy": dict(statuses),
        "generation_wall_seconds": generation_seconds,
        "verification_wall_seconds": verification_seconds,
        "tokens_per_generation_second": sum(lengths)
        / max(generation_seconds, 1e-9),
        "gpu_peak_bytes": resource["generation"]["gpu_peak_bytes"],
        "ram_peak_bytes": max(
            resource["generation"]["ram_peak_bytes"],
            resource["verification"]["ram_peak_bytes"],
        ),
        "pantograph_worker_restart_count": worker_restarts,
        "verification_cache_hits": sum(row["cache_hit"] for row in attempts),
        "verification_cache_misses": sum(
            not row["cache_hit"] for row in attempts
        ),
    }
    return summary, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--samples-per-statement", type=int, default=4)
    parser.add_argument("--expected-wb", type=int, default=50)
    parser.add_argument("--expected-ld", type=int, default=64)
    args = parser.parse_args()
    config = load_expert_iteration_config(args.config)
    if config.discovery.generation.max_new_tokens != args.max_new_tokens:
        raise ValueError("config max_new_tokens does not match CLI")
    statements = load_statements(args.dataset, DataRole.BENCHMARK)
    counts = Counter(dataset_name(statement) for statement in statements)
    expected_counts = Counter(
        {
            dataset: count
            for dataset, count in (
                ("wb", args.expected_wb),
                ("ld", args.expected_ld),
            )
            if count
        }
    )
    if counts != expected_counts:
        raise ValueError(f"unexpected frozen canary composition: {counts}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generation_path = output / "generations.jsonl"
    verification_path = output / "verifications.jsonl"
    states = {
        statement.statement_id: DiscoveryStatementState(
            statement_id=statement.statement_id
        )
        for statement in statements
    }
    expose_environment_cuda_toolkit(os.environ)
    runner = IsolatedStageRunner(
        output / "runtime",
        flashinfer_sampler=config.execution.flashinfer_sampler,
    )
    generation_start = time.monotonic()
    try:
        with resource_monitor() as generation_resource:
            generation_result = runner.run(
                "generation",
                {
                    "config": config.model_dump(mode="json"),
                    "base_model": args.model,
                    "adapter_path": args.adapter,
                    "selected": [
                        statement.model_dump(
                            mode="json", exclude={"reference_proof"}
                        )
                        for statement in statements
                    ],
                    "statement_states": {
                        key: value.model_dump(mode="json")
                        for key, value in states.items()
                    },
                    "iteration": -30,
                    "checkpoint": str(args.adapter or args.model),
                    "output_path": str(generation_path),
                    "force": False,
                    "sampling_budget": {
                        "new": args.samples_per_statement,
                        "frontier": args.samples_per_statement,
                        "unsolved": args.samples_per_statement,
                        "audit": args.samples_per_statement,
                    },
                    "generation_seed": args.seed,
                },
                timeout=config.execution.generation_timeout_seconds,
            )
    finally:
        runner.close()
    generation_seconds = round(time.monotonic() - generation_start, 4)
    generations = [
        GenerationRecord.model_validate(row) for row in iter_jsonl(generation_path)
    ]
    expected_generations = len(statements) * args.samples_per_statement
    if len(generations) != expected_generations:
        raise ValueError(
            f"expected {expected_generations} generations, found {len(generations)}"
        )

    by_statement = index_statements(statements)
    groups: dict[tuple[str, ...], list[GenerationRecord]] = defaultdict(list)
    for generation in generations:
        groups[tuple(by_statement[generation.statement_id].imports)].append(
            generation
        )
    group_runtime: list[dict[str, Any]] = []
    verification_start = time.monotonic()
    with resource_monitor() as verification_resource:
        for group_index, (imports, group_generations) in enumerate(
            sorted(groups.items())
        ):
            verification = config.verification.model_copy(
                update={"imports": list(imports)}
            )
            pool = VerificationPool(
                VerificationPoolConfig(
                    lean_project_path=verification.lean_project_path,
                    imports=imports,
                    timeout=verification.timeout_seconds,
                    warmup_timeout=verification.warmup_timeout_seconds,
                    num_workers=verification.num_workers,
                    queue_maxsize=verification.queue_maxsize,
                    heartbeat_interval=verification.heartbeat_interval_seconds,
                    heartbeat_timeout=verification.heartbeat_timeout_seconds,
                    max_worker_restarts=verification.max_worker_restarts,
                    max_task_retries=verification.max_task_retries,
                    shutdown_timeout=verification.shutdown_timeout_seconds,
                    task_spool_dir=str(
                        output / "verification_tasks" / f"group_{group_index:03d}"
                    ),
                    save_full_source_on_failure_only=(
                        verification.save_full_source_on_failure_only
                    ),
                )
            )
            try:
                pool.start()
                verify_candidates(
                    group_generations,
                    by_statement,
                    config=verification,
                    output_path=verification_path,
                    cache_path=(
                        output
                        / f"verification_cache_group_{group_index:03d}.sqlite"
                    ),
                    force=False,
                    verification_pool=pool,
                    collect_results=False,
                )
                group_runtime.append(
                    {
                        "group_index": group_index,
                        "imports": list(imports),
                        "generation_count": len(group_generations),
                        "runtime": pool.runtime_snapshot(),
                    }
                )
            finally:
                pool.close()
    verification_seconds = round(time.monotonic() - verification_start, 4)
    verifications = [
        VerificationRecord.model_validate(row)
        for row in iter_jsonl(verification_path)
    ]
    if len(verifications) != len(generations):
        raise ValueError(
            f"generation/verification mismatch: {len(generations)} vs "
            f"{len(verifications)}"
        )
    resource = {
        "generation": generation_resource,
        "verification": verification_resource,
    }
    summaries: dict[str, Any] = {}
    for dataset in expected_counts:
        dataset_generations = [
            row
            for row in generations
            if dataset_name(by_statement[row.statement_id]) == dataset
        ]
        dataset_verifications = [
            row
            for row in verifications
            if dataset_name(by_statement[row.statement_id]) == dataset
        ]
        dataset_output = output / dataset
        write_jsonl_atomic(
            dataset_output / "generations.jsonl",
            [row.model_dump(mode="json") for row in dataset_generations],
        )
        write_jsonl_atomic(
            dataset_output / "verifications.jsonl",
            [row.model_dump(mode="json") for row in dataset_verifications],
        )
        summary, attempts = metrics_for(
            dataset=dataset,
            statements=statements,
            generations=generations,
            verifications=verifications,
            generation_seconds=generation_seconds,
            verification_seconds=verification_seconds,
            resource=resource,
            group_runtime=group_runtime,
        )
        summary.update(
            {
                "model_name": args.model_name,
                "model_path": args.model,
                "adapter_path": args.adapter,
                "max_new_tokens": args.max_new_tokens,
                "generation_seed": args.seed,
                "generation_backend": generation_result.get("backend"),
                "execution_mode": "isolated_generation_then_grouped_pantograph",
            }
        )
        write_jsonl_atomic(dataset_output / "attempts.jsonl", attempts)
        write_json_atomic(dataset_output / "metrics.json", summary)
        summaries[dataset] = summary
    write_json_atomic(
        output / "runtime/resources.json",
        {
            "generation": generation_resource,
            "verification": verification_resource,
            "generation_seconds": generation_seconds,
            "verification_seconds": verification_seconds,
            "pantograph_import_groups": group_runtime,
        },
    )
    write_json_atomic(
        output / "run_summary.json",
        {
            "model_name": args.model_name,
            "max_new_tokens": args.max_new_tokens,
            "summaries": summaries,
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
