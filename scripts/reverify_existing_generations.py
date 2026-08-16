#!/usr/bin/env python3
"""Reverify persisted discovery and benchmark generations without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    compose_lean_theorem,
    normalize_proof_for_assembly,
)
from lean_prover.lean_training.evaluation.benchmark import (
    extract_proof_body,
    split_prompt_sections,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def build_task(
    *,
    generation_id: str,
    dataset_role: str,
    problem_id: str,
    problem_index: int,
    attempt_index: int,
    prompt: str,
    raw_output: str,
    statement: str,
    imports: Iterable[str],
    context_lines: Iterable[str],
    checkpoint: str | None,
    finish_reason: str | None,
    old_status: str | None,
    old_success: bool,
    priority: int,
    environment_hash: str,
) -> VerificationTask:
    extracted = extract_proof_body(raw_output)
    if not extracted:
        raise ValueError("proof extractor returned an empty proof")
    normalized, proof_format = normalize_proof_for_assembly(extracted)
    declaration = compose_lean_theorem(
        statement,
        normalized,
        proof_format=proof_format,
    )
    task = VerificationTask(
        priority=priority,
        problem_index=problem_index,
        attempt_index=attempt_index,
        problem_id=problem_id,
        prompt=prompt,
        generated_proof=normalized,
        raw_completion=raw_output,
        lean_code=declaration,
        imports=tuple(imports) or ("Mathlib",),
        context_lines=tuple(context_lines),
        payload={
            "generation_id": generation_id,
            "dataset_role": dataset_role,
            "statement": statement,
            "raw_output": raw_output,
            "extracted_proof": extracted,
            "normalized_proof": normalized,
            "proof_format": proof_format.value,
            "checkpoint": checkpoint,
            "finish_reason": finish_reason,
            "old_status": old_status,
            "old_success": old_success,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "environment_hash": environment_hash,
        },
        reject_forbidden=True,
    )
    assembled = build_labeled_lean_code(task, include_imports=True)
    task.payload["assembled_source"] = assembled
    task.payload["assembled_source_hash"] = sha256_text(assembled)
    return task


def summarize(rows: list[dict[str, Any]], dataset_role: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("dataset_role") == dataset_role]
    successes = sum(bool(row.get("success")) for row in selected)
    return {
        "total": len(selected),
        "successes": successes,
        "success_rate": successes / max(1, len(selected)),
        "old_successes": sum(bool(row.get("old_success")) for row in selected),
        "status_distribution": dict(Counter(str(row.get("status")) for row in selected)),
        "old_status_distribution": dict(
            Counter(str(row.get("old_status")) for row in selected)
        ),
        "finish_reason_distribution": dict(
            Counter(str(row.get("finish_reason")) for row in selected)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project", default="lean_project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "reverification_results.jsonl"
    discovery_generation_path = run_dir / "iteration_000/discovery/generations.jsonl"
    benchmark_attempts_path = run_dir / "benchmark/attempts.jsonl"
    input_hashes_before = {
        "discovery_generations": sha256_file(discovery_generation_path),
        "benchmark_attempts": sha256_file(benchmark_attempts_path),
    }
    identity = environment_identity(args.project, ("Mathlib",))
    environment_hash = str(identity["environment_hash"])
    completed = {
        str(row.get("generation_id")): row
        for row in read_jsonl(results_path)
        if row.get("environment_hash") == environment_hash
    } if results_path.exists() else {}

    old_discovery = {
        str(row["generation_id"]): row
        for row in read_jsonl(run_dir / "iteration_000/discovery/verifications.jsonl")
    }
    statements = {
        str(row["statement_id"]): row
        for row in read_jsonl(run_dir / "iteration_000/discovery/selected_pool.jsonl")
    }
    tasks: list[VerificationTask] = []
    local_results: list[dict[str, Any]] = []
    priority = 0
    discovery_generations = read_jsonl(discovery_generation_path)
    benchmark_attempts = read_jsonl(benchmark_attempts_path)
    expected_total = len(discovery_generations) + len(benchmark_attempts)
    for row in discovery_generations:
        generation_id = str(row["generation_id"])
        if generation_id in completed:
            continue
        statement_row = statements[str(row["statement_id"])]
        old = old_discovery.get(generation_id, {})
        try:
            task = build_task(
                generation_id=generation_id,
                dataset_role="discovery",
                problem_id=str(row["statement_id"]),
                problem_index=priority // 4,
                attempt_index=int(row.get("sample_index") or 0),
                prompt=str(row.get("prompt") or ""),
                raw_output=str(row.get("raw_output") or ""),
                statement=str(statement_row["statement"]),
                imports=statement_row.get("imports") or ("Mathlib",),
                context_lines=statement_row.get("context_lines") or (),
                checkpoint=row.get("checkpoint"),
                finish_reason=row.get("finish_reason")
                or (row.get("metadata") or {}).get("finish_reason"),
                old_status=old.get("status"),
                old_success=bool(old.get("verified") or old.get("success")),
                priority=priority,
                environment_hash=environment_hash,
            )
            tasks.append(task)
        except Exception as error:
            local_results.append(
                {
                    "generation_id": generation_id,
                    "dataset_role": "discovery",
                    "success": False,
                    "status": "extraction_or_assembly_error",
                    "diagnostics": f"{type(error).__name__}: {error}",
                    "old_status": old.get("status"),
                    "old_success": bool(old.get("verified") or old.get("success")),
                    "finish_reason": row.get("finish_reason")
                    or (row.get("metadata") or {}).get("finish_reason"),
                    "assembler_version": ASSEMBLER_VERSION,
                    "normalization_version": NORMALIZATION_VERSION,
                    "environment_hash": environment_hash,
                }
            )
        priority += 1

    for row in benchmark_attempts:
        generation_id = f"benchmark:{row['problem_index']}:{row['attempt_index']}"
        if generation_id in completed:
            continue
        sections = split_prompt_sections(str(row.get("prompt") or ""))
        try:
            task = build_task(
                generation_id=generation_id,
                dataset_role="benchmark",
                problem_id=str(row["problem_id"]),
                problem_index=int(row["problem_index"]),
                attempt_index=int(row["attempt_index"]),
                prompt=str(row.get("prompt") or ""),
                raw_output=str(row.get("raw_completion") or ""),
                statement=sections["lean_statement"],
                imports=row.get("imports") or ("Mathlib",),
                context_lines=row.get("context_lines") or (),
                checkpoint=str(
                    row.get("checkpoint")
                    or run_dir / "initial_sft/merged_anchor"
                ),
                finish_reason=row.get("finish_reason"),
                old_status=row.get("status"),
                old_success=bool(row.get("success")),
                priority=priority,
                environment_hash=environment_hash,
            )
            tasks.append(task)
        except Exception as error:
            local_results.append(
                {
                    "generation_id": generation_id,
                    "dataset_role": "benchmark",
                    "success": False,
                    "status": "extraction_or_assembly_error",
                    "diagnostics": f"{type(error).__name__}: {error}",
                    "old_status": row.get("status"),
                    "old_success": bool(row.get("success")),
                    "finish_reason": row.get("finish_reason"),
                    "assembler_version": ASSEMBLER_VERSION,
                    "normalization_version": NORMALIZATION_VERSION,
                    "environment_hash": environment_hash,
                }
            )
        priority += 1

    for row in local_results:
        append_jsonl(results_path, row)
        completed[str(row["generation_id"])] = row

    processed = len(completed)
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.startup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=8,
            max_worker_restarts=3,
            max_task_retries=1,
            shutdown_timeout=15,
        )
    )

    def persist(result: dict[str, Any]) -> None:
        nonlocal processed
        append_jsonl(results_path, result)
        completed[str(result["generation_id"])] = result
        processed += 1
        if processed % 50 == 0:
            print(
                json.dumps(
                    {
                        "phase": "reverification",
                        "completed": processed,
                        "expected": expected_total,
                    }
                ),
                flush=True,
            )

    try:
        run = pool.run_batch(tasks, on_result=persist) if tasks else None
        if run is not None and run.fatal_errors:
            raise RuntimeError(f"Pantograph pool failed: {run.fatal_errors}")
        rows = [
            row
            for row in read_jsonl(results_path)
            if row.get("environment_hash") == environment_hash
        ]
        input_hashes_after = {
            "discovery_generations": sha256_file(discovery_generation_path),
            "benchmark_attempts": sha256_file(benchmark_attempts_path),
        }
        summary = {
            "success": len(rows) == expected_total
            and input_hashes_before == input_hashes_after,
            "environment": identity,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "original_generation_files_unchanged": (
                input_hashes_before == input_hashes_after
            ),
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "discovery": summarize(rows, "discovery"),
            "benchmark": summarize(rows, "benchmark"),
            "overall": {
                "total": len(rows),
                "successes": sum(bool(row.get("success")) for row in rows),
                "success_rate": sum(bool(row.get("success")) for row in rows)
                / max(1, len(rows)),
            },
            "runtime_stats": run.runtime_stats if run is not None else {},
            "fatal_errors": run.fatal_errors if run is not None else [],
        }
        (output_dir / "reverification_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(0 if summary["success"] else 1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
