#!/usr/bin/env python3
"""Attest statement elaboration for protected monitor/benchmark datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.contracts import make_attestation_id
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    compose_lean_theorem,
)
from lean_prover.lean_training.expert_iteration.utils import (
    environment_identity,
    file_sha256,
)
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def update_manifest(
    output_dir: Path,
    *,
    monitor_count: int,
    benchmark_count: int,
    results: list[dict[str, Any]],
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.setdefault("actual_sizes", {}).update(
        {"monitor": monitor_count, "benchmark": benchmark_count}
    )
    manifest["statement_verification_report"] = (
        "audit/statement_verification_report.json"
    )
    manifest["statement_verification_results"] = (
        "audit/statement_verification_results.jsonl"
    )
    manifest["statement_only_roles"] = ["monitor", "benchmark"]
    manifest["statement_migration_rules"] = sorted(
        {
            rule
            for result in results
            for rule in result.get("migration_rules", [])
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def migrate_minif2f_context(row: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Apply only explicit target-version migrations with recorded rule IDs."""

    statement = str(row.get("lean_statement") or "").strip()
    context_lines = list(row.get("context_lines") or ())
    rules: list[str] = []
    if "open BigOperators" in context_lines and "open scoped BigOperators" not in context_lines:
        context_lines.append("open scoped BigOperators")
        rules.append("minif2f_bigoperators_scope_lean4_29_v1")
    if (
        row.get("id") == "amc12a_2020_p15"
        and "Complex.abs (a - b)" in statement
    ):
        statement = statement.replace("Complex.abs (a - b)", "‖a - b‖")
        rules.append("minif2f_complex_abs_to_norm_mathlib_5e932f9_v1")
    statement, replacements = re.subn(
        r"([∑∏]\s+[^\s,]+)\s+in\s+",
        r"\1 ∈ ",
        statement,
    )
    if replacements:
        rules.append("minif2f_finset_bigop_in_to_mem_lean4_29_v1")
    return statement, context_lines, rules


def statement_elaborated(result: dict[str, Any]) -> tuple[bool, str]:
    if result.get("success"):
        return True, "pantograph_success"
    diagnostics = str(result.get("diagnostics") or "")
    warning_only = (
        not result.get("timed_out")
        and "error:" not in diagnostics.lower()
        and "warning:" in diagnostics.lower()
    )
    return warning_only, "sorry_warning_only" if warning_only else "elaboration_error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-input", required=True)
    parser.add_argument("--benchmark-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--expected-monitor-count", type=int, default=32)
    parser.add_argument("--expected-benchmark-count", type=int, default=96)
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="reuse a successful existing statement report without starting Pantograph",
    )
    args = parser.parse_args()

    sources = {
        "monitor": Path(args.monitor_input),
        "benchmark": Path(args.benchmark_input),
    }
    output_dir = Path(args.output_dir)
    if args.reports_only:
        report = json.loads(
            (output_dir / "audit/statement_verification_report.json").read_text(
                encoding="utf-8"
            )
        )
        results = read_jsonl(output_dir / "audit/statement_verification_results.jsonl")
        monitor = read_jsonl(output_dir / "monitor.jsonl")
        benchmark = read_jsonl(output_dir / "benchmark.jsonl")
        current_hashes = {role: file_sha256(path) for role, path in sources.items()}
        if (
            report.get("success") is not True
            or current_hashes != report.get("input_hashes_after")
            or len(monitor) != args.expected_monitor_count
            or len(benchmark) != args.expected_benchmark_count
            or len(results)
            != args.expected_monitor_count + args.expected_benchmark_count
            or any(row.get("statement_verified") is not True for row in monitor + benchmark)
            or any(row.get("proof_verified") is not False for row in monitor + benchmark)
        ):
            raise RuntimeError("existing statement verification artifacts are not reusable")
        update_manifest(
            output_dir,
            monitor_count=len(monitor),
            benchmark_count=len(benchmark),
            results=results,
        )
        print(
            json.dumps(
                {
                    "success": True,
                    "reports_only": True,
                    "monitor": len(monitor),
                    "benchmark": len(benchmark),
                    "original_files_unchanged": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    input_hashes_before = {role: file_sha256(path) for role, path in sources.items()}
    identity = environment_identity(args.project, ("Mathlib",))
    tasks: list[VerificationTask] = []
    rows_by_generation: dict[str, tuple[dict[str, Any], str, list[str]]] = {}
    index = 0
    for role, path in sources.items():
        for source_index, row in enumerate(read_jsonl(path)):
            statement, context_lines, migration_rules = migrate_minif2f_context(row)
            if not statement:
                raise ValueError(f"{role} row {source_index} has no lean_statement")
            generation_id = f"statement-only:{role}:{source_index}"
            declaration = compose_lean_theorem(
                statement,
                "by\n  sorry",
                proof_format=ProofFormat.FULL_PROOF,
            )
            task = VerificationTask(
                priority=index,
                problem_index=index,
                attempt_index=0,
                problem_id=str(row.get("id") or f"{role}-{source_index}"),
                prompt=str(row.get("prompt") or ""),
                generated_proof="by\n  sorry",
                raw_completion="by\n  sorry",
                lean_code=declaration,
                imports=tuple(row.get("imports") or ("Mathlib",)),
                context_lines=tuple(context_lines),
                payload={
                    "generation_id": generation_id,
                    "dataset_role": role,
                    "source_index": source_index,
                },
                reject_forbidden=False,
            )
            assembled = build_labeled_lean_code(task, include_imports=True)
            task.payload["assembled_source_hash"] = sha256_text(assembled)
            tasks.append(task)
            rows_by_generation[generation_id] = (row, statement, migration_rules)
            index += 1

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
    try:
        run = pool.run_batch(tasks)
        verified: dict[str, list[dict[str, Any]]] = {role: [] for role in sources}
        quarantined: list[dict[str, Any]] = []
        results = []
        for result in run.results:
            generation_id = str(result["generation_id"])
            role = str(result["dataset_role"])
            row, migrated_statement, migration_rules = rows_by_generation[generation_id]
            record_id = str(row.get("id") or generation_id)
            assembled_hash = str(result["assembled_source_hash"])
            elaborated, acceptance_reason = statement_elaborated(result)
            enriched = {
                **row,
                "original_lean_statement": row.get("lean_statement"),
                "lean_statement": migrated_statement,
                "data_role": role,
                "data_state": "verified" if elaborated else "quarantined",
                "record_id": record_id,
                "statement_verified": elaborated,
                "proof_verified": False,
                "pantograph_verified": elaborated,
                "verification_scope": "statement_only",
                "verification_method": "pantograph_elaboration_with_by_sorry",
                "statement_acceptance_reason": acceptance_reason,
                "migration_applied": bool(migration_rules),
                "migration_rules": migration_rules,
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": identity["environment_hash"],
                "assembler_version": ASSEMBLER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "assembled_source_hash": assembled_hash,
                "attestation_id": (
                    make_attestation_id(
                        record_id=record_id,
                        environment_hash=str(identity["environment_hash"]),
                        assembler_version=ASSEMBLER_VERSION,
                        normalization_version=NORMALIZATION_VERSION,
                        assembled_source_hash=assembled_hash,
                    )
                    if elaborated
                    else None
                ),
            }
            results.append(
                {
                    **result,
                    "statement_elaborated": elaborated,
                    "statement_acceptance_reason": acceptance_reason,
                    "migration_rules": migration_rules,
                }
            )
            if elaborated:
                verified[role].append(enriched)
            else:
                quarantined.append(
                    {
                        **enriched,
                        "verification_error_message": result.get("diagnostics"),
                    }
                )

        write_jsonl(output_dir / "monitor.jsonl", verified["monitor"])
        write_jsonl(output_dir / "benchmark.jsonl", verified["benchmark"])
        write_jsonl(output_dir / "audit/statement_quarantined.jsonl", quarantined)
        write_jsonl(output_dir / "audit/statement_verification_results.jsonl", results)
        input_hashes_after = {role: file_sha256(path) for role, path in sources.items()}
        report = {
            "success": (
                len(verified["monitor"]) == args.expected_monitor_count
                and len(verified["benchmark"]) == args.expected_benchmark_count
                and not quarantined
                and input_hashes_before == input_hashes_after
                and not run.fatal_errors
            ),
            "monitor": {
                "input": args.expected_monitor_count,
                "statement_verified": len(verified["monitor"]),
            },
            "benchmark": {
                "input": args.expected_benchmark_count,
                "statement_verified": len(verified["benchmark"]),
            },
            "proof_verified": 0,
            "quarantined": len(quarantined),
            "verification_scope": "statement_only",
            "by_sorry_is_not_proof_attestation": True,
            "environment": identity,
            "original_files_unchanged": input_hashes_before == input_hashes_after,
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "runtime_stats": run.runtime_stats,
            "fatal_errors": run.fatal_errors,
        }
        (output_dir / "audit/statement_verification_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        update_manifest(
            output_dir,
            monitor_count=len(verified["monitor"]),
            benchmark_count=len(verified["benchmark"]),
            results=results,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(0 if report["success"] else 1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
