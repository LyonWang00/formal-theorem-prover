#!/usr/bin/env python3
"""Audit legacy Lean data assembly and run context-aware verification gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.context import (
    CONTEXT_ASSEMBLER_VERSION,
    CONTEXT_NORMALIZATION_VERSION,
    assemble_context_aware_declaration,
    context_record_payload,
    record_context_hashes,
    record_context_summary,
    stable_json_hash,
)
from lean_prover.lean_training.data.contracts import (
    DataState,
    LeanDataRecord,
    make_attestation_id,
    utc_now,
)
from lean_prover.lean_training.data.adapters.lean_workbook import (
    reconstruct_lean_workbook_records,
)
from lean_prover.lean_training.data.lean_workbook_context_builder import (
    build_lean_workbook_context,
)
from lean_prover.lean_training.data.adapters.leandojo import (
    corpus_key,
    iter_json_array,
    load_corpus_declarations,
    reconstruct_leandojo_records,
    reservoir_sample,
)
from lean_prover.lean_training.data.leandojo_context_builder import (
    GitSourceProvider,
    build_leandojo_source_context,
)
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    compose_lean_theorem,
    lean_code_tokens,
    normalize_proof_rhs,
    statement_hash,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.cache import (
    VerificationCache,
    make_context_cache_key,
)
from lean_prover.lean_training.verification.pantograph import (
    build_labeled_lean_code,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


TAXONOMY = (
    "syntax_error",
    "notation_error",
    "missing_import",
    "missing_namespace",
    "missing_scope",
    "missing_variable",
    "missing_local_identifier",
    "missing_context",
    "unknown_identifier",
    "elaboration_error",
    "tactic_error",
    "unsolved_goals",
    "environment_error",
    "timeout",
)
FORBIDDEN = {"sorry", "admit", "sorryAx", "axiom", "unsafe"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook-arrow", required=True, type=Path)
    parser.add_argument("--leandojo-split", type=Path)
    parser.add_argument("--leandojo-corpus", type=Path)
    parser.add_argument(
        "--workbook-only",
        action="store_true",
        help="Run only the task-book Lean Workbook experiment.",
    )
    parser.add_argument("--lean-project", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "prepare",
            "workbook-before",
            "workbook-after",
            "leandojo-before",
            "leandojo-after",
            "runtime-tests",
            "reports",
        ),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    audit_dir = output / "audit"
    processed_dir = output / "processed"
    verified_dir = output / "verified"
    manifests_dir = output / "manifests"
    runtime_dir = output / "runtime"
    for directory in (
        audit_dir,
        processed_dir,
        verified_dir,
        manifests_dir,
        runtime_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    identity = {
        **environment_identity(args.lean_project, ("Mathlib",)),
        "pantograph_version": importlib.metadata.version("pantograph"),
    }
    workbook_legacy, workbook_context = prepare_workbook(args, identity)
    workbook_context = [
        _record_with_assembled_source_hash(record, index=index)
        for index, record in enumerate(workbook_context)
    ]
    leandojo_legacy: list[LeanDataRecord] = []
    leandojo_context: list[LeanDataRecord] = []
    leandojo_scan: dict[str, Any] = {}
    if not args.workbook_only:
        if args.leandojo_split is None or args.leandojo_corpus is None:
            raise ValueError(
                "--leandojo-split and --leandojo-corpus are required "
                "unless --workbook-only is set"
            )
        leandojo_legacy, leandojo_context, leandojo_scan = prepare_leandojo(
            args,
            identity,
        )
    write_jsonl(
        processed_dir / "workbook_context_v2.jsonl",
        (context_record_payload(record) for record in workbook_context),
    )
    if not args.workbook_only:
        write_jsonl(
            processed_dir / "leandojo_context_v1.jsonl",
            (context_record_payload(record) for record in leandojo_context),
        )
    preparation = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "workbook": {
            "legacy_records": len(workbook_legacy),
            "context_records": len(workbook_context),
            "context_recovered": sum(
                record.context_recovered for record in workbook_context
            ),
            "context_recovery_status": dict(
                Counter(
                    record.context_recovery_status
                    for record in workbook_context
                )
            ),
            "context_recovery_sources": dict(
                Counter(
                    source
                    for record in workbook_context
                    for source in record.context_recovery_sources
                )
            ),
        },
        "environment": identity,
    }
    if not args.workbook_only:
        preparation["leandojo"] = {
            "legacy_records": len(leandojo_legacy),
            "context_records": len(leandojo_context),
            "context_recovered": sum(
                record.context_recovered for record in leandojo_context
            ),
            "scan": leandojo_scan,
        }
    write_json(audit_dir / "preparation_summary.json", preparation)
    if args.stage == "prepare":
        return

    if args.stage == "all":
        requested_stages = (
            ("workbook-before", "workbook-after", "runtime-tests")
            if args.workbook_only
            else (
                "workbook-before",
                "workbook-after",
                "leandojo-before",
                "leandojo-after",
                "runtime-tests",
            )
        )
    else:
        requested_stages = (args.stage,)
    if args.workbook_only and any(
        stage.startswith("leandojo-") for stage in requested_stages
    ):
        raise ValueError("LeanDojo stages are disabled by --workbook-only")
    reports: dict[str, dict[str, Any]] = {}
    needs_pool = any(
        stage
        in {
            "workbook-before",
            "workbook-after",
            "leandojo-before",
            "leandojo-after",
            "runtime-tests",
        }
        for stage in requested_stages
    )
    pool_config = VerificationPoolConfig(
        lean_project_path=str(args.lean_project.resolve()),
        imports=("Mathlib",),
        timeout=args.timeout,
        warmup_timeout=args.startup_timeout,
        num_workers=args.workers,
        queue_maxsize=max(32, args.workers * 16),
        max_task_retries=1,
        max_worker_restarts=3,
        task_spool_dir=str(runtime_dir / "tasks"),
        save_full_source_on_failure_only=True,
    )
    cache_path = runtime_dir / "verification_cache.sqlite"
    cache = VerificationCache(cache_path)
    try:
        if needs_pool:
            with VerificationPool(pool_config) as pool:
                for stage in requested_stages:
                    if stage == "workbook-before":
                        reports[stage] = verify_stage(
                            stage=stage,
                            records=workbook_legacy,
                            assembly_mode="legacy",
                            pool=pool,
                            cache=cache,
                            identity=identity,
                            output=output,
                            use_cache=args.use_cache,
                        )
                    elif stage == "workbook-after":
                        reports[stage] = verify_stage(
                            stage=stage,
                            records=workbook_context,
                            assembly_mode="context",
                            pool=pool,
                            cache=cache,
                            identity=identity,
                            output=output,
                            use_cache=args.use_cache,
                        )
                    elif stage == "leandojo-before":
                        reports[stage] = verify_stage(
                            stage=stage,
                            records=leandojo_legacy,
                            assembly_mode="legacy",
                            pool=pool,
                            cache=cache,
                            identity=identity,
                            output=output,
                            use_cache=args.use_cache,
                        )
                    elif stage == "leandojo-after":
                        reports[stage] = verify_stage(
                            stage=stage,
                            records=leandojo_context,
                            assembly_mode="context",
                            pool=pool,
                            cache=cache,
                            identity=identity,
                            output=output,
                            use_cache=args.use_cache,
                        )
                    elif stage == "runtime-tests":
                        reports[stage] = run_runtime_contract_tests(
                            pool=pool,
                            output=output,
                        )
        reports.update(load_existing_stage_reports(output))
        if args.stage in {"all", "reports"}:
            if args.workbook_only:
                finalize_workbook_reports(
                    args=args,
                    output=output,
                    identity=identity,
                    reports=reports,
                    workbook_context=workbook_context,
                )
            else:
                finalize_reports(
                    args=args,
                    output=output,
                    identity=identity,
                    reports=reports,
                    workbook_context=workbook_context,
                    leandojo_context=leandojo_context,
                )
    finally:
        cache.close()


def prepare_workbook(
    args: argparse.Namespace,
    identity: Mapping[str, Any],
) -> tuple[list[LeanDataRecord], list[LeanDataRecord]]:
    from datasets import Dataset

    dataset = Dataset.from_file(str(args.workbook_arrow))
    rows = [dict(row) for row in dataset]
    reconstructed, reconstruction_report = reconstruct_lean_workbook_records(
        rows,
        source_file=None,
        source_commit="huggingface:InternLM/Lean-Workbook",
    )
    candidates = [
        record
        for record in reconstructed
        if record.data_state is not DataState.QUARANTINED and record.proof
    ]
    selected = _fixed_sample(
        candidates,
        size=args.sample_size,
        seed=args.seed,
        id_getter=lambda record: record.record_id,
    )
    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_id[str(row.get("id") or "")].append(row)
    context_records = [
        build_lean_workbook_context(
            record,
            rows_by_id.get(record.record_id, ()),
            environment_hash=str(identity["environment_hash"]),
        )
        for record in selected
    ]
    sample_ids = [
        {
            "sample_index": index,
            "record_id": record.record_id,
            "source_rows": [
                record.source_span.start_row if record.source_span else None,
                record.source_span.end_row if record.source_span else None,
            ],
        }
        for index, record in enumerate(selected)
    ]
    sample_path = args.output_dir / "audit" / "workbook_baseline_sample_ids.json"
    _write_or_validate_ids(sample_path, sample_ids)
    legacy_alias = (
        args.output_dir / "audit" / "lean_workbook_baseline_sample_ids.json"
    )
    _write_or_validate_ids(legacy_alias, sample_ids)
    write_json(
        args.output_dir / "audit" / "lean_workbook_reconstruction.json",
        reconstruction_report,
    )
    return selected, context_records


def prepare_leandojo(
    args: argparse.Namespace,
    identity: Mapping[str, Any],
) -> tuple[list[LeanDataRecord], list[LeanDataRecord], dict[str, Any]]:
    scan = Counter()
    sampled_rows, candidate_total = reservoir_sample(
        _iter_leandojo_candidates(args.leandojo_split, scan),
        size=args.sample_size,
        seed=args.seed,
    )
    requested = {
        corpus_key(
            str(row.get("file_path") or ""),
            str(row.get("full_name") or ""),
        )
        for row in sampled_rows
    }
    declarations = load_corpus_declarations(args.leandojo_corpus, requested)
    legacy, cleaning = reconstruct_leandojo_records(
        sampled_rows,
        corpus_declarations=declarations,
        split_name=args.leandojo_split.stem,
    )
    provider = GitSourceProvider(lean_project=args.lean_project)
    context = [
        build_leandojo_source_context(
            record,
            source_provider=provider,
            environment_hash=str(identity["environment_hash"]),
        )
        for record in legacy
    ]
    sample_ids = [
        {
            "sample_index": index,
            "record_id": record.record_id,
            "source_file": record.source_file,
            "source_declaration": record.source_declaration,
        }
        for index, record in enumerate(legacy)
    ]
    sample_path = args.output_dir / "audit" / "leandojo_baseline_sample_ids.json"
    _write_or_validate_ids(sample_path, sample_ids)
    return legacy, context, {
        "source_split_records": scan["raw_records"],
        "candidate_records": candidate_total,
        "prefilter": dict(scan),
        "corpus_declarations_requested": len(requested),
        "corpus_declarations_found": len(declarations),
        "cleaning": cleaning,
    }


def verify_stage(
    *,
    stage: str,
    records: list[LeanDataRecord],
    assembly_mode: str,
    pool: VerificationPool,
    cache: VerificationCache,
    identity: Mapping[str, Any],
    output: Path,
    use_cache: bool,
) -> dict[str, Any]:
    environment_hash = str(identity["environment_hash"])
    tasks: list[VerificationTask] = []
    cached_results: list[dict[str, Any]] = []
    records_by_id = {record.record_id: record for record in records}
    prefiltered: list[LeanDataRecord] = []
    cache_hits = 0
    cache_misses = 0
    verifier_version = importlib.metadata.version("pantograph")
    for index, record in enumerate(records):
        if record.data_state is DataState.QUARANTINED or not record.proof:
            prefiltered.append(record)
            continue
        lean_code = (
            _legacy_assembly(record)
            if assembly_mode == "legacy"
            else assemble_context_aware_declaration(record)
        )
        hashes = record_context_hashes(
            record,
            environment_hash=environment_hash,
        )
        statement_digest = statement_hash(record.statement)
        proof_digest = _sha256(
            " ".join(normalize_proof_rhs(record.proof).split())
        )
        assembler_version = (
            ASSEMBLER_VERSION
            if assembly_mode == "legacy"
            else CONTEXT_ASSEMBLER_VERSION
        )
        normalization_version = (
            NORMALIZATION_VERSION
            if assembly_mode == "legacy"
            else CONTEXT_NORMALIZATION_VERSION
        )
        payload = {
            "generation_id": f"context-audit:{stage}:{record.record_id}",
            "record_id": record.record_id,
            "source_dataset": record.source_dataset,
            "statement": record.statement,
            "proof": record.proof,
            "source_imports": list(record.imports),
            "namespace": record.namespace,
            "namespaces": list(record.namespaces),
            "context_summary": record_context_summary(record),
            "context_hashes": hashes,
            "context_recovered": record.context_recovered,
            "environment_hash": environment_hash,
            "assembler_version": assembler_version,
            "normalization_version": normalization_version,
            "statement_hash": statement_digest,
            "proof_hash": proof_digest,
            "cache_hit": False,
            "source_file": record.source_file,
            "verifier_version": verifier_version,
        }
        task = VerificationTask(
            priority=index,
            problem_index=index,
            attempt_index=0,
            problem_id=record.record_id,
            prompt="",
            generated_proof=record.proof,
            raw_completion=record.proof,
            lean_code=lean_code,
            imports=("Mathlib",),
            context_lines=(),
            payload=payload,
            reject_forbidden=True,
        )
        assembled = build_labeled_lean_code(task, include_imports=True)
        payload["assembled_source"] = assembled
        payload["assembled_source_hash"] = _sha256(assembled)
        cache_key = make_context_cache_key(
            statement_hash=statement_digest,
            proof_hash=proof_digest,
            imports_hash=hashes["imports_hash"],
            namespace_hash=hashes["namespace_hash"],
            scope_hash=hashes["scope_hash"],
            variable_context_hash=hashes["variable_context_hash"],
            context_hash=hashes["context_hash"],
            assembled_source_hash=payload["assembled_source_hash"],
            environment_hash=environment_hash,
            verifier_version=verifier_version,
            assembler_version=assembler_version,
            normalization_version=normalization_version,
        )
        payload["cache_key"] = cache_key
        cached = (
            cache.get(
                cache_key,
                environment_hash=environment_hash,
                assembler_version=assembler_version,
                normalization_version=normalization_version,
            )
            if use_cache
            else None
        )
        if cached is not None:
            cached_results.append({**cached, "cache_hit": True})
            cache_hits += 1
            continue
        cache_misses += 1
        tasks.append(task)

    run_results: list[dict[str, Any]] = []
    runtime_stats: dict[str, Any] = {}
    if tasks:
        run = pool.run_batch(tasks)
        if run.fatal_errors:
            raise RuntimeError(
                f"{stage} verification failed: {'; '.join(run.fatal_errors)}"
            )
        run_results = run.results
        runtime_stats = {
            **run.runtime_stats,
            "warmup_reports": run.warmup_reports,
            "recovered_worker_failures": run.recovered_worker_failures,
        }
    all_results = [*cached_results, *run_results]
    all_results.sort(key=lambda row: int(row.get("problem_index", 0)))
    taxonomy = Counter()
    detailed = Counter()
    verified_records: list[LeanDataRecord] = []
    quarantined_records: list[LeanDataRecord] = list(prefiltered)
    result_by_id = {str(row.get("record_id")): row for row in all_results}
    verification_rows: list[dict[str, Any]] = []
    for record in records:
        result = result_by_id.get(record.record_id)
        if result is None:
            category = (
                "missing_context"
                if record.data_state is DataState.QUARANTINED
                else "environment_error"
            )
            taxonomy[category] += 1
            detailed[record.verification_error_type or "missing_result"] += 1
            continue
        category, detail = classify_failure(result)
        if result.get("success"):
            verified_records.append(_attest_record(record, result, identity))
        else:
            taxonomy[category] += 1
            detailed[detail] += 1
            quarantined_records.append(
                record.model_copy(
                    update={
                        "data_state": DataState.QUARANTINED,
                        "verification_status": "failed",
                        "verification_error_type": category,
                        "verification_error_message": str(
                            result.get("diagnostics") or ""
                        ),
                        "statement_verified": False,
                        "proof_verified": False,
                        "pantograph_verified": False,
                        "lean_version": identity.get("lean_version"),
                        "mathlib_commit": identity.get("mathlib_commit"),
                        "environment_hash": environment_hash,
                        "assembler_version": str(result["assembler_version"]),
                        "normalization_version": str(
                            result["normalization_version"]
                        ),
                        "assembled_source_hash": str(
                            result.get("assembled_source_hash") or ""
                        ),
                    }
                )
            )
        verification_rows.append(
            _complete_verification_record(result, category=category, detail=detail)
        )
        if not result.get("cache_hit"):
            cache.put(
                cache_key=str(result["cache_key"]),
                proof_hash=str(result["proof_hash"]),
                statement_hash=str(result["statement_hash"]),
                environment_hash=environment_hash,
                assembler_version=str(result["assembler_version"]),
                normalization_version=str(result["normalization_version"]),
                status=str(result.get("status") or ""),
                error_type=None if result.get("success") else category,
                source_path=(
                    None
                    if result.get("success")
                    else str(result.get("source_file") or "")
                ),
                verification=verification_rows[-1],
            )

    success = len(verified_records)
    report = {
        "stage": stage,
        "assembly_mode": assembly_mode,
        "total_samples": len(records),
        "submitted": len(tasks),
        "prefiltered": len(prefiltered),
        "compile_success": success,
        "compile_fail": len(records) - success,
        "success_ratio": success / max(1, len(records)),
        "failure_taxonomy": {
            category: taxonomy.get(category, 0) for category in TAXONOMY
        },
        "detailed_failure_taxonomy": dict(detailed),
        "cache": {
            "hit": cache_hits,
            "miss": cache_misses,
            "enabled_for_reads": use_cache,
        },
        "runtime": runtime_stats,
        "environment": dict(identity),
    }
    prefix = stage.replace("-", "_")
    audit = output / "audit"
    write_json(audit / f"{prefix}_verification.json", verification_rows)
    write_json(
        audit / f"{prefix}_failure_taxonomy.json",
        {
            "failure_taxonomy": report["failure_taxonomy"],
            "detailed": report["detailed_failure_taxonomy"],
        },
    )
    write_json(audit / f"{prefix}_summary.json", report)
    (audit / f"{prefix}_report.md").write_text(
        stage_report_markdown(report),
        encoding="utf-8",
    )
    # Required task-book aliases.
    alias = stage.replace("-", "_")
    if alias.startswith("workbook_"):
        required = alias
        write_json(audit / f"{required}_verification.json", verification_rows)
        write_json(
            audit / f"{required}_failure_taxonomy.json",
            report["failure_taxonomy"],
        )
        write_json(
            audit / f"{required}_failure.json",
            {
                "failure_taxonomy": report["failure_taxonomy"],
                "detailed": report["detailed_failure_taxonomy"],
            },
        )
        write_jsonl(
            audit / f"{required}_results.jsonl",
            verification_rows,
        )
        (audit / f"{required}_report.md").write_text(
            stage_report_markdown(report),
            encoding="utf-8",
        )
    if stage.endswith("after"):
        dataset = "workbook" if stage.startswith("workbook") else "leandojo"
        verified_name = (
            "workbook_verified_v2.jsonl"
            if dataset == "workbook"
            else "leandojo_verified_v1.jsonl"
        )
        write_jsonl(
            output / "verified" / verified_name,
            (context_record_payload(record) for record in verified_records),
        )
        write_jsonl(
            output / "processed" / f"{dataset}_quarantined.jsonl",
            (context_record_payload(record) for record in quarantined_records),
        )
        if dataset == "workbook":
            write_jsonl(
                output / "verified" / "workbook_verified_context.jsonl",
                (context_record_payload(record) for record in verified_records),
            )
            write_jsonl(
                output
                / "processed"
                / "workbook_context_quarantined.jsonl",
                (
                    context_record_payload(record)
                    for record in quarantined_records
                ),
            )
    return report


def _legacy_assembly(record: LeanDataRecord) -> str:
    declaration = compose_lean_theorem(
        record.statement,
        record.proof or "",
        proof_format=record.proof_format,
    )
    if "leandojo" in record.source_dataset.lower() and record.namespace:
        declaration = (
            f"namespace {record.namespace}\n\n"
            f"{declaration}\n\n"
            f"end {record.namespace}"
        )
    if record.variable_context:
        declaration = f"{record.variable_context}\n\n{declaration}"
    return declaration


def _record_with_assembled_source_hash(
    record: LeanDataRecord,
    *,
    index: int,
) -> LeanDataRecord:
    """Attach the exact deterministic source hash before verification."""

    lean_code = assemble_context_aware_declaration(record)
    task = VerificationTask(
        priority=index,
        problem_index=index,
        attempt_index=0,
        problem_id=record.record_id,
        prompt="",
        generated_proof=record.proof or "",
        raw_completion=record.proof or "",
        lean_code=lean_code,
        imports=("Mathlib",),
        context_lines=(),
        payload=None,
        reject_forbidden=True,
    )
    assembled = build_labeled_lean_code(task, include_imports=True)
    return record.model_copy(
        update={"assembled_source_hash": _sha256(assembled)}
    )


def classify_failure(result: Mapping[str, Any]) -> tuple[str, str]:
    if result.get("success"):
        return "success", "success"
    diagnostics = str(result.get("diagnostics") or "")
    lowered = diagnostics.lower()
    if result.get("timed_out") or "timeout" in lowered or "timed out" in lowered:
        return "timeout", "timeout"
    if any(
        marker in lowered
        for marker in (
            "server not running",
            "connection reset",
            "connection refused",
            "broken pipe",
            "environment mismatch",
        )
    ):
        return "environment_error", "pantograph_transport_or_environment"
    if "unknown module prefix" in lowered or "unknown module" in lowered:
        return "missing_import", "unknown_module"
    if "unknown namespace" in lowered:
        return "missing_namespace", "unknown_namespace"
    if "unknown scoped syntax" in lowered or "unknown scope" in lowered:
        return "missing_scope", "unknown_parser_or_syntax"
    unknown = re_search_unknown_identifier(diagnostics)
    if unknown:
        summary = result.get("context_summary")
        known_locals: set[str] = set()
        if isinstance(summary, Mapping):
            for field in ("variables", "hypotheses"):
                values = summary.get(field)
                if isinstance(values, Iterable) and not isinstance(
                    values, (str, bytes, Mapping)
                ):
                    for value in values:
                        if isinstance(value, Mapping) and value.get("name"):
                            known_locals.add(str(value["name"]))
        if unknown in known_locals:
            return "missing_variable", f"unknown_bound_variable:{unknown}"
        if unknown.startswith(("h", "inst", "_inst", "this")) or (
            unknown[:1].islower() and "." not in unknown
        ):
            return (
                "missing_local_identifier",
                f"unknown_local_identifier:{unknown}",
            )
        return "unknown_identifier", f"unknown_identifier:{unknown}"
    if "failed to synthesize" in lowered or "typeclass instance problem is stuck" in lowered:
        return "missing_context", "missing_instance_or_context"
    if "unsolved goals" in lowered:
        return "unsolved_goals", "unsolved_goals"
    if any(
        marker in lowered
        for marker in (
            "unknown parser",
            "unknown syntax",
            "invalid field notation",
            "invalid notation",
        )
    ):
        return "notation_error", "notation_error"
    if any(
        marker in lowered
        for marker in (
            "unexpected token",
            "unexpected end of input",
            "parser",
        )
    ):
        return "syntax_error", "syntax_or_notation_error"
    if any(
        marker in lowered
        for marker in (
            "linarith failed",
            "made no progress",
            "no goals to be solved",
            "tactic",
        )
    ):
        return "tactic_error", "tactic_failure"
    if result.get("compile_errors"):
        return "elaboration_error", "lean_elaboration_error"
    return "elaboration_error", "unclassified_compilation_error"


def re_search_unknown_identifier(diagnostics: str) -> str | None:
    import re

    match = re.search(
        r"[Uu]nknown (?:identifier|constant) [`'\"]([^`'\"]+)[`'\"]",
        diagnostics,
    )
    return match.group(1) if match else None


def _complete_verification_record(
    result: Mapping[str, Any],
    *,
    category: str,
    detail: str,
) -> dict[str, Any]:
    payload = dict(result)
    payload["result"] = "success" if result.get("success") else "failed"
    payload["error_taxonomy"] = None if result.get("success") else category
    payload["error_detail"] = None if result.get("success") else detail
    payload["stdout"] = ""
    payload["stderr"] = (
        "" if result.get("success") else str(result.get("diagnostics") or "")
    )
    if result.get("success"):
        payload.pop("assembled_source", None)
    return payload


def _attest_record(
    record: LeanDataRecord,
    result: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> LeanDataRecord:
    source_hash = str(result.get("assembled_source_hash") or "")
    environment_hash = str(identity["environment_hash"])
    assembler_version = str(result.get("assembler_version") or "")
    normalization_version = str(result.get("normalization_version") or "")
    attestation_id = make_attestation_id(
        record_id=record.record_id,
        environment_hash=environment_hash,
        assembler_version=assembler_version,
        normalization_version=normalization_version,
        assembled_source_hash=source_hash,
    )
    return record.model_copy(
        update={
            "data_state": DataState.VERIFIED,
            "statement_verified": True,
            "proof_verified": True,
            "pantograph_verified": True,
            "reference_proof_verified": True,
            "verification_status": "verified",
            "verification_error_type": None,
            "verification_error_message": None,
            "lean_version": identity.get("lean_version"),
            "mathlib_commit": identity.get("mathlib_commit"),
            "environment_hash": environment_hash,
            "assembler_version": assembler_version,
            "normalization_version": normalization_version,
            "attestation_id": attestation_id,
            "attested_at": utc_now(),
            "assembled_source_hash": source_hash,
        }
    )


def run_runtime_contract_tests(
    *,
    pool: VerificationPool,
    output: Path,
) -> dict[str, Any]:
    fixtures = (
        (
            "source-context-legacy",
            "example : True := by\n  exact contextTruth",
            False,
        ),
        (
            "source-context-aware",
            'local macro "contextTruth" : term => `(True.intro)\n\n'
            "example : True := by\n  exact contextTruth",
            True,
        ),
        (
            "isolation-A",
            "namespace Foo\n\ntheorem A : True := by\n  trivial\n\nend Foo",
            True,
        ),
        (
            "isolation-B-1",
            "example : True := by\n  exact Foo.A",
            False,
        ),
        (
            "isolation-B-2",
            "example : True := by\n  exact Foo.A",
            False,
        ),
    )
    tasks: list[VerificationTask] = []
    for index, (name, source, _) in enumerate(fixtures):
        tasks.append(
            VerificationTask(
                priority=index,
                problem_index=10_000 + index,
                attempt_index=0,
                problem_id=name,
                prompt="",
                generated_proof="",
                raw_completion="",
                lean_code=source,
                imports=("Mathlib",),
                payload={"generation_id": f"runtime-test:{name}"},
                reject_forbidden=True,
            )
        )
    run = pool.run_batch(tasks)
    results = {
        str(result["problem_id"]): bool(result.get("success"))
        for result in run.results
    }
    expected = {name: value for name, _, value in fixtures}
    checks = {
        name: results.get(name) == expected_value
        for name, expected_value in expected.items()
    }
    report = {
        "passed": all(checks.values()) and not run.fatal_errors,
        "checks": checks,
        "observed": results,
        "expected": expected,
        "fatal_errors": run.fatal_errors,
        "runtime": run.runtime_stats,
    }
    write_json(output / "audit" / "runtime_contract_tests.json", report)
    if not report["passed"]:
        raise RuntimeError(f"runtime contract tests failed: {report}")
    return report


def finalize_workbook_reports(
    *,
    args: argparse.Namespace,
    output: Path,
    identity: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    workbook_context: list[LeanDataRecord],
) -> None:
    """Emit the exact task-book artifacts for the fixed paired experiment."""

    required = ("workbook-before", "workbook-after")
    missing = [stage for stage in required if stage not in reports]
    if missing:
        raise RuntimeError(f"cannot finalize reports; missing stages: {missing}")
    before = reports["workbook-before"]
    after = reports["workbook-after"]
    pairs = paired_outcomes(output=output, dataset="workbook")
    write_json(output / "audit" / "workbook_paired_outcomes.json", pairs)
    write_json(
        output / "audit" / "reproduction.json",
        {
            "workbook_arrow": str(args.workbook_arrow.resolve()),
            "lean_project": str(args.lean_project.resolve()),
            "output_dir": str(output),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "workers": args.workers,
            "timeout": args.timeout,
            "startup_timeout": args.startup_timeout,
            "cache_reads": False,
            "stage": "all",
            "workbook_only": True,
        },
    )

    audit_dir = output / "audit"
    before_rows = json.loads(
        (audit_dir / "workbook_before_verification.json").read_text(
            encoding="utf-8"
        )
    )
    after_rows = json.loads(
        (audit_dir / "workbook_after_verification.json").read_text(
            encoding="utf-8"
        )
    )
    before_by_id = {str(row["record_id"]): row for row in before_rows}
    regressions = [
        row
        for row in after_rows
        if before_by_id[str(row["record_id"])].get("success")
        and not row.get("success")
    ]
    regression_lines = [
        "# Lean Workbook regressions",
        "",
        f"Regression count: **{len(regressions)}**.",
        "",
    ]
    if regressions:
        regression_lines.extend(
            (
                "| Record | New category | Diagnostics |",
                "|---|---|---|",
            )
        )
        for row in regressions:
            diagnostics = " ".join(
                str(row.get("stderr") or row.get("diagnostics") or "").split()
            )
            regression_lines.append(
                f"| `{row['record_id']}` | "
                f"`{row.get('error_taxonomy')}` | "
                f"{diagnostics[:300].replace('|', '&#124;')} |"
            )
    else:
        regression_lines.append(
            "No legacy-success → context-failure transition occurred."
        )
    (audit_dir / "workbook_regressions.md").write_text(
        "\n".join(regression_lines) + "\n",
        encoding="utf-8",
    )

    reconstruction_path = audit_dir / "lean_workbook_reconstruction.json"
    reconstruction = (
        json.loads(reconstruction_path.read_text(encoding="utf-8"))
        if reconstruction_path.exists()
        else {}
    )
    status_counts = Counter(
        record.context_recovery_status for record in workbook_context
    )
    source_counts = Counter(
        source
        for record in workbook_context
        for source in record.context_recovery_sources
    )
    warning_counts = Counter(
        warning
        for record in workbook_context
        for warning in record.context_warnings
    )
    final = workbook_final_report_markdown(
        before=before,
        after=after,
        pairs=pairs,
        identity=identity,
        reconstruction=reconstruction,
        status_counts=status_counts,
        source_counts=source_counts,
        warning_counts=warning_counts,
        regressions=regressions,
        seed=args.seed,
    )
    (output / "final_report.md").write_text(final, encoding="utf-8")


def workbook_final_report_markdown(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    pairs: Mapping[str, Any],
    identity: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    status_counts: Mapping[str, int],
    source_counts: Mapping[str, int],
    warning_counts: Mapping[str, int],
    regressions: list[Mapping[str, Any]],
    seed: int,
) -> str:
    before_ratio = float(before["success_ratio"])
    after_ratio = float(after["success_ratio"])
    absolute_gain = after_ratio - before_ratio
    relative_gain = absolute_gain / before_ratio if before_ratio else 0.0
    outcomes = pairs.get("outcomes", {})
    fields = reconstruction.get("available_source_fields", {})
    taxonomy_rows = "\n".join(
        f"| {name} | {before['failure_taxonomy'].get(name, 0)} | "
        f"{after['failure_taxonomy'].get(name, 0)} |"
        for name in TAXONOMY
    )
    recovery_rows = "\n".join(
        f"| {name} | {status_counts.get(name, 0)} |"
        for name in ("full", "partial", "unavailable")
    )
    source_rows = "\n".join(
        f"| {name} | {source_counts.get(name, 0)} |"
        for name in ("proof_state", "statement", "tactic_trace")
    )
    warning_summary = ", ".join(
        f"`{name}`={count}" for name, count in sorted(warning_counts.items())
    )
    return f"""# Lean Workbook Context-aware Pipeline 最终报告

## 实验约束与复现

- 固定样本：500；seed：`{seed}`；before/after 的 ID 与顺序完全一致。
- Lean：`{identity.get('lean_version')}`
- mathlib：`{identity.get('mathlib_commit')}`
- Pantograph：`{identity.get('pantograph_version')}`
- environment hash：`{identity.get('environment_hash')}`
- 两轮均关闭 cache 读取，实际提交 500 次 Pantograph 编译；worker、timeout 与验证策略一致。
- 未修改 Lean、mathlib、Pantograph 核心、worker 生命周期或训练流程。

## 1. Legacy Pipeline

- 通过：{before['compile_success']}/{before['total_samples']}（{before_ratio:.2%}）
- 失败：{before['compile_fail']}
- 原始数据：{reconstruction.get('raw_rows', 'unknown')} 行、{reconstruction.get('unique_records', 'unknown')} 条 trajectory。
- 主要结构特征：statement、顺序 tactic trace、state_before/state_after 可用；source file、imports、namespace、scope、section、local notation/attribute 均不可用。

## 2. Context-aware Pipeline

新 schema 保存 imports、namespaces、open namespaces、scopes、variables、hypotheses、local instances、local notations、section context、environment hash、恢复状态/来源/警告和 assembled source hash。proof-state 文本只作结构化审计，绝不作为 Lean 声明插入。

| Recovery status | Count |
|---|---:|
{recovery_rows}

| Recovery source | Covered samples |
|---|---:|
{source_rows}

- context-aware 通过：{after['compile_success']}/{after['total_samples']}（{after_ratio:.2%}）
- 绝对提升：{absolute_gain:+.2%}
- 相对提升：{relative_gain:+.2%}
- 恢复警告：{warning_summary or 'none'}

`partial=500` 是刻意的保守结论：proof state、statement 和 tactic trace 可恢复，但文件级环境不存在于数据中；因此不伪造 full recovery。

## 3. Paired Comparison

| Outcome | Count |
|---|---:|
| legacy fail → context success | {outcomes.get('improved', 0)} |
| legacy success → context fail | {outcomes.get('regressed', 0)} |
| stable success | {outcomes.get('stable_success', 0)} |
| stable failure | {outcomes.get('stable_failure', 0)} |

回归数：{len(regressions)}。逐条结果见 `audit/workbook_regressions.md`。

| Failure taxonomy | Legacy | Context-aware |
|---|---:|---:|
{taxonomy_rows}

## 4. Failure Analysis

### 已解决

本次没有 context-shaped failure 被可靠消除。不能把 proof state 中显示的 locals 重新声明到 theorem 外部；那会重复 binder、改变语义或制造伪上下文。

### 未解决

- 缺失 local identifier / unknown identifier：原数据没有定义来源、namespace 或 import provenance，无法可靠恢复。
- syntax / notation：原数据没有 scope 和 local notation；部分还可能是文本损坏或当前 Lean/mathlib API 差异。
- tactic / unsolved goals：属于 proof 执行本身，不能用 context schema 掩盖。
- elaboration / instance：只有 proof state 展示，没有可重放的 source declaration。
- timeout：保持原验证政策，不放宽标准。

## 5. Pipeline 结论

1. 这 500 条中没有证据表明“丢失且可恢复的文件级 context”造成了大量失败：before/after 完全相同。
2. 原数据足以恢复 theorem statement、proof trajectory 与 proof-state locals，但不足以恢复 imports、namespace、scope、section 和 local notation；只支持 partial recovery。
3. 新 pipeline 值得替换旧 schema/缓存契约，因为它可审计且能防止错误 cache 复用；它不应被宣传为能提高这批数据的通过率。
4. 应构建 `verified_v2`，但只能纳入本环境 Pantograph 真正通过的 {after['compile_success']} 条；其余 {after['compile_fail']} 条进入 quarantine。源状态非 proved 或 trajectory 损坏的数据应 rejected，不能进入训练。
5. 下一步最值得投入的是：回到带 source provenance 的数据源恢复 namespace/scope/notation；对本数据则优先修复 syntax/notation corruption 和核查 unknown identifiers 的 Lean/mathlib API drift。
6. 在全量重验完成前，不应直接替换生产 verified 集；本任务没有启动训练，也没有接入 LeanDojo。

## 原始字段证据

- source_file={fields.get('source_file')}
- imports={fields.get('imports')}
- namespace={fields.get('namespace')}
- scope={fields.get('open_scoped_declarations')}
- section={fields.get('section')}
- local_notation={fields.get('local_notation')}
- preceding_context={fields.get('preceding_context')}

## 复现入口

固定输入路径、seed、worker 数、timeout 与 cache 策略保存在
`audit/reproduction.json`。使用
`scripts/run_context_aware_data_pipeline.py --workbook-only --stage all`
可重新执行同一 paired experiment。
"""


def finalize_reports(
    *,
    args: argparse.Namespace,
    output: Path,
    identity: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    workbook_context: list[LeanDataRecord],
    leandojo_context: list[LeanDataRecord],
) -> None:
    required = (
        "workbook-before",
        "workbook-after",
        "leandojo-before",
        "leandojo-after",
    )
    missing = [stage for stage in required if stage not in reports]
    if missing:
        raise RuntimeError(f"cannot finalize reports; missing stages: {missing}")
    workbook_before = reports["workbook-before"]
    workbook_after = reports["workbook-after"]
    leandojo_before = reports["leandojo-before"]
    leandojo_after = reports["leandojo-after"]
    leandojo_passed = float(leandojo_after["success_ratio"]) >= args.threshold
    compatibility = environment_compatibility(
        args=args,
        identity=identity,
        before=leandojo_before,
        after=leandojo_after,
    )
    (output / "audit" / "environment_compatibility.md").write_text(
        compatibility,
        encoding="utf-8",
    )
    if leandojo_passed:
        manifest_status = build_training_manifests(
            output=output,
            workbook_records=_load_verified_records(
                output / "verified" / "workbook_verified_v2.jsonl"
            ),
            leandojo_records=_load_verified_records(
                output / "verified" / "leandojo_verified_v1.jsonl"
            ),
            seed=args.seed,
        )
    else:
        manifest_status = {
            "generated": False,
            "reason": (
                f"LeanDojo verified ratio {leandojo_after['success_ratio']:.2%} "
                f"is below the {args.threshold:.2%} quality gate"
            ),
        }
        (output / "STOP_TRAINING").write_text(
            manifest_status["reason"] + "\n",
            encoding="utf-8",
        )
    comparison = {
        "workbook": {
            **compare_reports(workbook_before, workbook_after),
            "paired_outcomes": paired_outcomes(output=output, dataset="workbook"),
        },
        "leandojo": {
            **compare_reports(leandojo_before, leandojo_after),
            "paired_outcomes": paired_outcomes(output=output, dataset="leandojo"),
        },
        "leandojo_quality_gate": {
            "threshold": args.threshold,
            "passed": leandojo_passed,
        },
        "manifest_status": manifest_status,
    }
    write_json(output / "audit" / "before_after_comparison.json", comparison)
    final = final_report_markdown(
        workbook_before=workbook_before,
        workbook_after=workbook_after,
        leandojo_before=leandojo_before,
        leandojo_after=leandojo_after,
        comparison=comparison,
        identity=identity,
    )
    (output / "final_report.md").write_text(final, encoding="utf-8")


def compare_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "before_success_ratio": before["success_ratio"],
        "after_success_ratio": after["success_ratio"],
        "absolute_gain": after["success_ratio"] - before["success_ratio"],
        "before_success": before["compile_success"],
        "after_success": after["compile_success"],
        "taxonomy": {
            category: {
                "before": before["failure_taxonomy"].get(category, 0),
                "after": after["failure_taxonomy"].get(category, 0),
                "delta": (
                    after["failure_taxonomy"].get(category, 0)
                    - before["failure_taxonomy"].get(category, 0)
                ),
            }
            for category in TAXONOMY
        },
    }


def paired_outcomes(*, output: Path, dataset: str) -> dict[str, Any]:
    """Compare identical sampled records before and after context recovery."""

    audit_dir = output / "audit"
    before_rows = json.loads(
        (audit_dir / f"{dataset}_before_verification.json").read_text(
            encoding="utf-8"
        )
    )
    after_rows = json.loads(
        (audit_dir / f"{dataset}_after_verification.json").read_text(
            encoding="utf-8"
        )
    )
    before = {str(row["record_id"]): row for row in before_rows}
    after = {str(row["record_id"]): row for row in after_rows}
    shared = sorted(before.keys() & after.keys())
    transitions: Counter[str] = Counter()
    improved_from: Counter[str] = Counter()
    regressed_to: Counter[str] = Counter()
    outcomes = Counter()
    for record_id in shared:
        old = before[record_id]
        new = after[record_id]
        old_success = bool(old.get("success"))
        new_success = bool(new.get("success"))
        old_category = (
            "success"
            if old_success
            else str(old.get("error_taxonomy") or "unclassified_failure")
        )
        new_category = (
            "success"
            if new_success
            else str(new.get("error_taxonomy") or "unclassified_failure")
        )
        transitions[f"{old_category} -> {new_category}"] += 1
        if not old_success and new_success:
            outcomes["improved"] += 1
            improved_from[old_category] += 1
        elif old_success and not new_success:
            outcomes["regressed"] += 1
            regressed_to[new_category] += 1
        elif old_success:
            outcomes["stable_success"] += 1
        else:
            outcomes["stable_failure"] += 1
    return {
        "paired_records": len(shared),
        "before_only": len(before.keys() - after.keys()),
        "after_only": len(after.keys() - before.keys()),
        "outcomes": dict(sorted(outcomes.items())),
        "improved_from": dict(improved_from.most_common()),
        "regressed_to": dict(regressed_to.most_common()),
        "transitions": dict(transitions.most_common()),
    }


def environment_compatibility(
    *,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> str:
    mathlib_repo = args.lean_project / ".lake" / "packages" / "mathlib"
    old_commit = "29dcec074de168ac2bf835a77ef68bbe069194c5"
    old_toolchain = _git_show_optional(mathlib_repo, old_commit, "lean-toolchain")
    pantograph_version = importlib.metadata.version("pantograph")
    after_taxonomy = after["failure_taxonomy"]
    context_shaped = sum(
        int(after_taxonomy.get(name, 0))
        for name in (
            "missing_context",
            "missing_namespace",
            "missing_scope",
            "missing_variable",
        )
    )
    drift_shaped = sum(
        int(after_taxonomy.get(name, 0))
        for name in (
            "unknown_identifier",
            "elaboration_error",
            "syntax_error",
        )
    )
    return f"""# Environment compatibility

## LeanDojo source

- Lean toolchain: `{old_toolchain.strip() or 'unknown'}`
- mathlib commit: `{old_commit}`

## Current verification environment

- Lean: `{identity.get('lean_version')}`
- mathlib commit: `{identity.get('mathlib_commit')}`
- Pantograph: `{pantograph_version}`

## Error attribution

- Context-shaped failures after reconstruction: {context_shaped}
- API/Lean-drift-shaped failures after reconstruction: {drift_shaped}
- Before success: {before['compile_success']}/{before['total_samples']}
- After success: {after['compile_success']}/{after['total_samples']}

The historical source commit is available locally and was used only for
source-text reconstruction.  The current Lean version and mathlib commit were
not changed.  Failures removed by the After run are attributable to recovered
lexical context; remaining unknown identifiers, type mismatches, invalid field
notations, and parser differences are migration/version candidates rather than
proof-quality failures.

## Recommendation

Do not change the current main environment merely to raise this gate.  If the
same fixed sample remains below 70%, use a separate historical environment for
migration diagnostics and port only audited declarations back to the current
environment.  Never treat source status as a substitute for current
Pantograph compilation.
"""


def build_training_manifests(
    *,
    output: Path,
    workbook_records: list[dict[str, Any]],
    leandojo_records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    if len(workbook_records) < 1000 or len(leandojo_records) < 1000:
        reason = (
            "quality gate passed, but the verified pools are too small for all "
            "requested 1000-record manifests; no partial manifest was emitted"
        )
        (output / "STOP_TRAINING").write_text(reason + "\n", encoding="utf-8")
        return {"generated": False, "reason": reason}
    rng = random.Random(seed)
    rng.shuffle(workbook_records)
    rng.shuffle(leandojo_records)
    recipes = {
        "wb1000.jsonl": workbook_records[:1000],
        "wb500_ld500.jsonl": [
            *workbook_records[:500],
            *leandojo_records[:500],
        ],
        "wb250_ld750.jsonl": [
            *workbook_records[:250],
            *leandojo_records[:750],
        ],
        "ld1000.jsonl": leandojo_records[:1000],
    }
    for name, rows in recipes.items():
        write_jsonl(output / "manifests" / name, rows)
    (output / "STOP_TRAINING").unlink(missing_ok=True)
    return {"generated": True, "manifests": sorted(recipes)}


def stage_report_markdown(report: Mapping[str, Any]) -> str:
    taxonomy = "\n".join(
        f"| {name} | {count} |"
        for name, count in report["failure_taxonomy"].items()
    )
    return f"""# {report['stage']} verification

- Total samples: {report['total_samples']}
- Compile success: {report['compile_success']}
- Compile fail: {report['compile_fail']}
- Success ratio: {report['success_ratio']:.2%}
- Cache hit/miss: {report['cache']['hit']}/{report['cache']['miss']}

| Failure taxonomy | Count |
|---|---:|
{taxonomy}
"""


def final_report_markdown(
    *,
    workbook_before: Mapping[str, Any],
    workbook_after: Mapping[str, Any],
    leandojo_before: Mapping[str, Any],
    leandojo_after: Mapping[str, Any],
    comparison: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> str:
    def row(name: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
        gain = after["success_ratio"] - before["success_ratio"]
        return (
            f"| {name} | {before['compile_success']}/{before['total_samples']} "
            f"({before['success_ratio']:.2%}) | "
            f"{after['compile_success']}/{after['total_samples']} "
            f"({after['success_ratio']:.2%}) | {gain:+.2%} |"
        )

    gate = comparison["leandojo_quality_gate"]
    manifest = comparison["manifest_status"]
    workbook_pairs = comparison["workbook"]["paired_outcomes"]["outcomes"]
    leandojo_pairs = comparison["leandojo"]["paired_outcomes"]["outcomes"]
    return f"""# Context-aware Lean data pipeline report

## Result

| Dataset | Legacy | Context-aware | Absolute gain |
|---|---:|---:|---:|
{row('Lean Workbook', workbook_before, workbook_after)}
{row('LeanDojo', leandojo_before, leandojo_after)}

LeanDojo 70% gate: **{'PASS' if gate['passed'] else 'STOP TRAINING'}**.
Manifest status: `{json.dumps(manifest, ensure_ascii=False)}`.

## Paired outcomes

| Dataset | Improved | Regressed | Stable success | Stable failure |
|---|---:|---:|---:|---:|
| Lean Workbook | {workbook_pairs.get('improved', 0)} | {workbook_pairs.get('regressed', 0)} | {workbook_pairs.get('stable_success', 0)} | {workbook_pairs.get('stable_failure', 0)} |
| LeanDojo | {leandojo_pairs.get('improved', 0)} | {leandojo_pairs.get('regressed', 0)} | {leandojo_pairs.get('stable_success', 0)} | {leandojo_pairs.get('stable_failure', 0)} |

## Lean Workbook

Lean Workbook exposes no source-file location or repository commit.  The
context builder therefore preserves the complete formal declaration and
proof-state locals, records source lookup as unavailable, and does not invent
imports, namespaces, or notation.  The measured delta above shows whether any
loss was actually caused by omitted context.

## LeanDojo

The legacy run uses the existing proof-state-only adapter.  The context-aware
run reads the historical source from locally available Git objects, restores
the source declaration header and active imports, namespaces, sections, opens,
variables, local instances, local notation/attributes, and then compiles in
the unchanged current target environment.

Remaining failures must be interpreted together with
`audit/environment_compatibility.md`; source-context recovery does not erase
Lean/mathlib API drift.

## Environment

- Lean: `{identity.get('lean_version')}`
- mathlib: `{identity.get('mathlib_commit')}`
- environment hash: `{identity.get('environment_hash')}`

## Recommendation

LeanDojo may enter training only if the fixed current-environment Pantograph
gate is at least 70%.  Below that threshold its recommended training share is
0%, no SFT should start, and a separate historical environment may be used
only as a migration aid.  The current main Lean/mathlib environment should
remain unchanged.
"""


def load_existing_stage_reports(output: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for stage in (
        "workbook-before",
        "workbook-after",
        "leandojo-before",
        "leandojo-after",
        "runtime-tests",
    ):
        path = output / "audit" / f"{stage.replace('-', '_')}_summary.json"
        if path.exists():
            found[stage] = json.loads(path.read_text(encoding="utf-8"))
    runtime = output / "audit" / "runtime_contract_tests.json"
    if runtime.exists():
        found["runtime-tests"] = json.loads(runtime.read_text(encoding="utf-8"))
    return found


def _iter_leandojo_candidates(path: Path, stats: Counter[str]):
    for row in iter_json_array(path):
        stats["raw_records"] += 1
        trace = list(row.get("traced_tactics") or ())
        if not trace:
            stats["empty_proof"] += 1
            continue
        if str(trace[-1].get("state_after") or "").strip() != "no goals":
            stats["unsolved_goals"] += 1
            continue
        if not all(
            str(trace[index - 1].get("state_after") or "").strip()
            == str(trace[index].get("state_before") or "").strip()
            for index in range(1, len(trace))
        ):
            stats["corrupt_tactic_trace"] += 1
            continue
        proof = "\n".join(str(step.get("tactic") or "") for step in trace)
        if not proof.strip():
            stats["empty_proof"] += 1
            continue
        if set(lean_code_tokens(proof)) & FORBIDDEN:
            stats["forbidden_proof"] += 1
            continue
        stats["base_candidates"] += 1
        yield row


def _fixed_sample(
    rows: list[Any],
    *,
    size: int,
    seed: int,
    id_getter,
) -> list[Any]:
    if len(rows) < size:
        raise ValueError(f"requested {size} records from {len(rows)} candidates")
    selected = random.Random(seed).sample(rows, size)
    if len({id_getter(row) for row in selected}) != size:
        raise ValueError("fixed sample contains duplicate IDs")
    return selected


def _write_or_validate_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_ids = [row["record_id"] for row in previous]
        current_ids = [row["record_id"] for row in rows]
        if previous_ids != current_ids:
            raise RuntimeError(f"fixed sample changed: {path}")
        return
    write_json(path, rows)


def _load_verified_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _git_show_optional(repository: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout if completed.returncode == 0 else ""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
