#!/usr/bin/env python3
"""Run the mandatory fixed 500-record LeanDojo Pantograph quality gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.audit import classify_verification_failure
from lean_prover.lean_training.data.contracts import DataState, LeanDataRecord
from lean_prover.lean_training.data.leandojo import (
    corpus_key,
    iter_json_array,
    leandojo_prompt_statement,
    load_corpus_declarations,
    reconstruct_leandojo_records,
    reservoir_sample,
)
from lean_prover.lean_training.data.preparation import compose_lean_theorem
from lean_prover.lean_training.data.preparation import lean_code_tokens
from lean_prover.lean_training.data.training import build_generation_prompt
from lean_prover.lean_training.data.verified_builder import (
    _attest,
    _quarantine_failure,
    write_jsonl,
)
from lean_prover.lean_training.expert_iteration.utils import (
    environment_identity,
    write_json_atomic,
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
    "missing_context",
    "missing_import",
    "environment_error",
    "elaboration_error",
    "tactic_error",
    "unsolved_goals",
    "timeout",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--lean-project", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    audit_dir = output / "audit"
    processed_dir = output / "processed"
    audit_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    scan_stats: Counter[str] = Counter()
    sample, candidate_total = reservoir_sample(
        _iter_base_candidates(args.split, scan_stats),
        size=args.sample_size,
        seed=args.seed,
    )
    keys = {
        corpus_key(
            str(row.get("file_path") or ""),
            str(row.get("full_name") or ""),
        )
        for row in sample
    }
    corpus = load_corpus_declarations(args.corpus, keys)
    records, cleaning = reconstruct_leandojo_records(
        sample,
        corpus_declarations=corpus,
        split_name=args.split.stem,
    )
    sample_ids = [
        {
            "sample_index": index,
            "record_id": record.record_id,
            "source_index": record.metadata.get("source_index"),
            "source_file": record.source_file,
            "source_declaration": record.source_declaration,
        }
        for index, record in enumerate(records)
    ]
    write_json_atomic(audit_dir / "leandojo_quality_gate_sample_ids.json", sample_ids)
    clean_records = [
        record for record in records if record.data_state is not DataState.QUARANTINED
    ]
    write_jsonl(
        processed_dir / "leandojo_sample_clean.jsonl",
        (record.model_dump(mode="json") for record in clean_records),
    )

    identity = environment_identity(args.lean_project, ("Mathlib",))
    tasks: list[VerificationTask] = []
    records_by_id = {record.record_id: record for record in clean_records}
    for index, record in enumerate(clean_records):
        declaration = compose_lean_theorem(
            record.statement,
            record.proof or "",
            proof_format=record.proof_format,
        )
        if record.namespace:
            declaration = (
                f"namespace {record.namespace}\n\n"
                f"{declaration}\n\n"
                f"end {record.namespace}"
            )
        generation_id = f"leandojo-gate:{record.record_id}"
        prompt = build_generation_prompt(
            {
                "informal_statement": "",
                "lean_statement": leandojo_prompt_statement(record),
            }
        )
        task = VerificationTask(
            priority=index,
            problem_index=index,
            attempt_index=0,
            problem_id=record.record_id,
            prompt=prompt,
            generated_proof=record.proof or "",
            raw_completion=record.proof or "",
            lean_code=declaration,
            imports=("Mathlib",),
            context_lines=tuple(
                line
                for line in (record.variable_context or "").splitlines()
                if line.strip()
            ),
            payload={
                "generation_id": generation_id,
                "record_id": record.record_id,
                "environment_hash": identity["environment_hash"],
                "assembler_version": record.assembler_version,
                "normalization_version": record.normalization_version,
            },
            reject_forbidden=True,
        )
        assembled = build_labeled_lean_code(task, include_imports=True)
        task.payload["assembled_source"] = assembled
        task.payload["assembled_source_hash"] = _sha256(assembled)
        tasks.append(task)

    pool_config = VerificationPoolConfig(
        lean_project_path=str(args.lean_project.resolve()),
        imports=("Mathlib",),
        timeout=args.timeout,
        warmup_timeout=max(120, args.timeout),
        num_workers=args.workers,
        queue_maxsize=max(32, args.workers * 16),
        max_task_retries=1,
        max_worker_restarts=3,
        task_spool_dir=str(output / "runtime" / "quality_gate_tasks"),
        save_full_source_on_failure_only=True,
    )
    with VerificationPool(pool_config) as pool:
        run = pool.run_batch(tasks)
    if run.fatal_errors:
        raise RuntimeError("Pantograph quality gate failed: " + "; ".join(run.fatal_errors))

    result_by_id = {str(result["record_id"]): result for result in run.results}
    verified: list[LeanDataRecord] = []
    failed: list[LeanDataRecord] = []
    taxonomy: Counter[str] = Counter()
    detailed_failures: Counter[str] = Counter()
    for record in clean_records:
        result = result_by_id.get(record.record_id)
        if result is None:
            synthetic = {
                "status": "missing_result",
                "diagnostics": "Pantograph pool returned no result",
                "compile_errors": ["missing verification result"],
                "assembled_source_hash": None,
            }
            failed.append(_quarantine_failure(record, synthetic, identity))
            taxonomy["environment_error"] += 1
            detailed_failures["missing_result"] += 1
        elif result.get("success"):
            verified.append(_attest(record, result, identity))
        else:
            failure = classify_verification_failure(result)
            detailed_failures[failure] += 1
            taxonomy[_quality_gate_category(failure)] += 1
            failed.append(_quarantine_failure(record, result, identity))

    prefilter_taxonomy: Counter[str] = Counter()
    for record in records:
        if record.data_state is not DataState.QUARANTINED:
            continue
        category = _quality_gate_category(record.verification_error_type or "unknown")
        prefilter_taxonomy[category] += 1
        taxonomy[category] += 1
        detailed_failures[record.verification_error_type or "unknown"] += 1

    write_jsonl(
        processed_dir / "leandojo_sample_verified.jsonl",
        (record.model_dump(mode="json") for record in verified),
    )
    write_jsonl(
        processed_dir / "leandojo_sample_quarantined.jsonl",
        (
            record.model_dump(mode="json")
            for record in [*failed, *(
                row for row in records if row.data_state is DataState.QUARANTINED
            )]
        ),
    )
    write_jsonl(audit_dir / "leandojo_quality_gate_verification.jsonl", run.results)

    syntax_invalid = sum(
        record.verification_error_type == "invalid_syntax"
        for record in records
    )
    statement_invalid_reasons = {
        "invalid_syntax",
        "missing_context",
        "private_declaration",
        "duplicate_statement",
    }
    statement_invalid = sum(
        record.verification_error_type in statement_invalid_reasons
        for record in records
    )
    success_rate = len(verified) / args.sample_size
    passed = success_rate >= args.threshold
    report = {
        "gate_name": "LeanDojo Benchmark 4 fixed 500-record quality gate",
        "source_split_records": scan_stats["raw_records"],
        "base_candidate_records": candidate_total,
        "base_prefilter": dict(scan_stats),
        "seed": args.seed,
        "threshold": args.threshold,
        "passed": passed,
        "training_allowed": passed,
        "metrics": {
            "total_sampled": args.sample_size,
            "syntax_valid": args.sample_size - syntax_invalid,
            "statement_valid": args.sample_size - statement_invalid,
            "proof_valid": len(clean_records),
            "pantograph_success": len(verified),
            "success_rate": success_rate,
        },
        "cleaning": cleaning,
        "corpus_declarations_requested": len(keys),
        "corpus_declarations_found": len(corpus),
        "failure_taxonomy": {
            category: taxonomy.get(category, 0) for category in TAXONOMY
        },
        "prefilter_failure_taxonomy": dict(prefilter_taxonomy),
        "detailed_failure_reasons": dict(detailed_failures),
        "environment": identity,
        "runtime": {
            **run.runtime_stats,
            "warmup_reports": run.warmup_reports,
            "recovered_worker_failures": run.recovered_worker_failures,
        },
        "decision": (
            "PASS: verified ratio reached 70%; downstream verified-pool and "
            "training phases are permitted."
            if passed
            else "STOP TRAINING: verified ratio is below 70%; only failure "
            "analysis is permitted."
        ),
    }
    write_json_atomic(audit_dir / "leandojo_quality_gate.json", report)
    markdown = f"""# LeanDojo 500-record quality gate

- Seed: {args.seed}
- Total sampled: {args.sample_size}
- Syntax valid: {report['metrics']['syntax_valid']}
- Statement valid: {report['metrics']['statement_valid']}
- Proof valid: {report['metrics']['proof_valid']}
- Pantograph success: {len(verified)}
- Verified ratio: {success_rate:.2%}
- Required threshold: {args.threshold:.2%}
- Decision: **{'PASS' if passed else 'STOP TRAINING'}**

## Failure taxonomy

```json
{json.dumps(report['failure_taxonomy'], ensure_ascii=False, indent=2)}
```

## Detailed failure reasons

```json
{json.dumps(report['detailed_failure_reasons'], ensure_ascii=False, indent=2)}
```
"""
    (audit_dir / "leandojo_quality_gate.md").write_text(markdown, encoding="utf-8")
    if not passed:
        (output / "STOP_TRAINING").write_text(
            report["decision"] + "\n", encoding="utf-8"
        )
    else:
        (output / "STOP_TRAINING").unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _quality_gate_category(value: str) -> str:
    if value in {"invalid_syntax", "reference_syntax_error", "corrupt_tactic_trace"}:
        return "syntax_error"
    if value in {
        "missing_context",
        "missing_hypothesis",
        "missing_local_variable",
        "missing_namespace",
        "missing_local_notation",
        "private_declaration",
        "duplicate_statement",
    }:
        return "missing_context"
    if value in {"missing_import"}:
        return "missing_import"
    if value in {
        "environment_error",
        "version_incompatible_identifier",
        "missing_result",
    }:
        return "environment_error"
    if value in {"reference_elaboration_error", "unknown"}:
        return "elaboration_error"
    if value in {
        "reference_tactic_failure",
        "empty_proof",
        "forbidden_declaration",
    }:
        return "tactic_error"
    if value in {"reference_unsolved_goals", "unsolved_goals"}:
        return "unsolved_goals"
    if value == "timeout":
        return "timeout"
    return "elaboration_error"


def _iter_base_candidates(
    split: Path, scan_stats: Counter[str]
):
    """Apply only source-intrinsic proof filters before the portability sample."""

    forbidden = {"sorry", "admit", "sorryAx", "axiom", "unsafe"}
    for row in iter_json_array(split):
        scan_stats["raw_records"] += 1
        trace = list(row.get("traced_tactics") or ())
        if not trace:
            scan_stats["empty_proof"] += 1
            continue
        if str(trace[-1].get("state_after") or "").strip() != "no goals":
            scan_stats["unsolved_goals"] += 1
            continue
        if not all(
            str(trace[index - 1].get("state_after") or "").strip()
            == str(trace[index].get("state_before") or "").strip()
            for index in range(1, len(trace))
        ):
            scan_stats["corrupt_tactic_trace"] += 1
            continue
        proof = "\n".join(str(step.get("tactic") or "") for step in trace)
        if not proof.strip():
            scan_stats["empty_proof"] += 1
            continue
        if set(lean_code_tokens(proof)) & forbidden:
            scan_stats["forbidden_proof"] += 1
            continue
        scan_stats["base_candidates"] += 1
        yield row


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
