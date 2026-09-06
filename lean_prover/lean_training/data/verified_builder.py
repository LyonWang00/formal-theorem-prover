"""End-to-end audited Lean-Workbook reconstruction and verified split builder."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .audit import (
    classify_verification_failure,
    normalized_proof_body,
    proof_distribution,
    tactic_signature,
)
from .contracts import DataState, LeanDataRecord, make_attestation_id, utc_now
from .adapters.lean_workbook import reconstruct_lean_workbook_records
from .preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    compose_lean_theorem,
)
from .training import build_generation_prompt
from lean_prover.lean_training.runtime.environment import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _updated(record: LeanDataRecord, **changes: Any) -> LeanDataRecord:
    payload = record.model_dump(mode="json")
    payload.update(changes)
    return LeanDataRecord.model_validate(payload)


def _verification_task(
    record: LeanDataRecord,
    *,
    problem_index: int,
    environment_hash: str,
) -> VerificationTask:
    if not record.proof:
        raise ValueError("proof audit requires a reconstructed proof")
    declaration = compose_lean_theorem(
        record.statement,
        record.proof,
        proof_format=record.proof_format,
    )
    generation_id = f"raw-audit:{record.record_id}"
    task = VerificationTask(
        priority=problem_index,
        problem_index=problem_index,
        attempt_index=0,
        problem_id=record.record_id,
        prompt="",
        generated_proof=record.proof,
        raw_completion=record.proof,
        lean_code=declaration,
        imports=tuple(record.imports or ("Mathlib",)),
        context_lines=(),
        payload={
            "generation_id": generation_id,
            "record_id": record.record_id,
            "environment_hash": environment_hash,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        },
        reject_forbidden=True,
    )
    assembled = build_labeled_lean_code(task, include_imports=True)
    task.payload["assembled_source"] = assembled
    task.payload["assembled_source_hash"] = sha256_text(assembled)
    return task


def _deduplicate_statements(
    records: list[LeanDataRecord],
) -> tuple[list[LeanDataRecord], list[LeanDataRecord]]:
    kept: list[LeanDataRecord] = []
    duplicates: list[LeanDataRecord] = []
    seen: dict[str, str] = {}
    for record in records:
        if record.data_state is DataState.QUARANTINED:
            continue
        previous = seen.get(record.statement_id)
        if previous is None:
            seen[record.statement_id] = record.record_id
            kept.append(record)
            continue
        duplicates.append(
            _updated(
                record,
                data_state=DataState.QUARANTINED,
                proof=None,
                verification_status="quarantined",
                verification_error_type="duplicate_statement",
                verification_error_message=(
                    f"normalized statement duplicates record {previous}"
                ),
            )
        )
    return kept, duplicates


def _attest(
    record: LeanDataRecord,
    result: dict[str, Any],
    identity: dict[str, Any],
) -> LeanDataRecord:
    assembled_hash = str(result["assembled_source_hash"])
    attestation_id = make_attestation_id(
        record_id=record.record_id,
        environment_hash=str(identity["environment_hash"]),
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash=assembled_hash,
    )
    return _updated(
        record,
        data_state=DataState.VERIFIED,
        statement_verified=True,
        proof_verified=True,
        pantograph_verified=True,
        reference_proof_verified=True,
        verification_status="verified",
        verification_error_type=None,
        verification_error_message=None,
        lean_version=identity["lean_version"],
        mathlib_commit=identity["mathlib_commit"],
        environment_hash=identity["environment_hash"],
        attestation_id=attestation_id,
        attested_at=utc_now(),
        assembled_source_hash=assembled_hash,
    )


def _quarantine_failure(
    record: LeanDataRecord,
    result: dict[str, Any],
    identity: dict[str, Any],
) -> LeanDataRecord:
    error_type = classify_verification_failure(result)
    return _updated(
        record,
        data_state=DataState.QUARANTINED,
        proof=None,
        statement_verified=False,
        proof_verified=False,
        pantograph_verified=False,
        verification_status=str(result.get("status") or "failed"),
        verification_error_type=error_type,
        verification_error_message=str(result.get("diagnostics") or ""),
        lean_version=identity["lean_version"],
        mathlib_commit=identity["mathlib_commit"],
        environment_hash=identity["environment_hash"],
        assembled_source_hash=result.get("assembled_source_hash"),
        metadata={
            **record.metadata,
            "compile_errors": result.get("compile_errors") or [],
            "timed_out": bool(result.get("timed_out")),
        },
    )


def _prepared_row(record: LeanDataRecord, *, role: str) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    prompt = build_generation_prompt(
        {
            "informal_statement": record.informal_statement,
            "lean_statement": record.statement,
        }
    )
    payload.update(
        {
            "id": record.record_id,
            "data_role": role,
            "lean_statement": record.statement,
            "informal_statement": record.informal_statement,
            "prompt": prompt,
            "statement_hash": record.statement_id.removeprefix("stmt_"),
            "has_reference_proof": bool(record.proof),
        }
    )
    if role in {"train", "eval"}:
        payload["completion"] = record.proof
        payload["proof"] = record.proof
        payload["reference_proof"] = None
    else:
        payload["reference_proof"] = record.proof
        payload["proof"] = None
        payload["completion"] = None
    return payload


def _filter_sft_length(
    records: list[LeanDataRecord],
    *,
    tokenizer_name_or_path: str | None,
    max_seq_length: int | None,
) -> tuple[list[LeanDataRecord], dict[str, Any]]:
    """Keep split candidates whose prompt and complete proof fit without truncation."""

    if tokenizer_name_or_path is None and max_seq_length is None:
        return records, {"enabled": False, "candidate_records": len(records)}
    if not tokenizer_name_or_path or max_seq_length is None:
        raise ValueError(
            "tokenizer_name_or_path and max_seq_length must be configured together"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    eligible: list[LeanDataRecord] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        row = _prepared_row(record, role="train")
        completion = str(row["completion"] or "")
        if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        token_count = len(
            tokenizer(
                str(row["prompt"]) + completion,
                add_special_tokens=False,
            )["input_ids"]
        )
        if token_count <= max_seq_length:
            eligible.append(record)
        else:
            excluded.append(
                {
                    "record_id": record.record_id,
                    "statement_id": record.statement_id,
                    "token_count": token_count,
                }
            )
    return eligible, {
        "enabled": True,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "max_seq_length": max_seq_length,
        "candidate_records": len(records),
        "eligible_records": len(eligible),
        "excluded_overlength_records": len(excluded),
        "excluded": excluded,
    }


def _select_diverse(
    records: list[LeanDataRecord], count: int, *, seed: int
) -> list[LeanDataRecord]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected: list[LeanDataRecord] = []
    seen_proofs: set[str] = set()
    for record in shuffled:
        proof_key = normalized_proof_body(record.proof or "")
        if proof_key in seen_proofs:
            continue
        selected.append(record)
        seen_proofs.add(proof_key)
        if len(selected) == count:
            return selected
    for record in shuffled:
        if record not in selected:
            selected.append(record)
        if len(selected) == count:
            break
    return selected


def _select_balanced_train(
    records: list[LeanDataRecord],
    count: int,
    *,
    seed: int,
    max_identical_proof_occurrences: int,
    max_single_tactic_fraction: float,
) -> tuple[list[LeanDataRecord], dict[str, Any]]:
    """Select a reproducible train set without discarding the verified reserve."""

    rng = random.Random(seed)
    proof_counts: Counter[str] = Counter()
    rejected_by_proof_cap = 0

    def balanced_take(candidates: list[LeanDataRecord], required: int) -> list[LeanDataRecord]:
        nonlocal rejected_by_proof_cap
        buckets: dict[tuple[str, ...], list[LeanDataRecord]] = defaultdict(list)
        for record in candidates:
            buckets[tactic_signature(record.proof or "")].append(record)
        signatures = list(buckets)
        rng.shuffle(signatures)
        for rows in buckets.values():
            rng.shuffle(rows)
        selected: list[LeanDataRecord] = []
        while len(selected) < required:
            made_progress = False
            for signature in signatures:
                rows = buckets[signature]
                while rows:
                    record = rows.pop()
                    proof_key = normalized_proof_body(record.proof or "")
                    if proof_counts[proof_key] >= max_identical_proof_occurrences:
                        rejected_by_proof_cap += 1
                        continue
                    proof_counts[proof_key] += 1
                    selected.append(record)
                    made_progress = True
                    break
                if len(selected) == required:
                    break
            if not made_progress:
                break
        return selected

    single = [
        record for record in records if len(tactic_signature(record.proof or "")) <= 1
    ]
    multi = [
        record for record in records if len(tactic_signature(record.proof or "")) > 1
    ]
    single_limit = int(count * max_single_tactic_fraction)
    multi_required = count - single_limit
    selected_multi = balanced_take(multi, multi_required)
    selected_single = balanced_take(single, count - len(selected_multi))
    selected = selected_multi + selected_single
    rng.shuffle(selected)
    if len(selected) != count or len(selected_single) > single_limit:
        raise RuntimeError(
            "verified reserve cannot satisfy the configured proof-distribution "
            f"policy: selected={len(selected)}/{count}, "
            f"single={len(selected_single)}/{single_limit}"
        )
    return selected, {
        "seed": seed,
        "max_identical_proof_occurrences": max_identical_proof_occurrences,
        "max_single_tactic_fraction": max_single_tactic_fraction,
        "balance_by_tactic_signature": True,
        "candidate_records": len(records),
        "candidate_single_tactic_records": len(single),
        "candidate_multi_step_records": len(multi),
        "selected_records": len(selected),
        "selected_single_tactic_records": len(selected_single),
        "selected_multi_step_records": len(selected_multi),
        "rejected_by_proof_cap_during_selection": rejected_by_proof_cap,
    }


def _trace_current_splits(
    current_files: dict[str, Path],
    reconstructed: dict[str, LeanDataRecord],
    verified_ids: set[str],
    failures: dict[str, LeanDataRecord],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, path in current_files.items():
        rows = read_jsonl(path)
        reasons: Counter[str] = Counter()
        for row in rows:
            record = reconstructed.get(str(row.get("id")))
            if record is None:
                reasons["source_record_missing"] += 1
                continue
            raw_status = str(record.metadata.get("raw_status"))
            if raw_status != "proved":
                reasons[f"source_status_{raw_status}"] += 1
                continue
            if split != "discovery":
                old_proof = str(row.get("proof") or row.get("completion") or "")
                if normalized_proof_body(old_proof) != normalized_proof_body(
                    record.proof or ""
                ):
                    reasons["source_extraction_corruption"] += 1
                    continue
            if record.record_id in verified_ids:
                reasons["verified"] += 1
            else:
                failure = failures.get(record.record_id)
                reasons[
                    failure.verification_error_type if failure else "unknown"
                ] += 1
        report[split] = {
            "total": len(rows),
            "classification": dict(reasons),
            "verified_rate": reasons["verified"] / max(1, len(rows)),
        }
    return report


def _write_markdown_reports(
    output_dir: Path,
    raw_report: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    raw_lines = [
        "# Lean Workbook raw schema audit",
        "",
        f"- Raw rows: {raw_report['raw_rows']}",
        f"- Unique declaration IDs: {raw_report['unique_records']}",
        f"- Proved declarations: {raw_report['status_records'].get('proved', 0)}",
        f"- Disproved declarations: {raw_report['status_records'].get('disproved', 0)}",
        f"- Trajectory transition mismatches: {raw_report['transition_mismatches']}",
        "",
        "## Context availability",
        "",
        "The source contains per-step `state_before/state_after`, but no source module, "
        "imports, namespace, section, open-scope, notation, attribute, file span, or "
        "repository commit fields. The cached Hugging Face snapshot is the only source "
        "version identifier.",
        "",
        "## Confirmed legacy preprocessing loss",
        "",
        "The parquet stores one tactic step per row. The legacy path shuffled rows before "
        "deduplicating by ID, so a random first/middle/final tactic step was treated as a "
        "complete theorem proof. It also did not exclude `status=disproved` trajectories.",
    ]
    (output_dir / "raw_schema_report.md").write_text(
        "\n".join(raw_lines) + "\n", encoding="utf-8"
    )
    summary_lines = [
        "# Lean Workbook verified-v2 audit",
        "",
        f"- Reconstructed records: {summary['total_records']}",
        f"- Pantograph verified proofs: {summary['proof_verified_records']}",
        f"- Quarantined records: {summary['quarantined_records']}",
        f"- Proof verification rate: {summary['proof_verification_rate']:.4%}",
        "",
        "## Failure taxonomy",
        "",
        *[
            f"- `{name}`: {count}"
            for name, count in sorted(summary["error_taxonomy"].items())
        ],
    ]
    (output_dir / "audit_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def build_verified_dataset(
    *,
    raw_parquet: Path,
    source_commit: str,
    lean_project: Path,
    output_dir: Path,
    current_files: dict[str, Path],
    num_workers: int,
    timeout: int,
    startup_timeout: int,
    seed: int,
    train_size: int,
    eval_size: int,
    discovery_size: int,
    tokenizer_name_or_path: str | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    """Warm Pantograph before loading the full raw corpus into memory."""

    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=str(lean_project),
            imports=("Mathlib",),
            timeout=timeout,
            warmup_timeout=startup_timeout,
            num_workers=num_workers,
            queue_maxsize=8,
            max_worker_restarts=3,
            max_task_retries=1,
            shutdown_timeout=15,
        )
    )
    with pool:
        return _build_verified_dataset_with_pool(
            pool=pool,
            raw_parquet=raw_parquet,
            source_commit=source_commit,
            lean_project=lean_project,
            output_dir=output_dir,
            current_files=current_files,
            timeout=timeout,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            discovery_size=discovery_size,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_seq_length=max_seq_length,
        )


def _build_verified_dataset_with_pool(
    *,
    pool: VerificationPool | None,
    raw_parquet: Path,
    source_commit: str,
    lean_project: Path,
    output_dir: Path,
    current_files: dict[str, Path],
    timeout: int,
    seed: int,
    train_size: int,
    eval_size: int,
    discovery_size: int,
    tokenizer_name_or_path: str | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = pq.read_table(raw_parquet).to_pylist()
    reconstructed, raw_report = reconstruct_lean_workbook_records(
        raw_rows,
        source_file=raw_parquet,
        source_commit=source_commit,
    )
    raw_report.update(
        {
            "source_file": str(raw_parquet.resolve()),
            "source_commit": source_commit,
            "old_preparation_loss": {
                "operation": "row-level shuffle followed by first-ID deduplication",
                "module": "data/preparation.py::sample_dataset -> normalize_records",
                "effect": "one random tactic step retained instead of the full trajectory",
                "disproved_filter_missing": True,
            },
        }
    )
    (audit_dir / "raw_schema_report.json").write_text(
        json.dumps(raw_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    initially_quarantined = [
        record for record in reconstructed if record.data_state is DataState.QUARANTINED
    ]
    candidates, duplicates = _deduplicate_statements(reconstructed)
    initially_quarantined.extend(duplicates)
    identity = environment_identity(lean_project, ("Mathlib",))
    results_path = audit_dir / "verification_results.jsonl"
    existing = {
        str(row.get("record_id")): row
        for row in read_jsonl(results_path)
        if row.get("environment_hash") == identity["environment_hash"]
        and row.get("assembler_version") == ASSEMBLER_VERSION
        and row.get("normalization_version") == NORMALIZATION_VERSION
    }
    tasks: list[VerificationTask] = []
    for index, record in enumerate(candidates):
        if record.record_id in existing:
            continue
        try:
            tasks.append(
                _verification_task(
                    record,
                    problem_index=index,
                    environment_hash=str(identity["environment_hash"]),
                )
            )
        except Exception as error:
            local = {
                "record_id": record.record_id,
                "generation_id": f"raw-audit:{record.record_id}",
                "success": False,
                "status": "assembly_error",
                "diagnostics": f"{type(error).__name__}: {error}",
                "compile_errors": [str(error)],
                "environment_hash": identity["environment_hash"],
                "assembler_version": ASSEMBLER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
            }
            append_jsonl(results_path, local)
            existing[record.record_id] = local

    completed = len(existing)

    def persist(result: dict[str, Any]) -> None:
        nonlocal completed
        append_jsonl(results_path, result)
        existing[str(result["record_id"])] = result
        completed += 1
        if completed % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "raw_proof_audit",
                        "completed": completed,
                        "expected": len(candidates),
                    }
                ),
                flush=True,
            )

    if tasks and pool is None:
        raise RuntimeError(
            "cached-report rebuild requested, but compatible verification results "
            f"are missing for {len(tasks)} records"
        )
    run = pool.run_batch(tasks, on_result=persist) if tasks and pool else None
    if run is not None and run.fatal_errors:
        raise RuntimeError(f"Pantograph pool failed: {run.fatal_errors}")

    verified: list[LeanDataRecord] = []
    failed: list[LeanDataRecord] = []
    for record in candidates:
        result = existing.get(record.record_id)
        if result is None:
            failed.append(
                _updated(
                    record,
                    data_state=DataState.QUARANTINED,
                    proof=None,
                    verification_status="missing_result",
                    verification_error_type="environment_error",
                    verification_error_message="verification result is missing",
                )
            )
        elif result.get("success"):
            verified.append(_attest(record, result, identity))
        else:
            failed.append(_quarantine_failure(record, result, identity))
    quarantined = initially_quarantined + failed
    error_taxonomy = Counter(
        record.verification_error_type or "unknown" for record in quarantined
    )
    current_audit = _trace_current_splits(
        current_files,
        {record.record_id: record for record in reconstructed},
        {record.record_id for record in verified},
        {record.record_id: record for record in failed},
    )
    distribution = proof_distribution(verified)
    summary = {
        "target_environment": identity,
        "total_records": len(reconstructed),
        "proof_audit_candidates": len(candidates),
        "proof_verified_records": len(verified),
        "quarantined_records": len(quarantined),
        "proof_verification_rate": len(verified) / max(1, len(candidates)),
        "error_taxonomy": dict(error_taxonomy),
        "current_processed_splits": current_audit,
        "source_status_counts": raw_report["status_records"],
        "migration_rules": [],
    }
    write_jsonl(
        audit_dir / "verified_records.jsonl",
        (record.model_dump(mode="json") for record in verified),
    )
    write_jsonl(
        output_dir / "quarantined.jsonl",
        (record.model_dump(mode="json") for record in quarantined),
    )
    write_jsonl(
        audit_dir / "quarantined_records.jsonl",
        (record.model_dump(mode="json") for record in quarantined),
    )
    (audit_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "error_taxonomy.json").write_text(
        json.dumps(dict(error_taxonomy), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "proof_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    error_examples: dict[str, list[LeanDataRecord]] = defaultdict(list)
    for record in quarantined:
        key = record.verification_error_type or "unknown"
        if len(error_examples[key]) < 5:
            error_examples[key].append(record)
    for error_type, examples in error_examples.items():
        (audit_dir / "error_examples").mkdir(exist_ok=True)
        (audit_dir / "error_examples" / f"{error_type}.json").write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in examples],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_markdown_reports(audit_dir, raw_report, summary)

    split_candidates, length_policy = _filter_sft_length(
        verified,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_seq_length=max_seq_length,
    )
    (audit_dir / "sft_length_selection.json").write_text(
        json.dumps(length_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train, distribution_policy = _select_balanced_train(
        split_candidates,
        min(train_size, len(split_candidates)),
        seed=seed,
        max_identical_proof_occurrences=50,
        max_single_tactic_fraction=0.40,
    )
    selected_train_ids = {record.record_id for record in train}
    shuffled = [
        record
        for record in split_candidates
        if record.record_id not in selected_train_ids
    ]
    random.Random(seed + 1).shuffle(shuffled)
    requested = train_size + eval_size + discovery_size
    evaluation = shuffled[:eval_size]
    discovery_start = len(evaluation)
    discovery = shuffled[discovery_start : discovery_start + discovery_size]
    write_jsonl(output_dir / "train.jsonl", (_prepared_row(row, role="train") for row in train))
    write_jsonl(output_dir / "eval.jsonl", (_prepared_row(row, role="eval") for row in evaluation))
    write_jsonl(
        output_dir / "discovery.jsonl",
        (_prepared_row(row, role="discovery") for row in discovery),
    )
    attestations = [
        {
            "record_id": row.record_id,
            "statement_id": row.statement_id,
            "attestation_id": row.attestation_id,
            "attested_at": row.attested_at,
            "environment_hash": row.environment_hash,
            "assembled_source_hash": row.assembled_source_hash,
            "assembler_version": row.assembler_version,
            "normalization_version": row.normalization_version,
            "statement_verified": row.statement_verified,
            "proof_verified": row.proof_verified,
        }
        for row in verified
    ]
    write_jsonl(output_dir / "attestations.jsonl", attestations)

    extra_pool = shuffled[discovery_start + len(discovery) :]
    train_distribution = proof_distribution(train)
    (audit_dir / "train_proof_distribution.json").write_text(
        json.dumps(train_distribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "proof_distribution_selection.json").write_text(
        json.dumps(distribution_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    golden_groups = {
        "train": _select_diverse(train, min(10, len(train)), seed=seed + 1),
        "eval": _select_diverse(evaluation, min(10, len(evaluation)), seed=seed + 2),
        "verified_extra": _select_diverse(extra_pool, min(10, len(extra_pool)), seed=seed + 3),
    }
    golden = [
        {
            **_prepared_row(record, role=group),
            "group": group,
            "proof": record.proof,
            "completion": record.proof,
        }
        for group, records in golden_groups.items()
        for record in records
    ]
    write_jsonl(output_dir / "golden_reference_roundtrip.jsonl", golden)

    manifest_path = output_dir / "manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest = {
        "dataset_version": "lean_workbook_verified_v2",
        "source_dataset": "InternLM/Lean-Workbook",
        "source_file": str(raw_parquet.resolve()),
        "source_commit": source_commit,
        "environment": identity,
        "assembler_version": ASSEMBLER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "requested_sizes": {
            "train": train_size,
            "eval": eval_size,
            "discovery": discovery_size,
        },
        "actual_sizes": {
            "train": len(train),
            "eval": len(evaluation),
            "discovery": len(discovery),
            "verified_reserve": max(
                0,
                len(split_candidates) - len(train) - len(evaluation) - len(discovery),
            ),
            "quarantined": len(quarantined),
        },
        "target_total_reached": len(split_candidates) >= requested,
        "role_statement_overlap": 0,
        "proof_distribution_report": "audit/proof_distribution.json",
        "train_proof_distribution_report": "audit/train_proof_distribution.json",
        "proof_distribution_selection": distribution_policy,
        "sft_length_selection": length_policy,
        "sft_length_selection_report": "audit/sft_length_selection.json",
        "audit_summary": "audit/audit_summary.json",
        "golden_reference_count": len(golden),
    }
    previous_sizes = previous_manifest.get("actual_sizes", {})
    for role in ("monitor", "benchmark"):
        if role in previous_sizes:
            manifest["actual_sizes"][role] = previous_sizes[role]
    for key in (
        "statement_verification_report",
        "statement_verification_results",
        "statement_only_roles",
        "statement_migration_rules",
    ):
        if key in previous_manifest:
            manifest[key] = previous_manifest[key]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest
