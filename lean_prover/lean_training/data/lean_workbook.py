"""Deterministic reconstruction of Lean-Workbook tactic trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contracts import DataState, LeanDataRecord, SourceSpan
from .preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    statement_hash,
    strip_dataset_placeholder_proof,
)


LEAN_WORKBOOK_DATASET = "InternLM/Lean-Workbook"
CONTEXT_RECOVERY_VERSION = "lean_workbook_trajectory_v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def declaration_name(statement: str) -> str | None:
    match = re.match(r"\s*(?:theorem|lemma|example)\s+([^\s(:{]+)", statement)
    return match.group(1) if match else None


def render_tactic_proof(tactics: Iterable[str]) -> str:
    lines = ["by"]
    for tactic in tactics:
        for line in tactic.strip().splitlines():
            lines.append(f"  {line}" if line.strip() else "")
    return "\n".join(lines).rstrip()


def reconstruct_lean_workbook_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_file: str | Path | None,
    source_commit: str | None,
) -> tuple[list[LeanDataRecord], dict[str, Any]]:
    """Group per-step rows by ID and reconstruct complete proof trajectories."""

    materialized = [dict(row) for row in rows]
    groups: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    for index, row in enumerate(materialized):
        row_id = str(row.get("id") or f"row-{index}")
        if not groups or groups[-1][0] != row_id:
            groups.append((row_id, []))
        groups[-1][1].append((index, row))

    records: list[LeanDataRecord] = []
    transition_mismatches = 0
    formal_statement_variants = 0
    status_conflicts = 0
    for row_id, trajectory in groups:
        statuses = {str(row.get("status") or "unknown") for _, row in trajectory}
        statements = {str(row.get("formal_statement") or "") for _, row in trajectory}
        if len(statuses) != 1:
            status_conflicts += 1
        if len(statements) != 1:
            formal_statement_variants += 1
        transition_ok = all(
            str(trajectory[index - 1][1].get("state_after") or "").strip()
            == str(trajectory[index][1].get("state_before") or "").strip()
            for index in range(1, len(trajectory))
        )
        if not transition_ok:
            transition_mismatches += 1
        first_index, first = trajectory[0]
        last_index, last = trajectory[-1]
        raw_declaration = str(first.get("formal_statement") or "")
        statement = strip_dataset_placeholder_proof(raw_declaration)
        status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
        complete = str(last.get("state_after") or "").strip() == "no goals"
        proof = render_tactic_proof(str(row.get("tactic") or "") for _, row in trajectory)
        raw_context = json.dumps(
            [
                {
                    "source_row": index,
                    "tactic": row.get("tactic"),
                    "state_before": row.get("state_before"),
                    "state_after": row.get("state_after"),
                }
                for index, row in trajectory
            ],
            ensure_ascii=False,
        )
        quarantine_reason = None
        if status != "proved":
            quarantine_reason = f"source_status_{status}"
        elif not transition_ok or not complete:
            quarantine_reason = "source_extraction_corruption"
        elif len(statements) != 1 or not statement:
            quarantine_reason = "source_extraction_corruption"
        data_state = (
            DataState.QUARANTINED if quarantine_reason else DataState.RAW
        )
        records.append(
            LeanDataRecord(
                record_id=row_id,
                statement_id=f"stmt_{statement_hash(statement)[:24]}",
                data_state=data_state,
                source_dataset=LEAN_WORKBOOK_DATASET,
                source_file=str(Path(source_file).resolve()) if source_file else None,
                source_module=None,
                source_declaration=declaration_name(statement) or row_id,
                source_commit=source_commit,
                source_span=SourceSpan(start_row=first_index, end_row=last_index),
                imports=["Mathlib"],
                informal_statement=str(first.get("natural_language_statement") or ""),
                statement=statement,
                proof=proof if status == "proved" else None,
                proof_format=ProofFormat.FULL_PROOF,
                raw_declaration=raw_declaration,
                raw_source_context=raw_context,
                verification_status=(
                    "raw" if quarantine_reason is None else "quarantined"
                ),
                verification_error_type=quarantine_reason,
                verification_error_message=(
                    f"Lean-Workbook trajectory status={status}, "
                    f"transition_ok={transition_ok}, complete={complete}"
                    if quarantine_reason
                    else None
                ),
                assembler_version=ASSEMBLER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
                context_recovery_version=CONTEXT_RECOVERY_VERSION,
                recovered_source_hash=sha256_text(raw_context),
                metadata={
                    "raw_status": status,
                    "trajectory_steps": len(trajectory),
                    "initial_state": first.get("state_before"),
                    "final_state": last.get("state_after"),
                    "imports_provenance": "fixed_target_environment",
                    "source_answer": first.get("answer"),
                },
            )
        )

    report = {
        "raw_rows": len(materialized),
        "unique_records": len(records),
        "row_schema": sorted({key for row in materialized for key in row}),
        "status_rows": dict(Counter(str(row.get("status")) for row in materialized)),
        "status_records": dict(
            Counter(str(record.metadata.get("raw_status")) for record in records)
        ),
        "trajectory_steps": dict(
            Counter(int(record.metadata["trajectory_steps"]) for record in records)
        ),
        "transition_mismatches": transition_mismatches,
        "formal_statement_variant_groups": formal_statement_variants,
        "status_conflicts": status_conflicts,
        "complete_trajectories": sum(
            record.metadata.get("final_state") == "no goals" for record in records
        ),
        "available_source_fields": {
            "source_file": False,
            "source_module": False,
            "declaration_name": True,
            "imports": False,
            "namespace": False,
            "section": False,
            "variable_declarations": "state_before_only",
            "local_hypotheses": "state_before_only",
            "open_declarations": False,
            "open_scoped_declarations": False,
            "local_notation": False,
            "local_attributes": False,
            "full_declaration": True,
            "theorem_statement": True,
            "proof": "ordered_tactic_trajectory",
            "source_span": "parquet_row_span",
            "source_commit": "huggingface_snapshot_only",
            "preceding_context": "proof_state_only",
        },
    }
    return records, report
