"""Context reconstruction for synthetic Lean-Workbook trajectories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .context import (
    LeanSourceContext,
    apply_context_to_record,
    split_proof_state_locals,
)
from .contracts import LeanDataRecord


LEAN_WORKBOOK_CONTEXT_RECOVERY_VERSION = "lean-workbook-context-v3"


def build_lean_workbook_context(
    record: LeanDataRecord,
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    environment_hash: str,
) -> LeanDataRecord:
    """Build a self-contained context record without inventing source provenance.

    Lean-Workbook is synthetic and its public schema has no source path,
    declaration location, import list, namespace stack, or repository commit.
    Its formal declaration already materializes theorem binders.  The proof
    state is therefore retained as structured context, while source lookup is
    explicitly recorded as unavailable rather than silently fabricated.
    """

    rows = [dict(row) for row in raw_rows]
    initial_state = str(rows[0].get("state_before") or "") if rows else ""
    raw_context = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    variables, hypotheses, instances = split_proof_state_locals(initial_state)
    recovery_sources = tuple(
        source
        for source, available in (
            ("proof_state", bool(initial_state and "⊢" in initial_state)),
            ("statement", bool(record.statement)),
            ("tactic_trace", bool(rows)),
        )
        if available
    )
    warnings = (
        "source_file_unavailable",
        "imports_not_recovered_using_fixed_target_import_Mathlib",
        "namespace_unavailable",
        "open_namespace_unavailable",
        "scope_unavailable",
        "section_context_unavailable",
        "local_notation_unavailable",
        "proof_state_locals_are_audit_only_and_not_replayed",
    )
    context = LeanSourceContext(
        imports=tuple(record.imports or ["Mathlib"]),
        variables=variables,
        hypotheses=hypotheses,
        local_instances=instances,
        source_file=None,
        source_commit=record.source_commit,
        source_line=(
            record.source_span.start_row
            if record.source_span is not None
            else None
        ),
        context_recovered=False,
        recovery_method="proof_state_statement_and_tactic_trace_only",
        recovery_status="partial",
        recovery_sources=recovery_sources,
        context_warnings=warnings,
        recovery_errors=(
            "source_lookup_unavailable: Lean-Workbook exposes no source file "
            "or declaration location",
        ),
        source_hash=hashlib.sha256(raw_context.encode("utf-8")).hexdigest(),
    )
    updated = apply_context_to_record(
        record,
        context,
        environment_hash=environment_hash,
    )
    return updated.model_copy(
        update={
            "context_recovery_version": LEAN_WORKBOOK_CONTEXT_RECOVERY_VERSION,
            "metadata": {
                **updated.metadata,
                "source_lookup_status": "not_available_in_dataset",
                "source_context_required": "unknown",
                "proof_state_materialized_as_explicit_binders": True,
                "proof_state_replayed_as_lean_syntax": False,
            },
        }
    )
