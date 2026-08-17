"""Role-specific adapters for verified NuminaMath SFT and GRPO records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import (
    has_supported_declaration_prefix,
    record_id,
    require_verified_scope,
    split_context_and_target,
    split_imports_and_body,
    strip_final_assignment,
)
from ..preparation import (
    NormalizedExample,
    contains_forbidden_proof_token,
    split_lean_statement_and_proof,
)


_REFERENCE_PROOF_FIELDS = {
    "proof",
    "verified_proof",
    "completion",
    "reference_proof",
    "reference_proof_hash",
    "reference_proof_length_tokens",
    "reference_proof_length_characters",
    "reference_proof_length_lines",
    "has_reference_proof",
}


def normalize_numinamath_sft_record(
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "AI-MO/NuminaMath-LEAN",
) -> NormalizedExample:
    """Adapt a full-proof Pantograph success into an SFT source example."""

    require_verified_scope(record, scope="full_proof")
    proof = str(record.get("proof") or "").strip()
    if not proof:
        raise ValueError("NuminaMath SFT record has no proof")
    if contains_forbidden_proof_token(proof):
        raise ValueError("NuminaMath SFT record contains a forbidden proof token")
    imports, body = split_imports_and_body(str(record.get("lean_statement") or ""))
    context_lines, target = split_context_and_target(body)
    statement = strip_final_assignment(target)
    if not statement:
        raise ValueError("NuminaMath SFT record lacks lean_statement")
    return NormalizedExample(
        id=record_id(record, prefix="numinamath-sft", index=index),
        source="numinamath-sft",
        source_name=source_name,
        informal_statement=str(record.get("problem") or "").strip(),
        lean_statement=statement,
        proof=proof,
        imports=imports or ("Mathlib",),
        context_lines=context_lines,
        pantograph_verified=True,
    )


def normalize_numinamath_grpo_record(
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "AI-MO/NuminaMath-LEAN",
) -> NormalizedExample:
    """Adapt a statement-only success without inspecting proof values."""

    require_verified_scope(record, scope="statement_only")
    forbidden_fields = _REFERENCE_PROOF_FIELDS.intersection(record)
    if forbidden_fields:
        raise ValueError(
            "NuminaMath GRPO record contains proof-bearing fields: "
            + ", ".join(sorted(forbidden_fields))
        )
    imports, body = split_imports_and_body(
        str(record.get("lean_statement") or "")
    )
    context_lines, target = split_context_and_target(body)
    # This verified-data field is already the proof-free target emitted by the
    # statement-only Pantograph pipeline.  Re-parsing ``:=`` here is unsafe:
    # proposition-level ``let`` binders are part of the theorem type.
    statement = target.strip()
    try:
        _, tactic_proof = split_lean_statement_and_proof(statement)
    except ValueError as error:
        if "ambiguous Lean declaration assignment" not in str(error):
            raise
        tactic_proof = ""
    if (
        tactic_proof == "by"
        or tactic_proof.startswith("by ")
        or tactic_proof.startswith("by\n")
    ):
        raise ValueError("NuminaMath GRPO main theorem must not contain a proof")
    if not statement or not has_supported_declaration_prefix(statement):
        raise ValueError("NuminaMath GRPO record lacks lean_statement")
    metadata = record.get("metadata")
    informal = str(metadata.get("original_problem") or "").strip() if isinstance(metadata, Mapping) else ""
    return NormalizedExample(
        id=record_id(record, prefix="numinamath-grpo", index=index),
        source="numinamath-grpo",
        source_name=source_name,
        informal_statement=informal,
        lean_statement=statement,
        imports=imports or ("Mathlib",),
        context_lines=context_lines,
        pantograph_verified=True,
    )


__all__ = [
    "normalize_numinamath_grpo_record",
    "normalize_numinamath_sft_record",
]
