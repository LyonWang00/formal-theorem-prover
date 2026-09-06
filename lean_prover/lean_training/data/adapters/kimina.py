"""Statement-only GRPO adapter for verified Kimina prompt records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import (
    has_supported_declaration_prefix,
    record_id,
    require_verified_scope,
    split_imports_and_body,
)
from ..preparation import NormalizedExample, split_lean_statement_and_proof


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


def normalize_kimina_grpo_record(
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "AI-MO/Kimina-Prover-Promptset",
) -> NormalizedExample:
    """Adapt one verified Kimina theorem prompt and discard placeholder fields."""

    require_verified_scope(record, scope="statement_only")
    forbidden_fields = _REFERENCE_PROOF_FIELDS.intersection(record)
    if forbidden_fields:
        raise ValueError(
            "Kimina GRPO record contains proof-bearing fields: "
            + ", ".join(sorted(forbidden_fields))
        )
    imports, _ = split_imports_and_body(str(record.get("formal_statement") or ""))
    if record.get("proof_present") not in {None, False}:
        raise ValueError("Kimina GRPO record declares a main-theorem proof")
    statement = str(record.get("lean_statement") or "").strip()
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
        raise ValueError("Kimina GRPO main theorem must not contain a proof")
    if not statement or not has_supported_declaration_prefix(statement):
        raise ValueError("Kimina GRPO record lacks lean_statement")
    return NormalizedExample(
        id=record_id(record, prefix="kimina-grpo", index=index),
        source="kimina-grpo",
        source_name=source_name,
        informal_statement=str(record.get("natural_language") or "").strip(),
        lean_statement=statement,
        imports=imports or ("Mathlib",),
        pantograph_verified=True,
    )


__all__ = ["normalize_kimina_grpo_record"]
