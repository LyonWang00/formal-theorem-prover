"""Four-layer normalization for LeanDojo-v2 whole-proof records."""

from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .fingerprints import (
    canonical_premises,
    exact_hash,
    normalized_hash,
    premise_set_hash,
    source_identity_hash,
    statement_hash,
    theorem_group_id,
)

NORMALIZATION_VERSION = "leandojo-v2-four-layer-v1"
_TRAILING_ASSIGN = re.compile(r"\s*:=\s*$")


def normalize_display_text(value: str) -> str:
    """Normalize line endings/outer whitespace without rewriting Lean syntax."""

    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def proof_format(proof: str, declared_style: str | None = None) -> str:
    stripped = proof.lstrip()
    style = (declared_style or "").lower()
    if stripped == "by" or stripped.startswith("by\n") or stripped.startswith("by "):
        return "tactic_by" if style != "mixed" else "mixed"
    if style == "mixed":
        return "mixed"
    return "term"


def _training_statement(statement: str) -> str:
    return _TRAILING_ASSIGN.sub("", normalize_display_text(statement))


def _training_proof(proof: str) -> str:
    return normalize_display_text(proof)


def _normalized_trace(raw_trace: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for position, item in enumerate(raw_trace):
        if not isinstance(item, Mapping):
            normalized.append(copy.deepcopy(item))
            continue
        row = copy.deepcopy(dict(item))
        row["normalized_step_index"] = position
        for key in ("state_before", "tactic", "state_after"):
            if isinstance(row.get(key), str):
                row[f"normalized_{key}"] = normalize_display_text(row[key])
        normalized.append(row)
    return normalized


def _trace_validation(raw_trace: list[Any]) -> dict[str, Any]:
    object_steps = [item for item in raw_trace if isinstance(item, Mapping)]
    indices = [item.get("step_index") for item in object_steps]
    continuous = not indices or indices == list(range(len(indices)))
    aligned = True
    for previous, current in zip(object_steps, object_steps[1:]):
        after = previous.get("state_after")
        before = current.get("state_before")
        if isinstance(after, str) and isinstance(before, str):
            aligned = aligned and normalize_display_text(after) == normalize_display_text(
                before
            )
    last_state = str(object_steps[-1].get("state_after") or "") if object_steps else ""
    final_goal_closed = not object_steps or normalize_display_text(last_state).lower() in {
        "no goals",
        "",
    }
    return {
        "step_indices_continuous": continuous,
        "adjacent_states_aligned": aligned,
        "final_goal_closed": final_goal_closed,
        "step_count": len(object_steps),
    }


def normalize_leandojo_record(
    candidate: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a normalized record while retaining all source-faithful fields."""

    raw = copy.deepcopy(dict(candidate))
    statement = str(candidate.get("statement") or "")
    proof = str(candidate.get("proof") or "")
    declaration = str(
        candidate.get("declaration_source")
        or candidate.get("raw_declaration")
        or ""
    )
    if not statement or not proof or not declaration:
        raise ValueError("statement, proof, and declaration source are required")

    training_statement = _training_statement(statement)
    training_proof = _training_proof(proof)
    if not training_proof:
        raise ValueError("training proof is empty")
    training_declaration = f"{training_statement} := {training_proof}"
    raw_premises = copy.deepcopy(list(candidate.get("premises") or []))
    raw_trace = copy.deepcopy(list(candidate.get("tactic_trace") or []))
    compile_success = bool(verification.get("compile_success"))
    timed_out = bool(verification.get("timed_out"))
    status = (
        "verified_default_timeout"
        if compile_success and not timed_out
        else "timeout_quarantine"
        if timed_out
        else "compile_failure_quarantine"
    )

    result: dict[str, Any] = {
        **raw,
        "raw_statement": statement,
        "raw_proof": proof,
        "raw_declaration_source": declaration,
        "raw_tactic_trace": raw_trace,
        "raw_premises": raw_premises,
        "raw_file_dependencies": copy.deepcopy(
            list(candidate.get("file_dependencies") or [])
        ),
        "training_statement": training_statement,
        "training_proof": training_proof,
        "training_declaration": training_declaration,
        "proof_style": proof_format(
            training_proof, str(candidate.get("proof_style") or "")
        ),
        "statement_hash_exact": exact_hash(statement),
        "statement_hash_normalized": statement_hash(statement),
        "proof_hash_exact": exact_hash(proof),
        "proof_hash_normalized": normalized_hash(proof),
        "declaration_hash_exact": exact_hash(declaration),
        "source_identity_hash": source_identity_hash(candidate),
        "premise_set_hash": premise_set_hash(raw_premises),
        "premise_set_canonical": canonical_premises(raw_premises),
        "theorem_group_id": theorem_group_id(statement),
        "verification_status": status,
        "verification_class": str(
            verification.get("error_category")
            or ("success" if compile_success else "unknown")
        ),
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_tactic_trace": _normalized_trace(raw_trace),
        "tactic_trace_validation": _trace_validation(raw_trace),
        "verification": copy.deepcopy(dict(verification)),
        "metadata": {
            **copy.deepcopy(dict(candidate.get("metadata") or {})),
            "normalization_layers": {
                "raw": "source-faithful fields prefixed raw_",
                "executable": "metadata.assembled_source and assembled_source_hash",
                "training": "training_statement/training_proof/training_declaration",
                "fingerprint": "exact and conservative normalized hashes",
            },
        },
    }
    return result


def normalize_workbook_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compatible minimum fingerprint/schema for Lean Workbook."""

    raw = copy.deepcopy(dict(record))
    statement = str(record.get("statement") or record.get("lean_statement") or "")
    proof = str(
        record.get("proof")
        or record.get("reference_proof")
        or record.get("completion")
        or ""
    )
    if not statement:
        raise ValueError("Lean Workbook statement is empty")
    return {
        **raw,
        "source": "lean_workbook",
        "raw_statement": statement,
        "raw_proof": proof,
        "training_statement": _training_statement(statement),
        "training_proof": _training_proof(proof),
        "statement_hash_exact": exact_hash(statement),
        "statement_hash_normalized": statement_hash(statement),
        "proof_hash_exact": exact_hash(proof),
        "proof_hash_normalized": normalized_hash(proof),
        "source_identity_hash": source_identity_hash(record),
        "theorem_group_id": theorem_group_id(statement),
        "proof_origin": record.get("source_dataset") or "lean_workbook",
        "normalization_version": NORMALIZATION_VERSION,
    }
