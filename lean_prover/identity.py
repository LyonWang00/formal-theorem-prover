"""Stable identifiers shared by Planner, Prover, and result collection."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


def normalize_identity_text(value: str | None) -> str:
    """Normalize harmless text differences before identity hashing."""

    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_unique_strings(values: Iterable[str]) -> list[str]:
    """Return stable, de-duplicated normalized strings in sorted order."""

    return sorted(
        {
            normalized
            for value in values
            if (normalized := normalize_identity_text(value))
        }
    )


def stable_sha256(namespace: str, payload: Mapping[str, Any]) -> str:
    """Hash a canonical JSON payload with an explicit identity namespace."""

    canonical = json.dumps(
        {"namespace": namespace, **dict(payload)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_input_hash(
    *,
    input_text: str,
    imports: Iterable[str],
    available_definitions: Iterable[str],
    formal_statement: str | None = None,
    informal_statement: str | None = None,
    header: str = "",
) -> str:
    return stable_sha256(
        "theorem-input-v1",
        {
            "input_text": normalize_identity_text(input_text),
            "imports": normalized_unique_strings(imports),
            "available_definitions": normalized_unique_strings(
                available_definitions
            ),
            "formal_statement": normalize_identity_text(formal_statement),
            "informal_statement": normalize_identity_text(informal_statement),
            "header": normalize_identity_text(header),
        },
    )


def compute_problem_hash(
    *,
    input_hash: str = "",
    natural_language_statement: str | None,
    target_lean_decl: str,
    imports: Iterable[str],
    available_definitions: Iterable[str],
    header: str = "",
) -> str:
    if input_hash:
        return stable_sha256(
            "routed-theorem-problem-v1",
            {"input_hash": input_hash},
        )
    return stable_sha256(
        "formal-theorem-problem-v1",
        {
            "natural_language_statement": normalize_identity_text(
                natural_language_statement
            ),
            "target_lean_decl": normalize_identity_text(target_lean_decl),
            "imports": normalized_unique_strings(imports),
            "available_definitions": normalized_unique_strings(
                available_definitions
            ),
            "header": normalize_identity_text(header),
        },
    )
