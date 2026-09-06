"""Evaluation-protection registry and conservative leakage audit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .fingerprints import (
    lexical_tokens,
    normalize_statement_lexical,
    structural_fingerprint,
)
from .loader import iter_jsonl
from .normalization import normalize_workbook_record

PROTECTED_ROLE_TERMS = (
    "eval",
    "monitor",
    "benchmark",
    "miniF2F",
    "strict_unseen",
    "unseen",
    "discovery_gate",
    "anchor_gate",
    "full500",
    "reserve",
)


def _id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("record_id") or row.get("statement_id"))


def load_protected_jsonl(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        normalized: list[dict[str, Any]] = []
        failures = 0
        for raw in iter_jsonl(path):
            try:
                normalized.append(normalize_workbook_record(raw))
            except ValueError:
                failures += 1
        records.extend(normalized)
        inventory.append(
            {
                "path": str(path),
                "records": len(normalized),
                "normalization_failures": failures,
                "role": next(
                    (
                        term
                        for term in PROTECTED_ROLE_TERMS
                        if term.lower() in str(path).lower()
                    ),
                    "evaluation_protected",
                ),
            }
        )
    return records, inventory


def audit_leakage(
    training_candidates: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    *,
    high_similarity_threshold: float = 0.9,
) -> dict[str, Any]:
    """Remove Levels 0-2; retain Levels 3-4 only for manual review."""

    protected_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protected_token_sets: list[tuple[dict[str, Any], set[str]]] = []
    for row in protected:
        protected_identity[row["source_identity_hash"]].append(row)
        protected_exact[row["statement_hash_exact"]].append(row)
        protected_normalized[row["statement_hash_normalized"]].append(row)
        protected_structural[structural_fingerprint(row["raw_statement"])].append(row)
        protected_token_sets.append(
            (
                row,
                set(lexical_tokens(normalize_statement_lexical(row["raw_statement"]))),
            )
        )

    exact_identity: list[dict[str, Any]] = []
    exact_statement: list[dict[str, Any]] = []
    normalized_statement: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []
    high_similarity: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    removed_groups: set[str] = set()

    preliminary: dict[str, set[int]] = defaultdict(set)
    for row in training_candidates:
        row_id = _id(row)
        matches = protected_identity.get(row["source_identity_hash"], [])
        if matches:
            exact_identity.append(_leak_record(row, matches, 0))
            preliminary[row["theorem_group_id"]].add(0)
        matches = protected_exact.get(row["statement_hash_exact"], [])
        if matches:
            exact_statement.append(_leak_record(row, matches, 1))
            preliminary[row["theorem_group_id"]].add(1)
        matches = protected_normalized.get(row["statement_hash_normalized"], [])
        if matches:
            normalized_statement.append(_leak_record(row, matches, 2))
            preliminary[row["theorem_group_id"]].add(2)
        structure = structural_fingerprint(row["raw_statement"])
        matches = protected_structural.get(structure, [])
        if matches and not preliminary[row["theorem_group_id"]]:
            structural.append(_leak_record(row, matches, 3))
        if not preliminary[row["theorem_group_id"]]:
            tokens = set(
                lexical_tokens(normalize_statement_lexical(row["raw_statement"]))
            )
            for candidate, candidate_tokens in protected_token_sets:
                maximum = max(len(tokens), len(candidate_tokens))
                minimum = min(len(tokens), len(candidate_tokens))
                if maximum and minimum / maximum < high_similarity_threshold:
                    continue
                union = tokens | candidate_tokens
                score = len(tokens & candidate_tokens) / len(union) if union else 1.0
                if score >= high_similarity_threshold:
                    high_similarity.append(
                        {
                            "training_id": row_id,
                            "protected_id": _id(candidate),
                            "token_jaccard": score,
                            "level": 4,
                            "action": "review_only",
                        }
                    )

    removed_groups = {
        theorem_group for theorem_group, levels in preliminary.items() if levels
    }
    for row in training_candidates:
        if row["theorem_group_id"] in removed_groups:
            removed.append(
                {
                    **row,
                    "leakage_levels": sorted(preliminary[row["theorem_group_id"]]),
                    "removal_reason": "evaluation_protected_theorem_group_level_0_2",
                }
            )
        else:
            retained.append(row)

    return {
        "retained": retained,
        "removed": removed,
        "exact_identity": exact_identity,
        "exact_statement": exact_statement,
        "normalized_statement": normalized_statement,
        "structural_near_duplicates": structural,
        "high_similarity_review": high_similarity,
        "statistics": {
            "level_0_records": len(exact_identity),
            "level_1_records": len(exact_statement),
            "level_2_records": len(normalized_statement),
            "level_3_candidates": len(structural),
            "level_4_candidates": len(high_similarity),
            "removed_theorem_groups": len(removed_groups),
            "removed_records": len(removed),
            "retained_records": len(retained),
        },
    }


def _leak_record(
    training: dict[str, Any], matches: list[dict[str, Any]], level: int
) -> dict[str, Any]:
    return {
        "training_id": _id(training),
        "theorem_group_id": training["theorem_group_id"],
        "protected_ids": sorted(_id(row) for row in matches),
        "protected_sources": sorted(
            {
                str(row.get("data_role") or row.get("source_dataset") or row.get("source"))
                for row in matches
            }
        ),
        "level": level,
    }
