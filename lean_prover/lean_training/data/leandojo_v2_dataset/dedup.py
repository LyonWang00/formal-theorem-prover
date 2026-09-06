"""Auditable within-source and cross-source theorem/proof deduplication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("record_id") or row.get("statement_id"))


def _canonical_score(row: dict[str, Any]) -> tuple[int, ...]:
    verification = row.get("verification") or {}
    metadata = row.get("metadata") or {}
    context = metadata.get("context_recovery") or {}
    return (
        int(row.get("verification_status") == "verified_default_timeout"),
        int(bool(row.get("source_file") and row.get("source_span"))),
        int(bool(row.get("repository_commit"))),
        int(context.get("recovery_status") == "full"),
        int(bool(row.get("raw_tactic_trace"))),
        int(not bool(verification.get("stderr"))),
    )


def _group(
    records: Iterable[dict[str, Any]], key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[tuple(row.get(field) for field in key_fields)].append(row)
    return groups


def _duplicate_groups(
    records: list[dict[str, Any]], level: int, fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    result = []
    for key, rows in _group(records, fields).items():
        if len(rows) < 2:
            continue
        result.append(
            {
                "level": level,
                "fields": list(fields),
                "key": list(key),
                "record_ids": sorted(_id(row) for row in rows),
                "record_count": len(rows),
            }
        )
    return result


def audit_internal_dedup(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deduplicate true statement/proof copies while preserving proof variants."""

    levels = {
        0: ("id",),
        1: ("source_identity_hash",),
        2: ("repository_commit", "qualified_name"),
        3: ("statement_hash_exact", "proof_hash_exact"),
        4: ("statement_hash_normalized", "proof_hash_normalized"),
    }
    duplicate_groups = [
        group
        for level, fields in levels.items()
        for group in _duplicate_groups(records, level, fields)
    ]

    by_normalized_pair = _group(
        records, ("statement_hash_normalized", "proof_hash_normalized")
    )
    canonical: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for rows in by_normalized_pair.values():
        ordered = sorted(rows, key=lambda row: (_canonical_score(row), _id(row)), reverse=True)
        winner = ordered[0]
        canonical.append(winner)
        if len(ordered) > 1:
            aliases.append(
                {
                    "canonical_id": _id(winner),
                    "alias_ids": sorted(_id(row) for row in ordered[1:]),
                    "sources": sorted({str(row.get("source")) for row in ordered}),
                    "dedup_reason": "normalized_statement_and_proof",
                }
            )

    variants: list[dict[str, Any]] = []
    for key, rows in _group(canonical, ("statement_hash_normalized",)).items():
        proof_hashes = {row.get("proof_hash_normalized") for row in rows}
        if len(proof_hashes) < 2:
            continue
        variants.append(
            {
                "theorem_group_id": rows[0]["theorem_group_id"],
                "statement_hash_normalized": key[0],
                "record_ids": sorted(_id(row) for row in rows),
                "proof_hashes": sorted(str(value) for value in proof_hashes),
                "proof_variant_count": len(proof_hashes),
            }
        )

    same_proof_different_statement: list[dict[str, Any]] = []
    for key, rows in _group(canonical, ("proof_hash_normalized",)).items():
        statements = {row.get("statement_hash_normalized") for row in rows}
        if len(statements) < 2:
            continue
        same_proof_different_statement.append(
            {
                "proof_hash_normalized": key[0],
                "record_ids": sorted(_id(row) for row in rows),
                "statement_count": len(statements),
                "action": "report_only_no_merge",
            }
        )

    canonical.sort(key=_id)
    aliases.sort(key=lambda row: row["canonical_id"])
    return {
        "canonical_records": canonical,
        "dedup_aliases": aliases,
        "same_statement_multiple_proofs": variants,
        "same_proof_different_statements": same_proof_different_statement,
        "dedup_groups": duplicate_groups,
        "statistics": {
            f"level_{level}_groups": sum(
                group["level"] == level for group in duplicate_groups
            )
            for level in levels
        }
        | {
            f"level_{level}_duplicate_rows": sum(
                group["record_count"] - 1
                for group in duplicate_groups
                if group["level"] == level
            )
            for level in levels
        }
        | {
            "input_records": len(records),
            "canonical_records": len(canonical),
            "deduplicated_records": len(records) - len(canonical),
            "same_statement_multiple_proof_groups": len(variants),
            "same_proof_different_statement_groups": len(
                same_proof_different_statement
            ),
        },
    }


def audit_cross_source(
    leandojo: list[dict[str, Any]], workbook: list[dict[str, Any]]
) -> dict[str, Any]:
    """Audit exact and theorem-level overlap without collapsing proof variants."""

    wb_by_pair = _group(
        workbook, ("statement_hash_normalized", "proof_hash_normalized")
    )
    wb_by_statement = _group(workbook, ("statement_hash_normalized",))
    wb_by_proof = _group(workbook, ("proof_hash_normalized",))
    overlaps: list[dict[str, Any]] = []
    workbook_alias_ids: set[str] = set()
    for ld in leandojo:
        statement_key = (ld["statement_hash_normalized"],)
        exact = wb_by_pair.get(
            (ld["statement_hash_normalized"], ld["proof_hash_normalized"]), []
        )
        statement_matches = wb_by_statement.get(statement_key, [])
        proof_matches = [
            row
            for row in wb_by_proof.get((ld["proof_hash_normalized"],), [])
            if row["statement_hash_normalized"] != ld["statement_hash_normalized"]
        ]
        if not exact and not statement_matches and not proof_matches:
            continue
        exact_ids = sorted(_id(row) for row in exact)
        different_proof_ids = sorted(
            _id(row) for row in statement_matches if row not in exact
        )
        workbook_alias_ids.update(exact_ids)
        overlaps.append(
            {
                "canonical_id": _id(ld),
                "theorem_group_id": ld["theorem_group_id"],
                "exact_statement_exact_proof_alias_ids": exact_ids,
                "same_statement_different_proof_ids": different_proof_ids,
                "same_proof_different_statement_ids": sorted(
                    _id(row) for row in proof_matches
                ),
                "cross_source_exact_duplicate": bool(exact_ids),
                "cross_source_theorem_overlap": bool(statement_matches),
            }
        )

    return {
        "overlaps": overlaps,
        "workbook_exact_alias_ids": workbook_alias_ids,
        "statistics": {
            "cross_source_exact_duplicate_records": sum(
                row["cross_source_exact_duplicate"] for row in overlaps
            ),
            "cross_source_theorem_overlap_records": sum(
                row["cross_source_theorem_overlap"] for row in overlaps
            ),
            "workbook_records_aliased_to_leandojo": len(workbook_alias_ids),
            "cross_source_same_proof_different_statement_records": sum(
                bool(row["same_proof_different_statement_ids"])
                for row in overlaps
            ),
        },
    }
