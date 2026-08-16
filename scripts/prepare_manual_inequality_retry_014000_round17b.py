#!/usr/bin/env python3
"""Extract the stronger embedded proof for one valuable inequality repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prepare_manual_structural_repairs_014000_round17 import (
    ROOT,
    RAW,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    sha,
)


RECORD_ID = "112c1916-d57a-5743-b903-d2f30cd6e9f0::le_to_lt_or_eq_faa47c585600"
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/manual_inequality_retry_014000_round17b"
BATCH_ID = "numinamath_expand_repair_manual_inequality_retry_014000_round17b_b00000"
BATCH = RUN / "batches" / BATCH_ID


def main() -> None:
    row = next(x for x in map(json.loads, RAW.open(encoding="utf-8")) if x["record_id"] == RECORD_ID)
    base_path = next(
        p
        for p in (ROOT / "outputs/numinamath_expand_verification/batches").glob("*/candidate_manifest.jsonl")
        if any(json.loads(line)["record_id"] == RECORD_ID for line in p.open(encoding="utf-8"))
    )
    base = next(x for x in map(json.loads, base_path.open(encoding="utf-8")) if x["record_id"] == RECORD_ID)

    original = row["proof"]
    embedded = original.index("\n    theorem inequalities_89689 ")
    exact = original.index("\n  exact lt_or_eq_of_le h_parent")
    second_by = original.index(" := by\n", embedded) + len(" := by\n")
    parent_by = original.index(" := by\n") + len(" := by\n")
    proof = original[:parent_by] + original[second_by:exact] + original[exact:]
    if "theorem inequalities_89689" in proof:
        raise SystemExit("embedded declaration was not fully removed")

    candidate = dict(base)
    candidate["batch_id"] = BATCH_ID
    candidate["repair_version"] = "numinamath_codex_manual_structural_repair_v1"
    candidate["statement_changed"] = False
    candidate["changes"] = ["extract_stronger_embedded_parent_proof"]
    candidate["campaign_metadata"] = {
        "reviewer": "codex-root-manual-inequality-retry-round17b",
        "local_pantograph_reverification_required": True,
        "statement_changed": False,
        "semantic_change": False,
    }
    candidate["variants"] = [{"strategy": "extract_stronger_embedded_parent_proof", "proof": proof}]
    identity = {
        "schema_version": candidate["schema_version"],
        "record_id": RECORD_ID,
        "imports": candidate["imports"],
        "source_body": candidate["source_body"],
        "proof": proof,
        "original_row_sha256": candidate["original_row_sha256"],
    }
    candidate["candidate_hash"] = canonical_hash(identity)
    review = {
        "record_id": RECORD_ID,
        "strategy": "extract_stronger_embedded_parent_proof",
        "reviewed_statement_sha256": sha(row["lean_statement"]),
        "reviewed_proof_sha256": sha(proof),
        "replacement": {"proof": proof},
    }
    atomic_jsonl(BATCH / "candidate_manifest.jsonl", [candidate])
    atomic_jsonl(RUN / "replacement_review_manifest.jsonl", [review])
    report = {
        "schema_version": "numinamath_expand_manual_structural_repair_v1",
        "batch_id": BATCH_ID,
        "records": 1,
        "semantic_changes": 0,
        "external_api_calls": 0,
        "candidate_manifest_sha256": hashlib.sha256((BATCH / "candidate_manifest.jsonl").read_bytes()).hexdigest(),
    }
    atomic_json(BATCH / "prepare_report.json", report)
    atomic_json(RUN / "prepare_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
