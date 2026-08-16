#!/usr/bin/env python3
"""Retry the monotonicity proof after explicitly unfolding function application."""

from __future__ import annotations

import hashlib
import json

from prepare_manual_structural_repairs_014000_round17 import ROOT, atomic_json, atomic_jsonl, canonical_hash, sha


RECORD_ID = "09cadbaf-b3ab-58bb-89aa-ee36d0804c87::and_left_a5fd445b0e2f"
SOURCE = ROOT / "outputs/numinamath_expand_verification/repairs/manual_projection_015000_round21"
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/manual_monotonicity_retry_015000_round21b"
BATCH_ID = "numinamath_expand_repair_manual_monotonicity_retry_015000_round21b_b00000"
BATCH = RUN / "batches" / BATCH_ID


def main() -> None:
    old = next(
        x
        for x in map(
            json.loads,
            (SOURCE / "batches/numinamath_expand_repair_manual_projection_015000_round21_b00000/candidate_manifest.jsonl").open(encoding="utf-8"),
        )
        if x["record_id"] == RECORD_ID
    )
    proof = old["variants"][0]["proof"].replace(
        "  rw [ident, ident]\n",
        "  change f x + g x - |f x - g x| ≤ f y + g y - |f y - g y|\n  rw [ident, ident]\n",
    )
    if proof == old["variants"][0]["proof"]:
        raise SystemExit("retry edit did not apply")
    candidate = dict(old)
    candidate["batch_id"] = BATCH_ID
    candidate["changes"] = ["unfold_function_application_before_rewrite"]
    candidate["campaign_metadata"] = {
        "reviewer": "codex-root-monotonicity-retry-round21b",
        "local_pantograph_reverification_required": True,
        "statement_changed": False,
        "semantic_change": False,
    }
    candidate["variants"] = [{"strategy": "unfold_function_application_before_rewrite", "proof": proof}]
    identity = {
        "schema_version": candidate["schema_version"],
        "record_id": RECORD_ID,
        "imports": candidate["imports"],
        "source_body": candidate["source_body"],
        "proof": proof,
        "original_row_sha256": candidate["original_row_sha256"],
    }
    candidate["candidate_hash"] = canonical_hash(identity)
    raw_statement_hash = next(
        x["reviewed_statement_sha256"]
        for x in map(json.loads, (SOURCE / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
        if x["record_id"] == RECORD_ID
    )
    review = {
        "record_id": RECORD_ID,
        "strategy": "unfold_function_application_before_rewrite",
        "reviewed_statement_sha256": raw_statement_hash,
        "reviewed_proof_sha256": sha(proof),
        "replacement": {"proof": proof},
    }
    atomic_jsonl(BATCH / "candidate_manifest.jsonl", [candidate])
    atomic_jsonl(RUN / "replacement_review_manifest.jsonl", [review])
    report = {
        "schema_version": "numinamath_expand_manual_projection_repair_v1",
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
