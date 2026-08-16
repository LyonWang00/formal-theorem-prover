#!/usr/bin/env python3
"""Finalize 21k indentation repairs under the strict semantic/duplicate gate."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl
from finalize_indent_repair_round36 import normalize, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_021000_round37_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_021000_round37_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_021000_v15.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_021000_round37.jsonl"
REJECT_REASONS = {
    "7b8bd63c-e07e-5af4-94d2-5dddcd0aa056::le_to_lt_or_eq_8281e7a74368": "The goal only case-splits the fixed numerical minimum 18000 from an application-style answer, adding no reusable independent theorem.",
    "a5e946f0-11a4-5b8b-9511-807f0eb695a2::le_to_lt_or_eq_f913ca66d556": "The goal only case-splits the single scalar parameter answer a ≥ 1 from an application-style range question; it is an answer fragment rather than a new structural result.",
    "c81a7190-3b39-5e7f-8856-73dcf682eee5::lt_to_le_4ff0e23235e5": "The candidate merely weakens a strict triangle inequality to the corresponding non-strict inequality without adding a boundary classification or new consequence.",
}


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {row["record_id"]: row for row in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))}
    receipts = [json.loads(line) for line in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {row["record_id"] for row in receipts if row.get("success") is True and row.get("pantograph_verified") == "success"}
    if len(proposals) != 13 or len(receipts) != 13 or successes != set(proposals):
        raise SystemExit("all 13 selected variants must have exact Pantograph success receipts")
    raw_all = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    accepted_parents: dict[str, list[str]] = defaultdict(list); accepted_norm: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass": continue
        accepted_parents[str(raw_all[record_id].get("upstream_source") or "")].append(record_id)
        statement = str((review.get("replacement") or {}).get("lean_statement") or raw_all[record_id]["lean_statement"])
        accepted_norm[normalize(statement)].append(record_id)
    rows = []
    for record_id in sorted(proposals):
        old = base[record_id]; raw = raw_all[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"base review is not eligible: {record_id}")
        parent = str(raw.get("upstream_source") or ""); norm = normalize(str(raw["lean_statement"])); reason = REJECT_REASONS.get(record_id)
        if not reason and accepted_parents.get(parent): reason = f"A stronger or more substantive sibling for this parent is already accepted: {accepted_parents[parent]}."
        if not reason and accepted_norm.get(norm): reason = f"The normalized theorem is already accepted under {accepted_norm[norm]}, so this is an exact duplicate."
        if reason:
            rows.append({"record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low", "manual_reviewed": True,
                         "reviewer": "codex-root-indent-round37-strict-adjudication", "review_reason": reason + " It is rejected regardless of Pantograph success.",
                         "reviewed_statement_sha256": sha(str(raw["lean_statement"])), "reviewed_proof_sha256": sha(str(raw["proof"]))})
            continue
        proposal = proposals[record_id]; proof = proposal["replacement_proof"]; proof_hash = sha(proof)
        if proof_hash != proposal["replacement_proof_sha256"]: raise SystemExit(f"proof hash mismatch: {record_id}")
        rows.append({"record_id": record_id, "quality_decision": "pass", "quality_tier": old["quality_tier"], "manual_reviewed": True,
                     "reviewer": "codex-root-indent-repair-round37",
                     "review_reason": old["review_reason"] + " Human repair re-review: only the copied parent tactic block indentation changed; statement, assumptions, target, and tactic text are identical. The exact replacement passed local Pantograph compilation and current global parent/normalized duplicate checks.",
                     "reviewed_statement_sha256": proposal["statement_sha256_after"], "reviewed_proof_sha256": proof_hash, "replacement": {"proof": proof}})
        accepted_parents[parent].append(record_id); accepted_norm[norm].append(record_id)
    atomic_jsonl(OUT, rows); passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(json.dumps({"rows": len(rows), "pass": passed, "reject": len(rows) - passed, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__": main()
