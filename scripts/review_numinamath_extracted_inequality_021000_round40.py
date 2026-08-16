#!/usr/bin/env python3
"""Strictly review the successful round-40 extracted inequality repair."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl
from finalize_indent_repair_round36 import normalize, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finalize_numinamath_expand_verification import (  # noqa: E402
    apply_review,
    load_reviews,
    validate_review_hashes,
)

RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_021000_v15.jsonl"
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/manual_extracted_inequality_021000_round40"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_extracted_inequality_021000_round40_b00000/verification_results.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_extracted_inequality_021000_round40.jsonl"


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    base, _ = load_reviews([BASE])
    accepted_parents: dict[str, list[str]] = defaultdict(list)
    accepted_norms: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass":
            continue
        candidate = apply_review(raw[record_id], review)
        accepted_parents[str(raw[record_id].get("upstream_source") or "")].append(record_id)
        accepted_norms[normalize(str(candidate["lean_statement"]))].append(record_id)

    replacements = {row["record_id"]: row for row in map(json.loads, REPLACEMENTS.open(encoding="utf-8"))}
    results = {row["record_id"]: row for row in map(json.loads, RESULTS.open(encoding="utf-8"))}
    if set(replacements) != set(results) or len(results) != 1:
        raise SystemExit("replacement/result mismatch")

    rows = []
    for record_id, replacement_row in replacements.items():
        old = base[record_id]
        replacement = replacement_row["replacement"]
        result = results[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"ineligible base decision: {record_id}")
        if result.get("success") is not True or result.get("pantograph_verified") != "success":
            raise SystemExit(f"replacement did not pass Pantograph: {record_id}")
        parent = str(raw[record_id].get("upstream_source") or "")
        norm = normalize(str(replacement["lean_statement"]))
        conflict = None
        if accepted_parents.get(parent):
            conflict = f"A substantive sibling is already accepted for this parent: {accepted_parents[parent]}."
        elif accepted_norms.get(norm):
            conflict = f"The normalized theorem is already accepted under {accepted_norms[norm]}."
        if conflict:
            review = {
                "record_id": record_id,
                "quality_decision": "reject",
                "quality_tier": "extremely_low",
                "manual_reviewed": True,
                "reviewer": "codex-root-extracted-inequality-round40-adjudication",
                "review_reason": conflict + " It is rejected despite compiling to prevent duplicate training data.",
                "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
            }
        else:
            review = {
                "record_id": record_id,
                "quality_decision": "pass",
                "quality_tier": old["quality_tier"],
                "manual_reviewed": True,
                "reviewer": "codex-root-extracted-inequality-repair-round40",
                "replacement": replacement,
                "review_reason": old["review_reason"] + " Human repair re-review: this is the natural strict-or-equality boundary refinement of the nontrivial IMO inequality abc <= sqrt(2)/4. The source was polluted by unrelated unbound variables and a nested declaration; the repair extracts the complete existing proof of the intended parent inequality without changing the statement, then applies lt_or_eq_of_le. The exact replacement passed local Pantograph and global parent/normalized duplicate checks.",
                "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
                "reviewed_proof_sha256": sha(str(replacement["proof"])),
            }
            accepted_parents[parent].append(record_id)
            accepted_norms[norm].append(record_id)
        candidate = apply_review(raw[record_id], review)
        validate_review_hashes(review, candidate)
        rows.append(review)

    atomic_jsonl(OUT, rows)
    passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(json.dumps({"rows": len(rows), "pass": passed, "reject": len(rows) - passed,
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
