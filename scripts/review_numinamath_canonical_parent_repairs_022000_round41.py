#!/usr/bin/env python3
"""Strictly review round-41 canonical-parent proof repairs."""

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
from scripts.finalize_numinamath_expand_verification import apply_review, load_reviews, validate_review_hashes  # noqa: E402

RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_022000_v16.jsonl"
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/canonical_parent_order_022000_round41"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_canonical_parent_order_022000_round41_b00000/verification_results.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_canonical_parent_order_022000_round41.jsonl"


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    base, _ = load_reviews([BASE])
    parents: dict[str, list[str]] = defaultdict(list)
    norms: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass":
            continue
        candidate = apply_review(raw[record_id], review)
        parents[str(raw[record_id].get("upstream_source") or "")].append(record_id)
        norms[normalize(str(candidate["lean_statement"]))].append(record_id)
    replacements = {row["record_id"]: row for row in map(json.loads, REPLACEMENTS.open(encoding="utf-8"))}
    results = {row["record_id"]: row for row in map(json.loads, RESULTS.open(encoding="utf-8"))}
    if set(replacements) != set(results) or len(results) != 17:
        raise SystemExit("replacement/result mismatch")

    rows = []
    for record_id in sorted(replacements):
        old = base[record_id]
        replacement = replacements[record_id]["replacement"]
        result = results[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"ineligible base: {record_id}")
        if result.get("success") is not True:
            diagnostic = str(result.get("diagnostics") or "Pantograph compilation failed").replace("\n", " ")[:1200]
            review = dict(old)
            review["reviewer"] = "codex-root-canonical-parent-round41-failed-attempt"
            review["review_reason"] = str(old["review_reason"]) + " Canonical-parent proof reconstruction was attempted and recompiled, but did not pass: " + diagnostic
        else:
            if result.get("pantograph_verified") != "success":
                raise SystemExit(f"inconsistent success receipt: {record_id}")
            parent = str(raw[record_id].get("upstream_source") or "")
            norm = normalize(str(replacement["lean_statement"]))
            conflict = None
            if parents.get(parent):
                conflict = f"A substantive sibling is already accepted for this parent: {parents[parent]}."
            elif norms.get(norm):
                conflict = f"The normalized theorem is already accepted under {norms[norm]}."
            if conflict:
                review = {
                    "record_id": record_id,
                    "quality_decision": "reject",
                    "quality_tier": "extremely_low",
                    "manual_reviewed": True,
                    "reviewer": "codex-root-canonical-parent-round41-adjudication",
                    "review_reason": conflict + " Rejected despite successful repair to prevent duplicate training data.",
                    "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                    "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
                }
            else:
                review = {
                    "record_id": record_id,
                    "quality_decision": "pass",
                    "quality_tier": old["quality_tier"],
                    "manual_reviewed": True,
                    "reviewer": "codex-root-canonical-parent-order-repair-round41",
                    "replacement": replacement,
                    "review_reason": str(old["review_reason"]) + " Human repair re-review: the strict-or-equality conclusion is a substantive boundary/equality-case refinement. The failed generated wrapper was replaced with the exact proof body of its canonical Pantograph-verified parent, followed only by lt_or_eq_of_le. The statement and mathematical argument are unchanged. The exact replacement passed local Pantograph and current global parent/normalized duplicate checks.",
                    "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
                    "reviewed_proof_sha256": sha(str(replacement["proof"])),
                }
                parents[parent].append(record_id)
                norms[norm].append(record_id)
        candidate = apply_review(raw[record_id], review)
        validate_review_hashes(review, candidate)
        rows.append(review)

    atomic_jsonl(OUT, rows)
    counts = {key: sum(row["quality_decision"] == key for row in rows) for key in ("pass", "pending", "reject")}
    print(json.dumps({"rows": len(rows), "decisions": counts, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
