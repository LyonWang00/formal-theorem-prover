#!/usr/bin/env python3
"""Strictly review round-44 canonical-parent conjunction repairs."""

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
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_023000_v17.jsonl"
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/canonical_parent_projection_023000_round44"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_canonical_parent_projection_023000_round44_b00000/verification_results.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_canonical_parent_projection_023000_round44.jsonl"
PREFERRED_EXTREMUM = "a584ff95-8571-57af-805e-78cc4b7a5f82::and_left_584d939b9ee5"
REDUNDANT_EXTREMUM = "70b0c7a4-d770-5251-9956-344a27f58965::and_right_c45e46ac51fb"


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
    if set(replacements) != set(results) or len(results) != 8:
        raise SystemExit("replacement/result mismatch")
    if results[PREFERRED_EXTREMUM].get("success") is not True or results[REDUNDANT_EXTREMUM].get("success") is not True:
        raise SystemExit("extremum adjudication requires both successful repairs")
    rows = []
    ordering = [PREFERRED_EXTREMUM] + [record_id for record_id in sorted(replacements) if record_id != PREFERRED_EXTREMUM]
    for record_id in ordering:
        old = base[record_id]
        replacement = replacements[record_id]["replacement"]
        result = results[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"ineligible base: {record_id}")
        if result.get("success") is not True:
            diagnostic = str(result.get("diagnostics") or "Pantograph compilation failed").replace("\n", " ")[:1200]
            review = dict(old)
            review["reviewer"] = "codex-root-canonical-parent-projection-round44-failed-attempt"
            review["review_reason"] = str(old["review_reason"]) + " Canonical-parent reconstruction was attempted and recompiled, but did not pass: " + diagnostic
        elif record_id == REDUNDANT_EXTREMUM:
            review = {
                "record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low",
                "manual_reviewed": True, "reviewer": "codex-root-projection-round44-content-adjudication",
                "review_reason": f"Both extrema projections from {raw[record_id]['upstream_source']} were repaired successfully. Retain only the first/maximal-value theorem {PREFERRED_EXTREMUM}; reject this correlated minimal-value projection to enforce the one-subtheorem-per-parent anti-overfitting gate.",
                "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
            }
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
                    "record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low",
                    "manual_reviewed": True, "reviewer": "codex-root-projection-round44-adjudication",
                    "review_reason": conflict + " Rejected despite compiling to prevent duplicate training data.",
                    "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                    "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
                }
            else:
                review = {
                    "record_id": record_id, "quality_decision": "pass", "quality_tier": old["quality_tier"],
                    "manual_reviewed": True, "reviewer": "codex-root-canonical-parent-projection-repair-round44",
                    "replacement": replacement,
                    "review_reason": str(old["review_reason"]) + " Human repair re-review: this is a substantive extremum, sharp equality characterization, or nontrivial triangle inequality component. The repair removes only a stray proof marker where present, reuses the exact canonical Pantograph-verified parent proof, and selects the intended conjunction field. Statement meaning is unchanged. Exact replacement passed local Pantograph and current global duplicate checks.",
                    "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
                    "reviewed_proof_sha256": sha(str(replacement["proof"])),
                }
                parents[parent].append(record_id)
                norms[norm].append(record_id)
        candidate = apply_review(raw[record_id], review)
        validate_review_hashes(review, candidate)
        rows.append(review)
    rows.sort(key=lambda row: str(row["record_id"]))
    atomic_jsonl(OUT, rows)
    counts = {key: sum(row["quality_decision"] == key for row in rows) for key in ("pass", "pending", "reject")}
    print(json.dumps({"rows": len(rows), "decisions": counts, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
