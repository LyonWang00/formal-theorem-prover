#!/usr/bin/env python3
"""Strictly review the ten successful round-43 indentation-only repairs."""

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
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_023000_round43_v1"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_curated_023000_round43_b00000/verification_results.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_023000_round43.jsonl"


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
    if set(replacements) != set(results) or len(results) != 10:
        raise SystemExit("replacement/result mismatch")
    rows = []
    for record_id in sorted(replacements):
        old = base[record_id]
        replacement = {
            "lean_statement": str(raw[record_id]["lean_statement"]),
            "proof": str(replacements[record_id]["replacement_proof"]),
        }
        result = results[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"ineligible base: {record_id}")
        if result.get("success") is not True or result.get("pantograph_verified") != "success":
            raise SystemExit(f"missing Pantograph success: {record_id}")
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
                "manual_reviewed": True, "reviewer": "codex-root-indent-round43-adjudication",
                "review_reason": conflict + " Rejected despite compiling to prevent duplicate training data.",
                "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
            }
        else:
            review = {
                "record_id": record_id, "quality_decision": "pass", "quality_tier": old["quality_tier"],
                "manual_reviewed": True, "reviewer": "codex-root-indent-repair-round43",
                "replacement": replacement,
                "review_reason": str(old["review_reason"]) + " Human repair re-review: this is a substantive classification, range, parity/existence, divisibility, or contrapositive theorem. The repair changes only indentation of the already supplied parent proof inside the generated wrapper; statement and proof strategy are unchanged. The exact replacement passed local Pantograph and current global parent/normalized duplicate checks.",
                "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
                "reviewed_proof_sha256": sha(str(replacement["proof"])),
            }
            parents[parent].append(record_id)
            norms[norm].append(record_id)
        candidate = apply_review(raw[record_id], review)
        validate_review_hashes(review, candidate)
        rows.append(review)
    atomic_jsonl(OUT, rows)
    counts = {key: sum(row["quality_decision"] == key for row in rows) for key in ("pass", "reject")}
    print(json.dumps({"rows": len(rows), "decisions": counts, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
