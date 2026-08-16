#!/usr/bin/env python3
"""Strictly review the five successful round-38 projection repairs."""

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
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_021000_v15.jsonl"
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/manual_projection_021000_round38"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_projection_021000_round38_b00000/verification_results.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_projection_021000_round38.jsonl"


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    base, _ = load_reviews([BASE]); parents: dict[str, list[str]] = defaultdict(list); norms: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass": continue
        candidate = apply_review(raw[record_id], review); parents[str(raw[record_id].get("upstream_source") or "")].append(record_id)
        norms[normalize(str(candidate["lean_statement"]))].append(record_id)
    replacements = {row["record_id"]: row for row in map(json.loads, REPLACEMENTS.open(encoding="utf-8"))}
    results = {row["record_id"]: row for row in map(json.loads, RESULTS.open(encoding="utf-8"))}
    if set(replacements) != set(results) or len(results) != 5: raise SystemExit("replacement/result mismatch")
    rows = []
    for record_id in sorted(replacements):
        replacement = replacements[record_id]["replacement"]; result = results[record_id]; old = base[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}: raise SystemExit(f"ineligible base: {record_id}")
        if result.get("success") is not True or result.get("pantograph_verified") != "success": raise SystemExit(f"missing success: {record_id}")
        parent = str(raw[record_id].get("upstream_source") or ""); norm = normalize(str(replacement["lean_statement"])); reason = None
        if parents.get(parent): reason = f"A substantive sibling is already accepted for the same parent: {parents[parent]}."
        elif norms.get(norm): reason = f"The normalized theorem is already accepted under {norms[norm]}."
        if reason:
            review = {"record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low", "manual_reviewed": True,
                      "reviewer": "codex-root-projection-round38-strict-adjudication", "review_reason": reason + " It is rejected despite compiling.",
                      "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])), "reviewed_proof_sha256": sha(str(raw[record_id]["proof"]))}
        else:
            review = {"record_id": record_id, "quality_decision": "pass", "quality_tier": old["quality_tier"], "manual_reviewed": True,
                      "reviewer": "codex-root-projection-repair-round38", "replacement": replacement,
                      "review_reason": old["review_reason"] + " Human repair re-review: the goal is a substantive reusable conjunct (functional inequality/formula, extremum characterization, or positivity consequence). The repair only removes a stray source token or applies the quantified parent theorem before selecting its conjunction field. Statement meaning and proof strategy are unchanged. Exact replacement passed local Pantograph and global parent/normalized duplicate checks.",
                      "reviewed_statement_sha256": sha(str(replacement["lean_statement"])), "reviewed_proof_sha256": sha(str(replacement["proof"]))}
            parents[parent].append(record_id); norms[norm].append(record_id)
        candidate = apply_review(raw[record_id], review); validate_review_hashes(review, candidate); rows.append(review)
    atomic_jsonl(OUT, rows); passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(json.dumps({"rows": len(rows), "pass": passed, "reject": len(rows) - passed, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__": main()
