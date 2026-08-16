#!/usr/bin/env python3
"""Reject two pending records that fail strict semantic quality review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_invalid_reject_021000_round39.jsonl"
REASONS = {
    "2b86c69c-c6cd-58c5-a951-67ae6d324eab::and_left_8d3f3c4d83bf": (
        "Strict semantic re-review rejects this conjunction projection. It drops the second required integer condition and leaves only "
        "the existential statement that (s i - s j) * r is integral; choosing r = 0 makes it trivially true for every sequence. "
        "The candidate therefore removes the mathematical core of the parent theorem and adds no useful training signal."
    ),
    "93d631ff-82d0-5c4e-9d38-67ada22b31fb::le_to_lt_or_eq_e26f24c02477": (
        "Strict structural re-review rejects this automatically generated target because it compares the proposition ‘for all n, a n’ "
        "with a natural-number expression. The quantifier scope was split incorrectly, leaving an ill-typed Prop-order expression rather "
        "than the intended pointwise bound. This is not a repairable presentation issue of the same theorem."
    ),
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8")) if row["record_id"] in REASONS}
    if set(raw) != set(REASONS): raise SystemExit("missing frozen raw rows")
    rows = []
    for record_id in sorted(REASONS):
        row = raw[record_id]
        rows.append({"record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low",
                     "manual_reviewed": True, "reviewer": "codex-root-invalid-round39", "review_reason": REASONS[record_id],
                     "reviewed_statement_sha256": sha(str(row["lean_statement"])), "reviewed_proof_sha256": sha(str(row["proof"]))})
    atomic_jsonl(OUT, rows)
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__": main()
