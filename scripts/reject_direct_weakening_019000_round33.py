#!/usr/bin/env python3
"""Reject a direct strict-to-nonstrict weakening found in the 19k review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_019000_v13.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_direct_weakening_reject_019000_round33.jsonl"
RECORD_ID = "e9e70190-6421-5999-9e1c-a17b50c873dd::lt_to_le_439812050142"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    base = next(row for row in map(json.loads, BASE.open(encoding="utf-8")) if row["record_id"] == RECORD_ID)
    raw = next(row for row in map(json.loads, RAW.open(encoding="utf-8")) if row["record_id"] == RECORD_ID)
    if base["quality_decision"] != "pending":
        raise SystemExit("direct weakening is not pending")
    row = {
        "record_id": RECORD_ID,
        "quality_decision": "reject",
        "quality_tier": "extremely_low",
        "manual_reviewed": True,
        "reviewer": "codex-root-direct-weakening-adjudication-round33",
        "review_reason": (
            "Strict re-review rejects this immediate strict-to-nonstrict weakening. The target only replaces the original < conclusion by <=, while the copied proof is also polluted by an adjacent theorem declaration; it adds no independent mathematical content."
        ),
        "reviewed_statement_sha256": sha(raw["lean_statement"]),
        "reviewed_proof_sha256": sha(raw["proof"]),
    }
    atomic_jsonl(OUT, [row])
    print(json.dumps({"rows": 1, "reject": 1, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
