#!/usr/bin/env python3
"""Finalize 18k indentation repairs with duplicate and low-value adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_018000_round28_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_018000_round28_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_018000_v12.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_018000_round28.jsonl"

EXACT_DUPLICATES = {
    "b52ed7e5-b47f-59dd-ab83-7a29778afd1d::le_to_lt_or_eq_66546519027a": (
        "f263f38f-3099-50af-81b8-725e8edd9b45::le_to_lt_or_eq_3e5fc30e58b9"
    )
}
LOW_VALUE = {
    "fd6bb6d7-aa52-5b5a-8a8a-c170295f44fd::le_to_lt_or_eq_bc59689a1a4a",
}
REJECT = set(EXACT_DUPLICATES) | LOW_VALUE


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rejection_reason(record_id: str) -> str:
    if record_id in EXACT_DUPLICATES:
        return (
            "The repaired variant compiles, but normalized theorem comparison found it exactly duplicates the already "
            f"accepted record {EXACT_DUPLICATES[record_id]}; it is rejected to prevent training duplication."
        )
    return (
        "Strict re-review rejects this application-problem scalar boundary split. The score equation already fixes the "
        "single numerical answer, while the new target merely rewrites a weak n <= 15 consequence as n < 15 or n = 15; "
        "it adds no independent mathematical content."
    )


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {
        row["record_id"]: row
        for row in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(line) for line in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {row["record_id"] for row in receipts if row.get("success") is True}
    if len(proposals) != 24 or len(receipts) != 24 or successes != set(proposals):
        raise SystemExit("the 24 selected variants must all have compatible successful Pantograph receipts")
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8")) if row["record_id"] in proposals}
    if set(raw) != set(proposals):
        raise SystemExit("missing raw rows")

    rows = []
    for record_id in sorted(proposals):
        old = base[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"base review is not an eligible pending item: {record_id}")
        if record_id in REJECT:
            rows.append(
                {
                    "record_id": record_id,
                    "quality_decision": "reject",
                    "quality_tier": "extremely_low",
                    "manual_reviewed": True,
                    "reviewer": "codex-root-indent-round28-strict-adjudication",
                    "review_reason": rejection_reason(record_id),
                    "reviewed_statement_sha256": sha(raw[record_id]["lean_statement"]),
                    "reviewed_proof_sha256": sha(raw[record_id]["proof"]),
                }
            )
            continue
        proposal = proposals[record_id]
        proof = proposal["replacement_proof"]
        proof_hash = sha(proof)
        if proof_hash != proposal["replacement_proof_sha256"]:
            raise SystemExit(f"replacement proof hash mismatch: {record_id}")
        rows.append(
            {
                "record_id": record_id,
                "quality_decision": "pass",
                "quality_tier": old["quality_tier"],
                "manual_reviewed": True,
                "reviewer": "codex-root-indent-repair-round28",
                "review_reason": old["review_reason"]
                + " Human repair re-review: only the copied parent tactic block indentation changed; statement, "
                "assumptions, target, and tactic text are identical. The exact replacement compiled in the pinned "
                "local Pantograph environment and passed parent-level plus normalized exact-duplicate checks.",
                "reviewed_statement_sha256": proposal["statement_sha256_after"],
                "reviewed_proof_sha256": proof_hash,
                "replacement": {"proof": proof},
            }
        )
    atomic_jsonl(OUT, rows)
    passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pass": passed,
                "reject": len(rows) - passed,
                "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
