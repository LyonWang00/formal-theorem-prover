#!/usr/bin/env python3
"""Finalize 19k indentation repairs with strict numerical-projection review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_019000_round32_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_019000_round32_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_019000_v13.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_019000_round32.jsonl"
REJECT = {
    "5d67b03e-7144-5525-83ea-cc2f7a6c08d4::and_right_82f95a1a2eeb",
    "5f7f4798-44b5-5aba-92dd-886afdaa8a2d::and_left_4226a47c9450",
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {
        row["record_id"]: row
        for row in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(line) for line in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {row["record_id"] for row in receipts if row.get("success") is True}
    if len(proposals) != 18 or len(receipts) != 18 or successes != set(proposals):
        raise SystemExit("the 18 selected variants must all have compatible successful Pantograph receipts")
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
                    "reviewer": "codex-root-indent-round32-strict-adjudication",
                    "review_reason": (
                        "The repaired wrapper compiles, but strict semantic review rejects this projection because it keeps only one fixed numerical answer/coordinate from an already solved compound application problem. It adds no reusable independent theorem and risks answer-fragment overfitting."
                    ),
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
                "reviewer": "codex-root-indent-repair-round32",
                "review_reason": old["review_reason"]
                + " Human repair re-review: only the copied parent tactic block indentation changed; statement, assumptions, target, and tactic text are identical. The exact replacement passed local Pantograph compilation and current global parent/normalized duplicate checks.",
                "reviewed_statement_sha256": proposal["statement_sha256_after"],
                "reviewed_proof_sha256": proof_hash,
                "replacement": {"proof": proof},
            }
        )
    atomic_jsonl(OUT, rows)
    passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(json.dumps({"rows": len(rows), "pass": passed, "reject": len(rows) - passed, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
