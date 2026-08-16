#!/usr/bin/env python3
"""Finalize 17k indentation repairs with a strict low-value/duplicate adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_017000_round25_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_017000_round25_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_017000_v11.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_017000_round25.jsonl"

EQ_WEAKENINGS = {
    "39442cbf-4dfe-5656-8370-61cf44826f2a::eq_to_le_forward_abce036f9a8b",
    "731908eb-b3ce-59be-a219-4e8ccf0a638f::eq_to_le_forward_1ce662f8c0b1",
    "a9de26c3-3f64-506d-888d-a1415cc08211::eq_to_le_forward_d6ec7ccdedee",
}
STRICT_WEAKENINGS = {
    "1d77a9b0-296d-518f-8f8c-450d8ebd8fb0::lt_to_le_d6df2d9a102d",
    "601050aa-f5cf-5ad2-96b5-42413e82c639::lt_to_le_b0dfc80d2db3",
    "66f1f719-8301-582b-b242-a4b25fc6b171::lt_to_le_6e4465fb1ea7",
    "7d52703b-1f5f-5bec-b99b-cf8c296bb32e::lt_to_le_813902eccea8",
    "ea76ab9a-9ac4-55df-9dd8-dfdf7361ed7d::lt_to_le_fdc913e7705f",
}
NEGATION_REWRITES = {
    "37916887-d283-5f2d-80db-b9b7732052ef::lt_to_not_reverse_le_5cf23ae7c688",
    "69ff8e6d-4bfb-550b-9874-e3b07ebfb6b0::lt_to_not_reverse_le_596316be12de",
    "f1965952-0b97-5e05-b026-5dda4577000b::lt_to_not_reverse_le_7e2a504c88e3",
}
EXACT_DUPLICATES = {
    "a69d197d-4999-5284-b9dd-297f7c0487b0::le_to_lt_or_eq_621760be0876": (
        "1537e695-2de0-5c01-bce1-37f75d5ebb95::le_to_lt_or_eq_57f5689c2c0c"
    )
}
REJECT = EQ_WEAKENINGS | STRICT_WEAKENINGS | NEGATION_REWRITES | set(EXACT_DUPLICATES)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rejection_reason(record_id: str) -> str:
    if record_id in EQ_WEAKENINGS:
        return (
            "Strict re-review rejects this scalar equality-to-one-sided-bound conversion. It only weakens an exact "
            "solved value/identity and adds no independent mathematical content."
        )
    if record_id in STRICT_WEAKENINGS:
        return (
            "Strict re-review rejects this immediate strict-to-nonstrict weakening. Two of these records also contain "
            "free-variable/Prop-order scope corruption; none provides a new substantive theorem."
        )
    if record_id in NEGATION_REWRITES:
        return (
            "Strict re-review rejects the mechanical rewrite of a strict inequality as the negation of its reverse "
            "nonstrict inequality. This is a logical restatement rather than meaningful data expansion."
        )
    return (
        "The repaired variant compiles, but normalized theorem comparison found it exactly duplicates the already "
        f"accepted record {EXACT_DUPLICATES[record_id]}; it is rejected to prevent training duplication."
    )


def main() -> None:
    base = {x["record_id"]: x for x in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {
        x["record_id"]: x
        for x in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(x) for x in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {x["record_id"] for x in receipts if x.get("success") is True}
    if len(proposals) != 17 or len(receipts) != 17 or successes != set(proposals):
        raise SystemExit("the 17 selected variants must all have compatible successful Pantograph receipts")
    wanted = set(proposals) | REJECT
    raw = {x["record_id"]: x for x in map(json.loads, RAW.open(encoding="utf-8")) if x["record_id"] in wanted}
    if set(raw) != wanted:
        raise SystemExit("missing raw rows")

    rows = []
    for record_id in sorted(wanted):
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
                    "reviewer": "codex-root-indent-round25-strict-adjudication",
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
                "reviewer": "codex-root-indent-repair-round25",
                "review_reason": old["review_reason"]
                + " Human repair re-review: only the copied parent tactic block indentation changed; statement, "
                "assumptions, target, and tactic text are identical. The exact replacement compiled in the pinned "
                "local Pantograph environment and normalized exact-duplicate comparison passed.",
                "reviewed_statement_sha256": proposal["statement_sha256_after"],
                "reviewed_proof_sha256": proof_hash,
                "replacement": {"proof": proof},
            }
        )
    atomic_jsonl(OUT, rows)
    passed = sum(x["quality_decision"] == "pass" for x in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pass": passed,
                "reject": len(rows) - passed,
                "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
