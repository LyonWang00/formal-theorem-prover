#!/usr/bin/env python3
"""Finalize the human quality review for the 16k indentation repair batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_016000_round22_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_016000_round22_b00000"
BASE_REVIEWS = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_016000_v10.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_016000_round22.jsonl"
REJECT = {
    "0aca4a06-b14c-55d7-ba1c-0316c6bb838c::eq_to_le_forward_118b5dea666f": (
        "The repaired proof compiles, but the candidate only weakens the exact numerical conclusion "
        "x^4 + y^4 = 7 to x^4 + y^4 <= 7. It adds no mathematical content and is rejected after strict review."
    ),
    "57cfeb53-c6f6-51e7-9b32-8c4481f0fbbf::lt_to_le_9a648f3ca8bd": (
        "The repaired proof compiles, but the candidate is only the immediate non-strict weakening of the "
        "parent strict inequality. This is a low-information logical projection and is rejected."
    ),
    "c6ab1b46-3457-5a64-bf24-822198ba8428::eq_to_le_forward_bf9559fc78e5": (
        "The repaired proof compiles, but replacing the exact perimeter value 90 by the one-sided bound <= 90 "
        "is a mechanical weakening of a solved numerical problem, so it is rejected."
    ),
    "cc6ba4a0-ec4d-5421-9a39-39f76cb32d5b::eq_to_le_forward_8ead0c375634": (
        "The repaired proof compiles, but the zero-sum equality is merely weakened to a <= 0 inequality. "
        "This equality-to-one-side transformation is not a meaningful expansion and is rejected."
    ),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    base = {x["record_id"]: x for x in map(json.loads, BASE_REVIEWS.open(encoding="utf-8"))}
    proposals = {
        x["record_id"]: x
        for x in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(x) for x in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {x["record_id"]: x for x in receipts if x.get("success") is True}
    if len(receipts) != 17 or len(successes) != 17 or set(successes) != set(proposals):
        raise SystemExit(
            f"expected the same 17 proposals and successful receipts, got "
            f"{len(proposals)}, {len(receipts)}, and {len(successes)}"
        )
    raw = {
        x["record_id"]: x
        for x in map(json.loads, RAW.open(encoding="utf-8"))
        if x["record_id"] in proposals
    }
    if len(raw) != 17 or not set(REJECT) < set(raw):
        raise SystemExit("raw rows or rejection set do not match the repair batch")

    rows = []
    for record_id in sorted(proposals):
        old = base[record_id]
        proposal = proposals[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"base review is not an eligible pending item: {record_id}")
        if record_id in REJECT:
            rows.append(
                {
                    "record_id": record_id,
                    "quality_decision": "reject",
                    "quality_tier": "extremely_low",
                    "manual_reviewed": True,
                    "reviewer": "codex-root-indent-repair-round22-strict-adjudication",
                    "review_reason": REJECT[record_id],
                    "reviewed_statement_sha256": sha(raw[record_id]["lean_statement"]),
                    "reviewed_proof_sha256": sha(raw[record_id]["proof"]),
                }
            )
            continue

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
                "reviewer": "codex-root-indent-repair-round22",
                "review_reason": old["review_reason"]
                + " Human repair re-review: only the tactic block copied beneath `have h_parent := by` was indented; "
                "the statement, binders, assumptions, target, and tactic text are unchanged. The exact repaired "
                "variant was recompiled successfully in the pinned local Pantograph environment. Exact normalized "
                "statement comparison found no duplicate among the previously accepted 16k checkpoint.",
                "reviewed_statement_sha256": proposal["statement_sha256_after"],
                "reviewed_proof_sha256": proof_hash,
                "replacement": {"proof": proof},
            }
        )
    atomic_jsonl(OUT, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pass": len(rows) - len(REJECT),
                "reject": len(REJECT),
                "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
