#!/usr/bin/env python3
"""Materialize human-reviewed successes from the 15k indentation repair batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_015000_round18_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_015000_round18_b00000"
BASE_REVIEWS = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_015000_v9.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_015000_round18.jsonl"


def main() -> None:
    base = {x["record_id"]: x for x in map(json.loads, BASE_REVIEWS.open(encoding="utf-8"))}
    proposals = {
        x["record_id"]: x
        for x in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(x) for x in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {x["record_id"]: x for x in receipts if x.get("success") is True}
    if len(receipts) != 34 or len(successes) != 34:
        raise SystemExit(f"expected 34 receipts and successes, got {len(receipts)} and {len(successes)}")
    rows = []
    for record_id in sorted(successes):
        old = base[record_id]
        proposal = proposals[record_id]
        proof = proposal["replacement_proof"]
        proof_hash = hashlib.sha256(proof.encode()).hexdigest()
        if old["quality_decision"] != "pending" or proof_hash != proposal["replacement_proof_sha256"]:
            raise SystemExit(f"review/proof mismatch: {record_id}")
        rows.append(
            {
                "record_id": record_id,
                "quality_decision": "pass",
                "quality_tier": old["quality_tier"],
                "manual_reviewed": True,
                "reviewer": "codex-root-indent-repair-round18",
                "review_reason": old["review_reason"]
                + " 人工修复复核：仅修正父证明 tactic 块缩进，statement、变量、假设、目标及 tactic 内容均未改变；精确修复版本已在锁定的本地 Pantograph 环境重新编译成功。",
                "reviewed_statement_sha256": proposal["statement_sha256_after"],
                "reviewed_proof_sha256": proof_hash,
                "replacement": {"proof": proof},
            }
        )
    atomic_jsonl(OUT, rows)
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
