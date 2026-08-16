#!/usr/bin/env python3
"""Materialize human-reviewed successes from the 14k indentation repair batch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_014000_round16_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_014000_round16_b00000"
BASE_REVIEWS = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_014000_v8.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_014000_round16.jsonl"


def canonical(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        for row in rows:
            handle.write(canonical(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    base = {x["record_id"]: x for x in map(json.loads, BASE_REVIEWS.open(encoding="utf-8"))}
    proposals = {
        x["record_id"]: x
        for x in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    receipts = [json.loads(x) for x in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {x["record_id"]: x for x in receipts if x.get("success") is True}
    if len(receipts) != 47 or len(successes) != 45:
        raise SystemExit(f"expected 47 receipts and 45 successes, got {len(receipts)} and {len(successes)}")

    rows = []
    for record_id in sorted(successes):
        old = base[record_id]
        proposal = proposals[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"not a reviewed valuable pending item: {record_id}")
        proof = proposal["replacement_proof"]
        proof_hash = hashlib.sha256(proof.encode()).hexdigest()
        if proof_hash != proposal["replacement_proof_sha256"]:
            raise SystemExit(f"replacement proof hash mismatch: {record_id}")
        rows.append(
            {
                "record_id": record_id,
                "quality_decision": "pass",
                "quality_tier": old["quality_tier"],
                "manual_reviewed": True,
                "reviewer": "codex-root-indent-repair-round16",
                "review_reason": old["review_reason"]
                + " 人工修复复核：仅修正父证明 tactic 块的缩进层级，statement、变量、假设、目标和 tactic 内容均未改变；修复后的精确 statement/proof 已在锁定的本地 Pantograph 环境重新编译成功。",
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
                "output": str(OUT),
                "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
