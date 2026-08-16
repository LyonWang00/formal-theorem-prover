#!/usr/bin/env python3
"""Approve only the 45 compiled-success, proof-indent-only round-8 repairs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_010000_round8_v1"
BATCH = REPAIR / "batches/numinamath_expand_repair_curated_010000_round8_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_010000_v4.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_010000_round8.jsonl"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {
        row["record_id"]: row
        for row in map(json.loads, (REPAIR / "replacement_review_manifest.jsonl").open(encoding="utf-8"))
    }
    results = {
        row["record_id"]: row
        for row in map(json.loads, (BATCH / "verification_results.jsonl").open(encoding="utf-8"))
    }
    successes = sorted(record_id for record_id, result in results.items() if result.get("success") is True)
    failures = sorted(record_id for record_id, result in results.items() if result.get("success") is False)
    if len(successes) != 45 or len(failures) != 3 or set(results) != set(proposals):
        raise SystemExit(f"unexpected round-8 result split: {len(successes)} success, {len(failures)} fail")
    output_rows = []
    for record_id in successes:
        prior = base[record_id]
        proposal = proposals[record_id]
        if prior["quality_decision"] != "pending" or prior["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"repair is not a retained pending row: {record_id}")
        proof = proposal["replacement_proof"]
        if sha_text(proof) != proposal["replacement_proof_sha256"]:
            raise SystemExit(f"replacement proof hash mismatch: {record_id}")
        output_rows.append(
            {
                "record_id": record_id,
                "quality_decision": "pass",
                "quality_tier": prior["quality_tier"],
                "manual_reviewed": True,
                "reviewer": "codex-root-curated-indent-repair-round8",
                "review_reason": (
                    prior["review_reason"]
                    + " 人工修复复核：命题保持逐字不变，仅把自动包装器下的父证明主体统一增加两个空格；"
                    "proof diff 已锁定为 indentation_only，未改变 tactic、事实或证明顺序。"
                    "修复后的精确 proof 已在本地锁定环境经 Pantograph 重新编译成功，故晋级。"
                ),
                "reviewed_statement_sha256": prior["reviewed_statement_sha256"],
                "reviewed_proof_sha256": sha_text(proof),
                "replacement": {"proof": proof},
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(OUTPUT)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "success_rows": len(output_rows),
                "remaining_failed_ids": failures,
                "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
