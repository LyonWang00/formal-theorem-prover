#!/usr/bin/env python3
"""Reject three stray-`:=` rows that are vacuous or prompt-polluted duplicates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_013000_v7.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_invalid_stray_assignment_013000.jsonl"
PROMPT_DUPLICATE = "132ef1d5-606a-5d2b-9332-9d7b4ad2804c::iff_forward_2873ade23878"
VACUOUS_BIJECTION = "6271f82c-bbf4-5579-b1c0-720b30cb88c8::and_right_bb0008ce92e0"
VACUOUS_MONOTONE = "a9d3299e-85ed-5f87-84d9-7d2c84a47123::iff_forward_5d6709836a36"


def main() -> None:
    reviews = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    rows = []
    for record_id in (PROMPT_DUPLICATE, VACUOUS_BIJECTION, VACUOUS_MONOTONE):
        row = dict(reviews[record_id])
        if row["quality_decision"] != "pending":
            raise SystemExit(f"expected pending row: {record_id}")
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-stray-assignment-semantic-audit"
        if record_id == PROMPT_DUPLICATE:
            row["review_reason"] = (
                "人工终审剔除：题文混入“翻译并直接输出”的指令，且与同批干净的正弦六次方参数范围题近重复；"
                "即使删除遗留 := 可编译，也会保留 prompt 污染与重复训练信号，故不修复晋级。"
            )
        elif record_id == VACUOUS_BIJECTION:
            row["review_reason"] = (
                "人工终审剔除：假设 f:ℝ→ℝ 为满射，同时又要求所有 f x∈[0,1)，两者直接矛盾；"
                "不连续点不可数的结论仅由不可能前提推出。删除 := 只会让 vacuous 命题通过，数学数据无效。"
            )
        else:
            row["review_reason"] = (
                "人工终审剔除：加法函数满足 f(0)=0，而 Monotone f 与 f(1)=-1 又强制 0=f(0)≤f(1)=-1，"
                "父假设自相矛盾；参数区间结论完全由空前提推出。删除 := 不能修复语义质量。"
            )
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows), "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
