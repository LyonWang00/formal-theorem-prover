#!/usr/bin/env python3
"""Reject a free-witness projection and a weak numeric equality bound."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_013000_v7.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_invalid_or_low_pending_013000.jsonl"
FREE_WITNESS = "167d39af-1085-52b1-8229-e4985d1a028e::and_right_b712e6253d41"
WEAK_NUMERIC_BOUND = "6f503b77-5baa-5d9b-ba98-6db3311dc461::eq_to_le_forward_f859799d4f0f"


def main() -> None:
    reviews = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    rows = []
    for record_id in (FREE_WITNESS, WEAK_NUMERIC_BOUND):
        row = dict(reviews[record_id])
        if row["quality_decision"] != "pending":
            raise SystemExit(f"expected pending row: {record_id}")
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-pending-semantic-audit"
        if record_id == FREE_WITNESS:
            row["review_reason"] = (
                "人工终审剔除：父命题为“存在 y，使两个矩阵等式同时成立”，生成器却将第二分量投影成含自由 y 的结论；"
                "目标既丢失存在量词，也没有可推断的标量类型，补类型不能恢复原语义。属于 witness/binder 丢失的无效扩充。"
            )
        else:
            row["review_reason"] = (
                "人工终审剔除：父题结论只是某个闭合数值表达式精确等于 1989，当前扩充仅弱化为 ≤1989；"
                "没有新增结构、界估计或证明思想，且源 proof 还依赖已删除 API。按低质量比例门槛不值得修复。"
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
