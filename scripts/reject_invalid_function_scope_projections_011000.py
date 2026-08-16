#!/usr/bin/env python3
"""Reject function-scope projections that are vacuous or not entailed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_011000_v5.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_invalid_function_scope_projections_011000.jsonl"
VACUOUS = "0eb4be05-18da-5869-bf14-c5c3ad69f52e::and_right_85910dbc139d"
NOT_ENTAILED = "62075bf7-552d-5208-b403-a1dd2e9cd676::and_right_fcb81c1c462a"
MOD5_NOT_ENTAILED = "25c22de5-c2d9-5c19-9d0f-9e5a4200f4f3::and_right_027f631b45aa"
SQUARES_NOT_ENTAILED = "30dd75b5-251b-5a62-9c39-458c2007e9d9::and_left_b76abbd8a900"


def main() -> None:
    reviews = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    rows = []
    for record_id in (VACUOUS, NOT_ENTAILED, MOD5_NOT_ENTAILED, SQUARES_NOT_ENTAILED):
        row = dict(reviews[record_id])
        if row["quality_decision"] != "pending":
            raise SystemExit(f"expected pending row: {record_id}")
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-function-scope-audit"
        if record_id == VACUOUS:
            row["review_reason"] = (
                "人工终审剔除：生成器丢失 n、x 及区间前提；即使恢复绑定，父假设对四个相同点同时要求"
                "严格函数值不等式和 0<0，本身矛盾，结论仅由空前提推出，属于 vacuous 数据。"
            )
        elif record_id == NOT_ENTAILED:
            row["review_reason"] = (
                "人工终审剔除：父 Lean 命题因 →/∃/∧ 绑定范围实际只在给定第一方向见证时携带第二方向函数；"
                "它不能在没有第一方向假设时推出当前逆向命题。当前 and_right 不被父证明蕴含，语义无效。"
            )
        elif record_id == MOD5_NOT_ENTAILED:
            row["review_reason"] = (
                "人工终审剔除：Pantograph 复编确认父命题实际解析为“m%5=0 → (值=4 ∧ (m%5≠0 → 值=-1))”。"
                "当前目标只给 m%5≠0，无法取得父命题所需的 m%5=0，故所谓第二投影并不被父命题蕴含；显式恢复 m 后仍失败，属于生成器误切箭头/合取作用域。"
            )
        else:
            row["review_reason"] = (
                "人工终审剔除：Pantograph 复编确认父命题中的 ∃/∧/→ 结合范围不是两个并列蕴含。"
                "父证明在选取见证后要求一个额外函数前提，并未给出 m²=2n-1；当前目标因此不能由父命题投影得到。第二次按真实绑定范围重写仍无法构造见证，判定语义无效。"
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
