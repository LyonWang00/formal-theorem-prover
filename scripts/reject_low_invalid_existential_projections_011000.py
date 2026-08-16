#!/usr/bin/env python3
"""Reject seven invalid or mathematically vacuous existential projections."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_011000_v5.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_low_invalid_existential_projections_011000.jsonl"
INVALID_UNIQUE = {
    "b30d8a67-a24b-50a5-a674-31dd1c8e6c70::and_left_7295eb5f6c7f",
}
VACUOUS_OR_TRIVIAL = {
    "0686c753-5665-5045-be35-42aeb0740906::and_left_c32b6208841a",
    "4fb61e0c-201b-57a2-bf0b-3b845349eea3::and_left_2e2e0984a56d",
    "78ad861d-7d05-5469-aa86-3cfc20ec8020::and_left_83f58bc51773",
    "98bc0958-9289-5ceb-8f03-b4ff1ec6840c::and_left_c7214fefcb68",
    "fa0b1da7-b1ac-542c-89f4-eecf6081ba0d::and_left_d116bd6278ae",
    "fcdea5ea-223e-58ab-8a28-b3b5909a04aa::and_left_dc7e729fe514",
}


def main() -> None:
    reviews = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    selected = INVALID_UNIQUE | VACUOUS_OR_TRIVIAL
    if any(reviews[record_id]["quality_decision"] != "pending" for record_id in selected):
        raise SystemExit("expected all seven projections to remain pending")
    rows = []
    for record_id in sorted(selected):
        row = dict(reviews[record_id])
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-existential-projection-audit"
        if record_id in INVALID_UNIQUE:
            row["review_reason"] = (
                "人工终审剔除：父命题仅保证满足 P∧Q 的函数唯一，不能推出只满足 P 的函数唯一；"
                "删除第二个约束会扩大候选集合，当前 ∃! 投影在逻辑上无效，不能靠改 tactic 修复。"
            )
        else:
            row["review_reason"] = (
                "人工终审剔除：该存在投影丢掉原题的核心联立条件后退化为显然或近乎显然的见证，"
                "例如任取整数商、q=1、k=0、任取两个不同元素或固定 gcd=1 三元组；"
                "不再承载父题的主要数学内容。为避免低质量过拟合，不修复、不进入 verified fail。"
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
