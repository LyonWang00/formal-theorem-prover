#!/usr/bin/env python3
"""Materialize strict manual rejections for malformed pending expansions."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_009000_v3.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_invalid_pending_009000_v2.jsonl"

EQ_SCOPE_OR_MECHANICAL = {
    "1cec38c3-4487-5403-9951-ce7af1c3f3ae::eq_to_le_forward_443c43f8897f",
    "286d0a07-3f0e-5938-829b-4d923a3ab5fb::eq_to_le_forward_ef3d0a707386",
    "4d314523-cc51-55f8-b4cc-a34b47861e58::eq_to_le_forward_ad8834f35b4a",
    "617130a6-be14-5735-9b86-f1bb26474a23::eq_to_le_forward_d5b5bc9d4efd",
    "6f907cb1-d5f8-582c-adaf-87012879d951::eq_to_le_forward_606fba7f10fc",
    "9b39792b-5a87-5475-a37e-a21e01eada53::eq_to_le_forward_bfe85e22f61c",
    "a536adc4-18ce-5ca8-b979-0e47803ee10d::eq_to_le_forward_30277172cc5c",
    "a9120731-748e-598d-850c-b0c80dac2856::eq_to_le_forward_b7c85fd4053c",
    "c72ca028-4901-5213-b0f0-018c0e062910::eq_to_le_forward_6ea52df06cc6",
    "f2f177e8-657d-5e2f-9f7f-7ac2fa88e097::eq_to_le_forward_a84063b4dc57",
    "f6a5e3eb-57af-599f-84ec-bf67ed0e7fd6::eq_to_le_forward_b9ea07ce6097",
}

MALFORMED_CONTRAPOSITIVE = {
    "02ab5c28-678f-5ea2-96d6-b07204f5348d::implication_contrapositive_b62f23e0ab6a",
    "147e5650-9218-5e1a-8d2c-8f24f15a5ed0::implication_contrapositive_ff8cdf88203e",
    "3fee1f41-5a71-5b5a-8212-730a582f4ff4::implication_contrapositive_1f9e57eccb71",
    "62f1bb55-f4c7-5e70-b16f-fc95f055835b::implication_contrapositive_b7c7b5ffcb15",
    "846e7f5f-896a-57fd-a396-7c8724cac5f9::implication_contrapositive_4a0c824706d5",
    "9737693c-75e3-5a49-bedf-92e4ae6cfba0::implication_contrapositive_372e1a543952",
}

LOW_VALUE_CONTRAPOSITIVE = {
    "53a100bb-e9f6-59c9-9099-080b1c25495d::implication_contrapositive_97776fcf7ab4",
}


def main() -> None:
    rows = {row["record_id"]: row for row in map(json.loads, INPUT.open(encoding="utf-8"))}
    selected = EQ_SCOPE_OR_MECHANICAL | MALFORMED_CONTRAPOSITIVE | LOW_VALUE_CONTRAPOSITIVE
    missing = selected - rows.keys()
    if missing:
        raise SystemExit(f"missing review rows: {sorted(missing)}")

    output_rows = []
    for record_id in sorted(selected):
        row = dict(rows[record_id])
        if row["quality_decision"] != "pending":
            raise SystemExit(f"expected pending decision: {record_id}")
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-strict-scope-audit"
        if record_id in EQ_SCOPE_OR_MECHANICAL:
            row["review_reason"] = (
                "人工终审剔除：自动生成器在存在量词、全称量词或合取内部切分等号，"
                "得到自由变量、Prop 与数值混合比较或丢失绑定范围的畸形目标；"
                "少数可恢复成逐点不等式的条目也只是把精确公式机械弱化为单侧界，"
                "没有新增独立数学内容。该条不修复、不进入 verified fail。"
            )
        elif record_id in MALFORMED_CONTRAPOSITIVE:
            row["review_reason"] = (
                "人工终审剔除：所谓逆否命题把父命题的存在/唯一存在/全称绑定范围切断，"
                "使见证或变量逃逸，并非父命题的合法逆否；即使重新补变量也不能由父证明"
                "推出当前任意见证版本。该条语义无效，不修复、不进入 verified fail。"
            )
        else:
            row["review_reason"] = (
                "人工终审剔除：修正绑定范围后只能得到“若 n 不整除任何 A m，则 n<0”的点态逆否，"
                "但 A 0=0 已使任意整数 n 都整除 A 0；该变体只是对显然见证的逻辑换皮，"
                "没有独立数学内容。为避免低质量过拟合，不修复、不进入 verified fail。"
            )
        output_rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "rows": len(output_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
