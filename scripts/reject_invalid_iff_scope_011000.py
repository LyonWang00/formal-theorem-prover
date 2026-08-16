#!/usr/bin/env python3
"""Reject pending iff projections whose parent binder scope does not entail them."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_011000_v5.jsonl"
ROUND9 = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_011000_round9.jsonl"
RESULTS = ROOT / "outputs/numinamath_expand_verification/batches"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_corrections/reject_invalid_iff_scope_011000.jsonl"


def main() -> None:
    reviews = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    reviews.update({row["record_id"]: row for row in map(json.loads, ROUND9.open(encoding="utf-8"))})
    results = {}
    for path in sorted(RESULTS.glob("*/verification_results.jsonl")):
        for row in map(json.loads, path.open(encoding="utf-8")):
            results[row["record_id"]] = row
    markers = ("Exists.mp", "Function.mp", "And.mp")
    selected = []
    for record_id, review in reviews.items():
        diagnostics = str(results.get(record_id, {}).get("diagnostics") or "")
        if (
            review["quality_decision"] == "pending"
            and "::iff_forward_" in record_id
            and any(marker in diagnostics for marker in markers)
        ):
            selected.append(record_id)
    if len(selected) != 24:
        raise SystemExit(f"expected 24 invalid binder-scope iff rows, got {len(selected)}")
    output_rows = []
    for record_id in sorted(selected):
        row = dict(reviews[record_id])
        row["quality_decision"] = "reject"
        row["quality_tier"] = "extremely_low"
        row["reviewer"] = "codex-root-binder-precedence-audit"
        row["review_reason"] = (
            "人工终审剔除：父 Lean 定理受 ∃/∀/→ 与 ↔ 的绑定优先级影响，实际类型是“存在一个固定见证"
            "（或在固定前提/逐点变量下）使 P ↔ Q”，而扩充目标把它误当成“(∃见证 P) ↔ Q”的正向投影。"
            "父证明只建立固定见证处的等价，不能推出任意存在见证或丢失绑定变量后的当前目标；Pantograph 的 "
            "Exists.mp/Function.mp/And.mp 诊断正是该语义错位的结果。该条不是合法 iff_forward，不修复、"
            "不进入 verified fail。"
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
    print(json.dumps({"output": str(OUTPUT), "rows": len(output_rows), "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
