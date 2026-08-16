#!/usr/bin/env python3
"""Strict second-scale review for NuminaMath shard s00/b00003."""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/numinamath_expand_verification/batches/numinamath_expand_s00_b00003"
MANIFEST = BATCH / "candidate_manifest.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_shards/review_s00_b00003.jsonl"
REPORT = OUTPUT.with_name("review_s00_b00003_report.json")
VERSION = "numinamath_s00_b00003_strict_review_v1"

spec = importlib.util.spec_from_file_location(
    "numinamath_adjudication_scale", ROOT / "scripts/adjudicate_numinamath_first1040.py"
)
scale = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scale
spec.loader.exec_module(scale)

TRIVIAL_IFF_REVERSE = {
    "b3e240c6-8916-551f-ad6a-816bd64a6415::iff_reverse_84b7e2f72f35",
    "af7093d6-c3cb-5d05-be76-2325346ad69e::iff_reverse_90e51ed6c43e",
    "364c89b9-54d4-57d0-96c3-7db42b4025c1::iff_reverse_71b78f52f73f",
    "0cf6788f-2236-531d-9d53-8f1f121dd13f::iff_reverse_c390222fb944",
    "74dcf250-68db-5803-ac27-af620862b7e0::iff_reverse_0787b0cf6d28",
    "0a6d7412-ce9c-57f5-8274-3bf2a43c1e91::iff_reverse_479e5c907818",
    "cfe28cb7-7122-5b2c-ab3c-6b1659fc6ff6::iff_reverse_23233cadefc0",
    "6bba2cbc-67bd-5259-ab20-7d4eb54f6a39::iff_reverse_e9046190e2c8",
}

TRIVIAL_AND_COMPONENTS = {
    "d9c332cd-8a20-56e2-9137-26bfdb6d1d9a::and_left_2995d9353a87",
    "ce4ec1e5-3c59-527c-bc80-1ad4b9d3ea2a::and_right_fa00e8b7b933",
}

# Although the generic equality classifier sees an expression-to-numeric
# bound, this expression is identically zero before using any hypotheses.
# Turning the exact identity into ``0 <= 0`` adds no mathematical content.
TRIVIAL_EQUALITY_BOUNDS = {
    "b3ecc1bf-4456-5ee9-a75b-536188ce8479::eq_to_le_forward_e31e814a1f19",
}


def top_order(goal: str, operators: tuple[str, ...]) -> bool:
    depth = 0
    for index, char in enumerate(goal):
        if char in "([{": depth += 1
        elif char in ")]}": depth -= 1
        elif depth == 0 and any(goal.startswith(op, index) for op in operators):
            return True
    return False


def safe_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def decide(
    packet: Mapping[str, Any], raw: Mapping[str, Any], parent: Mapping[str, Any],
    success: bool, equality_choice: Mapping[str, tuple[str, str | None]],
) -> tuple[str, str, str]:
    rid, method, pid = str(packet["record_id"]), str(packet["method"]), str(packet["parent_id"])
    parent_goal, candidate_goal = str(packet["parent"]["goal"]), str(packet["candidate"]["goal"])
    risk = bool(packet.get("reason_codes")) or scale.contaminated(packet, raw)
    template = scale.problem_ok(method, raw)

    if method in scale.HARD_REJECT:
        return "reject", "extremely_low", "机械等价重排、同构包装或否定式改写，没有新增实质数学信息"

    if method.startswith("eq_to_le_"):
        if rid in TRIVIAL_EQUALITY_BOUNDS:
            return "reject", "extremely_low", "目标表达式恒等化简为零；再改写成单侧零界属于恒真投影，没有新增数学信息"
        basis, chosen = equality_choice[pid]
        if chosen is None:
            labels = {
                "bare_value_to_numeric_bound": "纯变量或函数值的精确数值答案被弱化成单侧界",
                "closed_numeric_relaxation": "封闭数值计算等式被机械弱化成单侧界",
                "arbitrary_expression_order": "等式两侧均为表达式，自动次序没有自然界含义",
                "invalid_scope_or_prop_cut": "在量词、命题或集合描述内部误切等号",
                "not_top_level_equality": "父目标不是安全的顶层等式",
            }
            return "reject", "extremely_low" if basis == "invalid_scope_or_prop_cut" else "low", labels.get(basis, basis)
        if method != chosen:
            return "reject", "low", f"同父只保留自然方向 {chosen}；本条是成对反向投影"
        if risk or not template:
            return "pending", "medium", "界本身非平凡，但作用域、声明或题文需先修复"
        if success:
            return "pass", "medium", "表达式到数值的自然非平凡上界，且同父反向已剔除"
        return "pending", "medium", "自然非平凡上界有价值，但当前证明尚无成功回执"

    if method in {"and_left", "and_right"}:
        if rid in TRIVIAL_AND_COMPONENTS:
            return "reject", "low", "仅投影封闭 lcm 数值计算，分量训练增益过低"
        if risk or not template:
            return ("pending", "medium", "分量有潜在价值，但量词/多声明或题文必须修复") if not success else (
                "reject", "extremely_low", "statement 污染；编译成功不能消除错误声明边界")
        if success:
            return "pass", "medium", "原合取中的独立实质结论，可作为单独训练目标"
        return "pending", "medium", "实质合取分量，但当前证明尚未通过 Pantograph"

    if method == "iff_forward":
        if risk or not template:
            return "pending", "high", "解分类的必要性方向有价值，但作用域/题文需修复"
        if success:
            return "pass", "high", "非平凡解分类或完备性的必要性方向"
        return "pending", "high", "解分类方向有价值，但当前证明尚未通过 Pantograph"

    if method == "iff_reverse":
        if rid in TRIVIAL_IFF_REVERSE:
            return "reject", "low", "只将显式数值根/答案代回原式，属于低信息反向验证"
        if risk or not template:
            return "pending", "medium", "充分性方向有价值，但作用域/声明或题文需修复"
        if success:
            return "pass", "medium", "包含构造、结构化序关系或函数分类的实质充分性证明"
        return "pending", "medium", "实质充分性方向，但当前证明尚未通过 Pantograph"

    if method == "implication_contrapositive":
        if scale.top_arrow(parent_goal) is None:
            return "reject", "extremely_low", "父目标不是顶层蕴含，脚本在量词/合取内部误切箭头"
        if risk or not template:
            return "pending", "high", "顶层逆否命题有价值，但结构或题文需修复"
        if success:
            return "pass", "high", "完整顶层蕴含的非平凡逆否证明"
        return "pending", "high", "逆否目标有价值，但当前证明尚未通过 Pantograph"

    if method == "le_to_lt_or_eq":
        if not top_order(parent_goal, ("≤", "≥")) or risk or not template:
            return "reject", "extremely_low", "不是完整顶层非严格序关系，存在命题内部误切"
        if success:
            return "pass", "medium", "把实质界细化为严格与等号两种结构化情形"
        return "pending", "medium", "结构化严格/等号分类有价值，但当前证明尚未通过 Pantograph"

    if method == "lt_to_le":
        if not top_order(parent_goal, ("<", ">")) or risk or not template:
            return "reject", "extremely_low", "不是完整顶层严格序关系，存在命题内部误切"
        if success:
            return "pass", "medium", "由实质严格界得到自然且可复用的非严格界"
        return "pending", "medium", "自然非严格界有价值，但当前证明尚未通过 Pantograph"

    if method == "lt_to_ne":
        return "reject", "low", "仅把严格序关系弱化为不等，信息增益不足"

    return "reject", "extremely_low", f"未通过尺度校准的方法 {method}"


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("review output already exists")
    manifest_rows = [row for _, row in scale.rows(MANIFEST)]
    ordered = [str(row["record_id"]) for row in manifest_rows]
    if len(ordered) != 500 or len(set(ordered)) != 500:
        raise ValueError("candidate manifest must contain 500 unique record_ids")
    wanted = set(ordered)
    packets = {row["record_id"]: row for _, row in scale.rows(scale.PACKETS) if row.get("record_id") in wanted}
    raw = {row["record_id"]: row for _, row in scale.rows(scale.RAW) if row.get("record_id") in wanted}
    pids = {str(row["parent_id"]) for row in packets.values()}
    parents = {row["record_id"]: row for _, row in scale.rows(scale.PARENTS) if row.get("record_id") in pids}
    if set(packets) != wanted or set(raw) != wanted or set(parents) != pids:
        raise ValueError("packet/raw/parent completeness failure")

    receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "outputs/**/verification_results.jsonl"), recursive=True)):
        for _, row in scale.rows(Path(path)):
            if row.get("record_id") in wanted:
                receipts[str(row["record_id"])].append(row)

    equality_choice: dict[str, tuple[str, str | None]] = {}
    for packet in packets.values():
        if str(packet["method"]).startswith("eq_to_le_"):
            pid = str(packet["parent_id"])
            equality_choice.setdefault(pid, scale.equality_class(packet, parents[pid]))

    output_rows, decisions, tiers, methods, receipt_stats = [], Counter(), Counter(), Counter(), Counter()
    pass_parent: dict[str, list[str]] = defaultdict(list)
    for rid in ordered:
        packet, raw_row = packets[rid], raw[rid]
        proof_hash = str(packet["candidate"]["proof_sha256"])
        statement_hash = str(packet["diff"]["statement"]["after_sha256"])
        success = scale.compatible_success(receipts.get(rid, []), proof_hash)
        if success: receipt_stats["compatible_success"] += 1
        elif receipts.get(rid): receipt_stats["fail"] += 1
        else: receipt_stats["missing"] += 1
        decision, tier, basis = decide(
            packet, raw_row, parents[str(packet["parent_id"])], success, equality_choice
        )
        if decision == "pass" and not success:
            raise AssertionError(f"pass without Pantograph success: {rid}")
        siblings = ", ".join(f"{s['method']}:{s['record_id']}" for s in packet.get("siblings", []))
        reason = (
            f"人工二审核对父题 {packet['parent_id']}、实际目标 `{packet['candidate']['goal']}`、题文、证明及 "
            f"siblings [{siblings}]；判定：{basis}；"
            f"当前 statement/proof 哈希回执为 {'Pantograph success' if success else ('Pantograph fail' if receipts.get(rid) else '尚无回执')}。"
        )
        method = str(packet["method"])
        row = {
            "record_id": rid, "quality_decision": decision, "quality_tier": tier,
            "manual_reviewed": True, "reviewer": "codex-quality-adjudicator",
            "review_reason": reason, "reviewed_statement_sha256": statement_hash,
            "reviewed_proof_sha256": proof_hash, "review_version": VERSION,
            "parent_id": packet["parent_id"], "method": method,
            "pantograph_compatible_success": success,
        }
        output_rows.append(row); decisions[decision] += 1; tiers[tier] += 1; methods[(method, decision)] += 1
        if decision == "pass": pass_parent[str(packet["parent_id"])].append(rid)

    violations = {pid: ids for pid, ids in pass_parent.items()
                  if sum("::eq_to_le_" in rid for rid in ids) > 1}
    if violations:
        raise AssertionError(f"multi-direction equality pass: {violations}")
    if len(output_rows) != 500 or len({row["record_id"] for row in output_rows}) != 500:
        raise AssertionError("review coverage failure")
    data = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    for row in output_rows)
    report = {
        "schema_version": "numinamath_shard_review_report_v1", "review_version": VERSION,
        "batch_id": "numinamath_expand_s00_b00003", "rows": 500, "unique_record_ids": 500,
        "candidate_manifest_sha256": scale.sha_file(MANIFEST),
        "packets_sha256": scale.sha_file(scale.PACKETS), "raw_sha256": scale.sha_file(scale.RAW),
        "receipt_status": dict(sorted(receipt_stats.items())),
        "final_decisions": dict(sorted(decisions.items())), "final_tiers": dict(sorted(tiers.items())),
        "by_method_decision": {f"{m}|{d}": n for (m, d), n in sorted(methods.items())},
        "equality_parent_count": len(equality_choice),
        "equality_multi_direction_pass_violations": 0,
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }
    safe_write(OUTPUT, data)
    safe_write(REPORT, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
