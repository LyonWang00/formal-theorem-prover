#!/usr/bin/env python3
"""Validate Codex judgments and materialize the expanded LD-easy pool.

The utility never infers or changes a label.  Missing batches remain unreviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.prepare_initial_anchor_ld_expansion import (
    identities,
    normalized_statement,
    protected_index,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)
from scripts.prepare_ld_difficulty_review import default_timeout_verified


DIFFICULTIES = {"easy", "medium", "difficult", "uncertain", "exclude"}
CONFIDENCES = {"high", "medium", "low"}
EVIDENCE_FIELDS = {
    "statement_complexity",
    "proof_structure",
    "tactic_complexity",
    "premise_dependency",
    "context_dependency",
    "fit_for_subgoal_whole_proof",
}
JUDGMENT_FIELDS = {
    "difficulty",
    "confidence",
    "reason_summary",
    "evidence",
    "trainable_for_short_whole_proof",
}
REVIEW_CODE_FIELDS = {"sample_id", "review_code", "note"}
REVIEW_CODES: dict[str, dict[str, Any]] = {
    "E_RFL": {
        "difficulty": "easy",
        "confidence": "high",
        "trainable_for_short_whole_proof": True,
        "reason_summary": "人工核对后确认目标与结论是直接的定义性等式，完整证明仅需 rfl 或等价的定义展开。",
        "evidence": {
            "statement_complexity": "命题虽可含抽象类型，但结论是单一直接等式。",
            "proof_structure": "证明为无分支的定义性归约，可直接闭合目标。",
            "tactic_complexity": "仅需 rfl 或等价的最小定义展开步骤。",
            "premise_dependency": "不依赖需要检索的专用外部证明引理。",
            "context_dependency": "从命题暴露的定义与当前导入即可复现。",
            "fit_for_subgoal_whole_proof": "非常适合作为短完整证明的训练样本。",
        },
    },
    "E_SIMP": {
        "difficulty": "easy",
        "confidence": "high",
        "trainable_for_short_whole_proof": True,
        "reason_summary": "人工核对后确认目标可由标准 simp、decide、norm_cast 或单个常用自动化步骤直接解决。",
        "evidence": {
            "statement_complexity": "目标结构清晰，化简方向可从结论直接识别。",
            "proof_structure": "证明无实质分支，单次标准化简即可完成。",
            "tactic_complexity": "只使用常见且稳定的基础自动化策略。",
            "premise_dependency": "无需猜测隐藏的专用同文件证明链条。",
            "context_dependency": "不要求额外局部记号或未暴露上下文知识。",
            "fit_for_subgoal_whole_proof": "适合 statement 到短完整证明的直接学习。",
        },
    },
    "E_HYP": {
        "difficulty": "easy",
        "confidence": "high",
        "trainable_for_short_whole_proof": True,
        "reason_summary": "人工核对后确认结论是已有假设、字段、投影或其直接函数应用，证明路径显式且唯一。",
        "evidence": {
            "statement_complexity": "命题的关键证明对象已经显式出现在假设中。",
            "proof_structure": "证明直接返回假设、投影或一次函数应用。",
            "tactic_complexity": "不需要搜索、分情况或多步自动化策略。",
            "premise_dependency": "核心依赖由命题参数直接提供而非隐藏前提。",
            "context_dependency": "仅使用当前声明中显式可见的对象和字段。",
            "fit_for_subgoal_whole_proof": "非常适合学习短 term 风格完整证明。",
        },
    },
    "E_RW": {
        "difficulty": "easy",
        "confidence": "medium",
        "trainable_for_short_whole_proof": True,
        "reason_summary": "人工核对后确认只需一到两个显然方向的重写或常用基础引理，证明短且无复杂搜索。",
        "evidence": {
            "statement_complexity": "命题关系明确，主要变化是直接等式替换。",
            "proof_structure": "证明是短线性的重写链，没有分支子目标。",
            "tactic_complexity": "使用 rw、simpa、exact 等基础组合步骤。",
            "premise_dependency": "所需引理是命题中可见或高度常用的全局事实。",
            "context_dependency": "不依赖强同文件上下文或特殊局部记号。",
            "fit_for_subgoal_whole_proof": "适合生成长度受限的完整短证明。",
        },
    },
    "E_SHORT": {
        "difficulty": "easy",
        "confidence": "medium",
        "trainable_for_short_whole_proof": True,
        "reason_summary": "人工核对后确认完整证明仅含少量常见步骤，结构局部、可预测且适合短 whole-proof 训练。",
        "evidence": {
            "statement_complexity": "命题存在一定抽象符号但目标结构仍然局部清楚。",
            "proof_structure": "证明至多包含少量线性步骤且没有深层嵌套。",
            "tactic_complexity": "使用常见 tactic 或短 term 组合即可结束。",
            "premise_dependency": "依赖可从命题和通用库接口合理获得。",
            "context_dependency": "未发现必须依赖源文件邻近声明的强上下文。",
            "fit_for_subgoal_whole_proof": "仍适合作为完整短证明训练数据使用。",
        },
    },
    "M_LOCAL": {
        "difficulty": "medium",
        "confidence": "high",
        "trainable_for_short_whole_proof": False,
        "reason_summary": "人工核对后发现证明虽短但关键步骤依赖特定同文件引理或命名接口，statement-only 输入难以恢复。",
        "evidence": {
            "statement_complexity": "表面目标不长，但语义依赖专用库结构。",
            "proof_structure": "证明核心是调用特定同文件或邻近声明。",
            "tactic_complexity": "策略本身简单，但精确引理检索难度明显。",
            "premise_dependency": "存在决定性且不由命题直接暴露的局部前提。",
            "context_dependency": "需要较强源文件上下文才能可靠重建证明。",
            "fit_for_subgoal_whole_proof": "不宜纳入当前 statement-only 的 LD-easy 训练池。",
        },
    },
    "M_COMPLEX": {
        "difficulty": "medium",
        "confidence": "high",
        "trainable_for_short_whole_proof": False,
        "reason_summary": "人工核对后发现命题结构、类型层级或代数表达较复杂，短参考证明不足以说明生成任务本身容易。",
        "evidence": {
            "statement_complexity": "命题含多层结构、复杂表达或较多隐式关系。",
            "proof_structure": "参考证明短但依赖对复杂目标形态的准确识别。",
            "tactic_complexity": "需要较强自动化选择或非显然的规范化入口。",
            "premise_dependency": "可能依赖领域专用知识而非基础通用事实。",
            "context_dependency": "仅凭 statement 重建证明存在明显不确定性。",
            "fit_for_subgoal_whole_proof": "不符合当前严格 LD-easy 的可训练标准。",
        },
    },
    "D_HARD": {
        "difficulty": "difficult",
        "confidence": "high",
        "trainable_for_short_whole_proof": False,
        "reason_summary": "人工核对后确认该题需要复杂推导、多个非显然步骤或强领域知识，不属于短完整证明 easy 范围。",
        "evidence": {
            "statement_complexity": "命题具有明显复杂的数学或类型论结构。",
            "proof_structure": "证明需要多步推导、分支或关键构造。",
            "tactic_complexity": "包含难以直接预测的高级策略选择。",
            "premise_dependency": "需要多个领域专用前提或复杂检索。",
            "context_dependency": "离开完整源上下文后难以可靠生成。",
            "fit_for_subgoal_whole_proof": "不适合当前短 whole-proof easy 训练协议。",
        },
    },
    "X_DUP": {
        "difficulty": "exclude",
        "confidence": "high",
        "trainable_for_short_whole_proof": False,
        "reason_summary": "Manual audit found that this sample belongs to a theorem group already represented in the trainable easy pool, so it is excluded to preserve theorem-group deduplication.",
        "evidence": {
            "statement_complexity": "The statement itself may be easy, but it duplicates an already retained theorem group.",
            "proof_structure": "Proof difficulty is not used because theorem-group duplication is a hard exclusion criterion.",
            "tactic_complexity": "Tactic simplicity does not override the deduplication gate.",
            "premise_dependency": "Premise usage is not the reason for exclusion.",
            "context_dependency": "The sample is excluded because its normalized theorem identity is already represented.",
            "fit_for_subgoal_whole_proof": "Not eligible as an additional training row because it would duplicate a retained theorem group.",
        },
    },
    "X_CONTEXT": {
        "difficulty": "exclude",
        "confidence": "high",
        "trainable_for_short_whole_proof": False,
        "reason_summary": "人工核对后确认样本依赖私有声明、局部宏记号或无法从训练输入获得的上下文，因此从训练池排除。",
        "evidence": {
            "statement_complexity": "命题包含仅在源文件局部可解释的构造。",
            "proof_structure": "证明依赖私有实现或局部语法环境。",
            "tactic_complexity": "离开原文件后策略含义或行为不可稳定复现。",
            "premise_dependency": "关键依赖未作为可访问的全局前提提供。",
            "context_dependency": "存在强局部宏、私有声明或隐藏上下文依赖。",
            "fit_for_subgoal_whole_proof": "不满足当前统一训练输入协议，必须排除。",
        },
    },
}


def annotation(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("difficulty_annotation")
    return value if isinstance(value, dict) else {}


def validate_judgment(sample_id: str, row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if "review_code" in row:
        if set(row) - REVIEW_CODE_FIELDS:
            problems.append("unexpected compact judgment fields")
        if row.get("review_code") not in REVIEW_CODES:
            problems.append("invalid review_code")
        if not isinstance(row.get("note", ""), str):
            problems.append("compact judgment note must be a string")
        return [f"{sample_id}: {problem}" for problem in problems]
    if set(row) - JUDGMENT_FIELDS - {"sample_id"}:
        problems.append("unexpected judgment fields")
    if JUDGMENT_FIELDS - set(row):
        problems.append("missing judgment fields")
    if row.get("difficulty") not in DIFFICULTIES:
        problems.append("invalid difficulty")
    if row.get("confidence") not in CONFIDENCES:
        problems.append("invalid confidence")
    if len(str(row.get("reason_summary") or "").strip()) < 40:
        problems.append("reason_summary too short")
    evidence = row.get("evidence") or {}
    if set(evidence) != EVIDENCE_FIELDS:
        problems.append("evidence fields incomplete")
    for field in EVIDENCE_FIELDS:
        if len(str(evidence.get(field) or "").strip()) < 15:
            problems.append(f"evidence.{field} too short")
    trainable = row.get("trainable_for_short_whole_proof")
    if not isinstance(trainable, bool):
        problems.append("trainability must be explicit bool")
    if trainable and row.get("difficulty") != "easy":
        problems.append("only easy may be trainable")
    return [f"{sample_id}: {problem}" for problem in problems]


def expand_judgment(row: dict[str, Any]) -> dict[str, Any]:
    if "review_code" not in row:
        return {field: row[field] for field in JUDGMENT_FIELDS if field in row}
    expanded = json.loads(json.dumps(REVIEW_CODES[str(row["review_code"])]))
    expanded["manual_review_code"] = str(row["review_code"])
    note = str(row.get("note") or "").strip()
    if note:
        expanded["manual_note"] = note
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion"
        ),
    )
    parser.add_argument(
        "--classification-version", default="ld-manual-anchor-expansion-v1"
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    pool_path = root / "frozen_expansion_pool.jsonl"
    pool = read_jsonl(pool_path)
    pool_by_id = {str(row["id"]): row for row in pool}
    labels: list[dict[str, Any]] = []
    problems: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    completed_batches = 0
    for template_path in sorted((root / "label_templates").glob("batch_*.jsonl")):
        judgment_path = root / "manual_judgments" / template_path.name
        if not judgment_path.is_file():
            continue
        templates = read_jsonl(template_path)
        judgments = read_jsonl(judgment_path)
        by_id = {str(row.get("sample_id") or ""): row for row in judgments}
        expected = {str(row["sample_id"]) for row in templates}
        if len(by_id) != len(judgments) or set(by_id) != expected:
            problems.append(
                f"{judgment_path.name}: expected exactly {len(expected)} unique IDs"
            )
            continue
        completed_batches += 1
        for template in templates:
            sample_id = str(template["sample_id"])
            judgment = by_id[sample_id]
            problems.extend(validate_judgment(sample_id, judgment))
            label = {
                **template,
                **expand_judgment(judgment),
                "reviewed_by": "codex",
                "classification_version": args.classification_version,
                "classified_at": timestamp,
            }
            labels.append(label)
    if problems:
        write_json(root / "label_validation_errors.json", problems)
        raise ValueError(f"{len(problems)} label validation errors")
    label_ids = [str(row["sample_id"]) for row in labels]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("duplicate expansion labels")
    old_labels_path = (
        project
        / "outputs/ld_length_difficulty_pipeline/difficulty/difficulty_labels.jsonl"
    )
    old_label_ids = {str(row["sample_id"]) for row in read_jsonl(old_labels_path)}
    if set(label_ids) & old_label_ids:
        raise ValueError("expansion labels overlap the original frozen review pool")
    write_jsonl(root / "expansion_labels.jsonl", labels)

    protected_ids, protected_statements, protected_inventory = protected_index(project)
    old_easy_path = (
        project / "outputs/ld_length_difficulty_pipeline/difficulty/easy_pool.jsonl"
    )
    old_easy = read_jsonl(old_easy_path)
    selected: list[dict[str, Any]] = []
    excluded = Counter()
    for row in old_easy:
        label = annotation(row)
        if not (
            label.get("difficulty") == "easy"
            and label.get("confidence") in {"high", "medium"}
            and label.get("trainable_for_short_whole_proof") is True
        ):
            excluded["old_not_trainable"] += 1
            continue
        if identities(row) & protected_ids or normalized_statement(row) in protected_statements:
            excluded["old_protected_overlap"] += 1
            continue
        if not default_timeout_verified(row):
            excluded["old_verification_invalid"] += 1
            continue
        selected.append(row)
    for label in labels:
        if not (
            label.get("difficulty") == "easy"
            and label.get("confidence") in {"high", "medium"}
            and label.get("trainable_for_short_whole_proof") is True
        ):
            excluded["new_not_trainable_easy"] += 1
            continue
        source = json.loads(json.dumps(pool_by_id[str(label["sample_id"])]))
        if identities(source) & protected_ids or normalized_statement(source) in protected_statements:
            raise ValueError(f"protected leakage in expansion label {label['sample_id']}")
        if not default_timeout_verified(source):
            raise ValueError(f"invalid verification receipt {label['sample_id']}")
        source["difficulty_annotation"] = label
        selected.append(source)
    groups = [str(row.get("theorem_group_id") or row.get("id")) for row in selected]
    if len(groups) != len(set(groups)):
        raise ValueError("expanded trainable easy pool has theorem-group duplicates")
    write_jsonl(root / "trainable_easy_pool.jsonl", selected)
    new_counts = Counter(str(row.get("difficulty")) for row in labels)
    summary = {
        "classification_version": args.classification_version,
        "automatic_difficulty_labels": 0,
        "frozen_pool_sha256": sha256(pool_path),
        "completed_batches": completed_batches,
        "reviewed_new_rows": len(labels),
        "new_difficulty_counts": dict(new_counts),
        "old_trainable_nonprotected_easy": len(selected)
        - sum(
            row.get("difficulty") == "easy"
            and row.get("confidence") in {"high", "medium"}
            and row.get("trainable_for_short_whole_proof") is True
            for row in labels
        ),
        "new_trainable_easy": sum(
            row.get("difficulty") == "easy"
            and row.get("confidence") in {"high", "medium"}
            and row.get("trainable_for_short_whole_proof") is True
            for row in labels
        ),
        "total_trainable_nonprotected_easy": len(selected),
        "arm_gates": {
            "A5_250": len(selected) >= 250,
            "A10_500": len(selected) >= 500,
            "A20_1000": len(selected) >= 1000,
        },
        "excluded": dict(excluded),
        "protected_sets": protected_inventory,
        "hard_evaluation_leaks": 0,
        "trainable_easy_pool_sha256": sha256(root / "trainable_easy_pool.jsonl"),
    }
    write_json(root / "expansion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
