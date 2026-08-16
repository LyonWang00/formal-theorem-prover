#!/usr/bin/env python3
"""Classify frozen EI discovery results by current-model dynamic difficulty."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.expert_iteration.discovery_scheduler import (
    DiscoveryScheduler,
)
from lean_prover.lean_training.expert_iteration.dynamic_difficulty import (
    build_statistics,
    classify_discovery,
    iter_jsonl,
    sha256_file,
)
from lean_prover.lean_training.expert_iteration.dynamic_difficulty_sampler import (
    sample_by_difficulty,
)


DEFAULT_MANIFEST = (
    "outputs/expert_iteration/round0/discovery/discovery_manifest.jsonl"
)
DEFAULT_CANDIDATES = (
    "outputs/expert_iteration/round0/discovery/candidate_results.jsonl"
)
DEFAULT_OUTPUT = "outputs/expert_iteration/round0/dynamic_difficulty"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def distribution_text(distribution: Mapping[str, int]) -> str:
    return ", ".join(
        f"{name}={count}" for name, count in sorted(distribution.items())
    ) or "none"


def report_markdown(stats: Mapping[str, Any]) -> str:
    overall = stats["overall"]
    source_lines = [
        "| Source | Theorems | Candidate success | Mean theorem success | Difficulty |",
        "|---|---:|---:|---:|---|",
    ]
    for source, row in stats["by_source"].items():
        source_lines.append(
            f"| {source} | {row['theorems']} | {pct(row['candidate_success_rate'])} "
            f"| {pct(row['mean_theorem_success_rate'])} "
            f"| {distribution_text(row['difficulty_distribution'])} |"
        )
    domain_lines = [
        "| Domain | Theorems | Candidate success | Difficulty |",
        "|---|---:|---:|---|",
    ]
    for domain, row in stats["by_domain"].items():
        domain_lines.append(
            f"| {domain} | {row['theorems']} | {pct(row['candidate_success_rate'])} "
            f"| {distribution_text(row['difficulty_distribution'])} |"
        )

    success_bank = stats["success_bank"]
    verified_distribution = success_bank["verified_candidate_distribution"]
    verified_total = sum(verified_distribution.values())
    mastered = verified_distribution.get("too_easy", 0) + verified_distribution.get(
        "easy", 0
    )
    mastered_share = mastered / verified_total if verified_total else 0.0
    if mastered_share >= 0.5:
        success_only_explanation = (
            f"Success Bank 中 {pct(mastered_share)} 的 verified candidates 来自 "
            "TooEasy/Easy theorem；这些样本更偏向重复当前模型已掌握行为，"
            "因此 Success-only 容易形成自蒸馏冗余，而不是推动能力边界。"
        )
    else:
        success_only_explanation = (
            f"Success Bank 中 TooEasy/Easy verified candidate 占 {pct(mastered_share)}；"
            "Success-only 的有限收益不能只归因于简单题占比，还应结合 proof diversity、"
            "样本规模和训练稳定性继续分析。"
        )
    source_note = ""
    if int(stats["by_source"].get("LD-easy", {}).get("theorems", 0)) < 20:
        source_note = (
            "本轮 Discovery 中 LD-easy 仅有少量记录，因此 WB/LD 差异只作描述性统计，"
            "不能据此作显著性结论。"
        )
    return "\n".join(
        [
            "# EI Round0 Dynamic Difficulty Report",
            "",
            "## 总体分布",
            "",
            f"- Theorems: {overall['theorems']}",
            f"- Candidates: {overall['attempts']}",
            f"- Verified candidates: {overall['successes']} ({pct(overall['candidate_success_rate'])})",
            f"- Difficulty: {distribution_text(overall['difficulty_distribution'])}",
            f"- Zero-success subtype: {distribution_text(overall['zero_success_subtypes'])}",
            "",
            "## 按 source 统计",
            "",
            *source_lines,
            "",
            source_note,
            "",
            "## 按 domain 统计",
            "",
            *domain_lines,
            "",
            "## Failure 分布",
            "",
            f"- All failures: {distribution_text(stats['failure_distribution'])}",
            "- Hard/Impossible failures: "
            f"{distribution_text(stats['hard_impossible_failure_distribution'])}",
            "",
            "## Success Bank 与 Success-only",
            "",
            f"- Success Bank contributing theorems: {success_bank['contributing_theorems']}",
            "- Statement difficulty: "
            f"{distribution_text(success_bank['statement_difficulty_distribution'])}",
            "- Verified candidates by difficulty: "
            f"{distribution_text(verified_distribution)}",
            f"- {success_only_explanation}",
            "",
            "## 下一轮 EI 建议",
            "",
            "- 主采样 Medium：最接近当前能力边界，适合 verified self-improvement。",
            "- 次采样 Hard（success_count > 0）：保留偶发成功的 latent capability。",
            "- 探索 capability-only Impossible，但不得把 environment/syntax/unknown-identifier "
            "失败当成训练 target。",
            "- Easy 仅作有限 foundation/proof-diversity replay；TooEasy 严格限量。",
            "- 调度器应按目标有效样本数和实测 yield 动态追加 Discovery，而不是固定题量比例。",
            "",
            "## 数据审计",
            "",
            f"- theorem duplicate = {stats['audit']['theorem_duplicates']}",
            f"- record duplicate = {stats['audit']['record_duplicates']}",
            f"- difficulty coverage = {pct(stats['audit']['difficulty_coverage'])}",
            "- 原 discovery、candidate results、checkpoint 和人工 difficulty 标签均未修改。",
            "- 未使用 evaluation data 或 reference proof 作为训练 target。",
            "",
            "Dynamic difficulty classification completed.",
            "",
            "Dynamic difficulty pipeline completed.",
            "",
            "No training started.",
            "",
            "Waiting for EI data selection experiment.",
            "",
        ]
    )


def sampling_smoke(records: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    cases = {
        "medium": ["medium"],
        "easy_medium": ["easy", "medium"],
        "hard": ["hard"],
    }
    payload: dict[str, Any] = {"seed": seed, "cases": {}}
    for name, categories in cases.items():
        selected = sample_by_difficulty(
            records, categories=categories, total=None, seed=seed
        )
        payload["cases"][name] = {
            "categories": categories,
            "rows": len(selected),
            "theorem_duplicates": len(selected)
            - len({str(row["theorem_id"]) for row in selected}),
            "sample_theorem_ids": [str(row["theorem_id"]) for row in selected[:10]],
        }
    payload["status"] = "PASS" if all(
        row["theorem_duplicates"] == 0 for row in payload["cases"].values()
    ) else "FAIL"
    return payload


def scheduler_smoke(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scheduler = DiscoveryScheduler({"WB": 400, "LD-easy": 200})
    scheduler.ingest(records)
    source_rows = Counter(str(row["source"]) for row in records)
    effective_rows = Counter(
        str(row["source"]) for row in records if row["effective_for_ei_default"]
    )
    yields = {
        source: max(0.01, effective_rows[source] / source_rows[source])
        for source in scheduler.target_effective_by_source
        if source_rows[source]
    }
    for source in scheduler.target_effective_by_source:
        yields.setdefault(source, 0.25)
    return {
        "state": scheduler.state_dict(),
        "observed_effective_yield": yields,
        "next_batch_plan": scheduler.plan_next_batch(yields, max_batch=500),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-results", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-seed", type=int, default=20260804)
    args = parser.parse_args()
    project = args.project.resolve()
    manifest_path = project / args.manifest
    candidates_path = project / args.candidate_results
    output = project / args.output
    manifest_rows = list(iter_jsonl(manifest_path))
    candidate_rows = list(iter_jsonl(candidates_path))
    records = classify_discovery(manifest_rows, candidate_rows)
    stats = build_statistics(
        records,
        manifest_sha256=sha256_file(manifest_path),
        candidate_results_sha256=sha256_file(candidates_path),
    )
    sampling = sampling_smoke(records, args.sampling_seed)
    scheduler = scheduler_smoke(records)
    write_jsonl(output / "dynamic_difficulty_manifest.jsonl", records)
    write_json(output / "dynamic_difficulty_statistics.json", stats)
    write_json(output / "sampling_interface_test.json", sampling)
    write_json(output / "discovery_scheduler_interface_test.json", scheduler)
    (output / "dynamic_difficulty_report.md").write_text(
        report_markdown(stats), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": stats["status"],
                "rows": len(records),
                "difficulty": stats["overall"]["difficulty_distribution"],
                "sampling_test": sampling["status"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

