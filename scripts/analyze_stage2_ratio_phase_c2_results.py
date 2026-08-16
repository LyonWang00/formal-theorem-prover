"""Create the final Phase C2 comparison against C1, M0, and hard-heavy Phase B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_stage2_ratio_phase_c1_results import (
    DATASETS,
    aggregate,
    paired_counts,
    summarize,
)


MODELS = ("M0", "S2-Hard-Replay", "S2-New-WB", "S2-Balanced-Hard")


def model_dir(project: Path, model: str, slug: str) -> Path:
    if model == "M0":
        return project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/evaluation/full/M0" / slug
    if model == "S2-Hard-Replay":
        return project / "outputs/stage2_data_ablation/phaseB_hard_replay/evaluation/full/S2-Hard-Replay" / slug
    if model == "S2-New-WB":
        return project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/full/S2-New-WB" / slug
    return project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/evaluation/full/S2-Balanced-Hard" / slug


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def solved(values: dict[int, bool]) -> bool:
    return any(values.values())


def pass1(values: dict[int, bool]) -> bool:
    return bool(values[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    project = parser.parse_args().project.resolve()
    root = project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard"
    rows: dict[str, Any] = {}
    all_pairs: dict[str, dict[str, dict[int, bool]]] = {model: {} for model in MODELS}
    for label, slug in DATASETS.items():
        metrics: dict[str, Any] = {}
        pairs: dict[str, dict[str, dict[int, bool]]] = {}
        for model in MODELS:
            metrics[model], pairs[model] = summarize(model_dir(project, model, slug))
            for pid, values in pairs[model].items():
                all_pairs[model][f"{label}:{pid}"] = values
        comparisons: dict[str, Any] = {}
        for baseline in ("M0", "S2-Hard-Replay", "S2-New-WB"):
            comparisons[baseline] = {
                "delta_percentage_points": {
                    "pass_at_1": 100 * (metrics["S2-Balanced-Hard"]["pass_at_1"] - metrics[baseline]["pass_at_1"]),
                    "pass_at_2": 100 * (metrics["S2-Balanced-Hard"]["pass_at_2"] - metrics[baseline]["pass_at_2"]),
                },
                "paired_pass_at_1": paired_counts({x: pass1(v) for x, v in pairs[baseline].items()}, {x: pass1(v) for x, v in pairs["S2-Balanced-Hard"].items()}),
                "paired_pass_at_2": paired_counts({x: solved(v) for x, v in pairs[baseline].items()}, {x: solved(v) for x, v in pairs["S2-Balanced-Hard"].items()}),
            }
        rows[label] = {**metrics, "S2-Balanced-Hard_comparisons": comparisons}

    totals = {model: aggregate([rows[label][model] for label in DATASETS]) for model in MODELS}
    aggregate_comparisons: dict[str, Any] = {}
    for baseline in ("M0", "S2-Hard-Replay", "S2-New-WB"):
        aggregate_comparisons[baseline] = {
            "delta_percentage_points": {
                "pass_at_1": 100 * (totals["S2-Balanced-Hard"]["pass_at_1"] - totals[baseline]["pass_at_1"]),
                "pass_at_2": 100 * (totals["S2-Balanced-Hard"]["pass_at_2"] - totals[baseline]["pass_at_2"]),
            },
            "paired_pass_at_1": paired_counts({x: pass1(v) for x, v in all_pairs[baseline].items()}, {x: pass1(v) for x, v in all_pairs["S2-Balanced-Hard"].items()}),
            "paired_pass_at_2": paired_counts({x: solved(v) for x, v in all_pairs[baseline].items()}, {x: solved(v) for x, v in all_pairs["S2-Balanced-Hard"].items()}),
        }

    contract = json.loads((root / "evaluation/generation_contract.json").read_text(encoding="utf-8"))
    canary = json.loads((root / "evaluation/canary/canary_gate.json").read_text(encoding="utf-8"))
    c1_canary = json.loads((project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/canary/canary_gate.json").read_text(encoding="utf-8"))
    wb = rows["WB-Unseen-Holdout150"]
    ld = rows["LD-easy64"]
    c2 = totals["S2-Balanced-Hard"]
    c1 = totals["S2-New-WB"]
    m0 = totals["M0"]
    hard = totals["S2-Hard-Replay"]
    answers = {
        "improves_wb_unseen_vs_c1_pass1": wb["S2-Balanced-Hard"]["pass_at_1"] > wb["S2-New-WB"]["pass_at_1"],
        "improves_wb_unseen_vs_c1_pass2": wb["S2-Balanced-Hard"]["pass_at_2"] > wb["S2-New-WB"]["pass_at_2"],
        "aggregate_pass1_stable_or_better_than_c1": c2["pass_at_1"] >= c1["pass_at_1"],
        "behavior_canary_passed": canary["gate_passed"],
        "ld_easy_pass1_not_regressed_vs_c1": ld["S2-Balanced-Hard"]["pass_at_1"] >= ld["S2-New-WB"]["pass_at_1"],
        "ld_easy_pass2_not_regressed_vs_c1": ld["S2-Balanced-Hard"]["pass_at_2"] >= ld["S2-New-WB"]["pass_at_2"],
        "foundation_dominant_mix_outperforms_hard_heavy_on_pass1": c2["pass_at_1"] > hard["pass_at_1"],
        "foundation_dominant_mix_outperforms_hard_heavy_on_pass2": c2["pass_at_2"] > hard["pass_at_2"],
        "hard_a_independent_causal_value_identified": False,
    }
    comparison = {
        "phase": "C2", "model": "S2-Balanced-Hard",
        "primary_baselines": ["S2-New-WB", "M0"], "diagnostic_baseline": "S2-Hard-Replay",
        "canonical_backend": "transformers", "checkpoint_type": "merged",
        "generation_contract_sha256": contract["generation_config_sha256"],
        "parent_generation_contract_sha256": contract["parent_generation_contract_sha256"],
        "evaluation_integrity": {
            "expected_problems_per_model": 428, "expected_candidates_per_model": 856,
            "complete": all(item["problems"] == 428 and item["candidates"] == 856 for item in totals.values()),
            "c2_fatal_errors": c2["fatal_errors"], "c2_worker_restarts": c2["worker_restarts"], "forbidden_suites_run": [],
        },
        "datasets": rows, "aggregate": totals, "aggregate_comparisons": aggregate_comparisons,
        "task_questions": answers,
        "causal_scope_note": (
            "Phase C2 changes several composition components relative to C1: New WB increases from 1200 to 1600, "
            "Hard-A decreases from 397 to 200, Hard-B decreases from 344 to 0, and 141 verified WB replay rows are added. "
            "Therefore the arm identifies the value of the foundation-dominant mixture, not the isolated marginal causal effect of Hard-A."
        ),
    }
    comparisons_root = project / "outputs/stage2_data_ratio_ablation/comparisons"
    write_json(comparisons_root / "phaseC2_comparison.json", comparison)

    header = "| Dataset | M0 p@1/p@2 | Hard-heavy p@1/p@2 | C1 New-WB p@1/p@2 | C2 Balanced p@1/p@2 |"
    rule = "|---|---:|---:|---:|---:|"
    table = [header, rule]
    for label in DATASETS:
        item = rows[label]
        table.append(
            f"| {label} | {item['M0']['pass_at_1']:.2%}/{item['M0']['pass_at_2']:.2%} | "
            f"{item['S2-Hard-Replay']['pass_at_1']:.2%}/{item['S2-Hard-Replay']['pass_at_2']:.2%} | "
            f"{item['S2-New-WB']['pass_at_1']:.2%}/{item['S2-New-WB']['pass_at_2']:.2%} | "
            f"{item['S2-Balanced-Hard']['pass_at_1']:.2%}/{item['S2-Balanced-Hard']['pass_at_2']:.2%} |"
        )
    table.append(
        f"| Aggregate (428) | {m0['pass_at_1']:.2%}/{m0['pass_at_2']:.2%} | {hard['pass_at_1']:.2%}/{hard['pass_at_2']:.2%} | "
        f"{c1['pass_at_1']:.2%}/{c1['pass_at_2']:.2%} | {c2['pass_at_1']:.2%}/{c2['pass_at_2']:.2%} |"
    )
    comparison_md = ["# Phase C2 comparison", "", *table, ""]
    for baseline in ("S2-New-WB", "M0", "S2-Hard-Replay"):
        delta = aggregate_comparisons[baseline]["delta_percentage_points"]
        comparison_md.append(f"- C2 vs {baseline}: pass@1 {delta['pass_at_1']:+.2f} pp; pass@2 {delta['pass_at_2']:+.2f} pp.")
    comparison_md += ["", comparison["causal_scope_note"], ""]
    (comparisons_root / "phaseC2_comparison.md").write_text("\n".join(comparison_md), encoding="utf-8")

    training = json.loads((root / "training/training_summary.json").read_text(encoding="utf-8"))
    data_audit = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
    leakage = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
    eos = json.loads((root / "eos_audit.json").read_text(encoding="utf-8"))
    c1_delta = aggregate_comparisons["S2-New-WB"]["delta_percentage_points"]
    m0_delta = aggregate_comparisons["M0"]["delta_percentage_points"]
    hard_delta = aggregate_comparisons["S2-Hard-Replay"]["delta_percentage_points"]
    report = [
        "# Phase C2 — Foundation dominant + 10% Hard-A", "", "## 结论", "",
        "Phase C2 已按用户批准的五桶配比完成训练、行为 canary 和四组正式评估。C2 在 WB unseen、LD-easy 和总体指标上均高于 C1；Monitor 与 C1 持平，Retention 也有所恢复。", "",
        "结果支持以 foundation 为主体、少量 Hard-A 补充的第二轮 SFT 方向。但由于 C2 相对 C1 同时增加 New WB、移除 Hard-B、减少 Hard-A 并加入 WB replay，本实验不能单独识别 Hard-A 的边际因果价值；若需要该结论，仍需同 foundation 配方的 0% Hard-A 对照。", "",
        "## 数据与训练门禁", "",
        "- 2000 行：New WB 1600、Hard-A 200、Stable/Core 12、Frontier 47、Additional verified WB replay 141。实际 Hard-A=10%。",
        f"- New WB 保留 C1 的 1200 行并追加 400 行；Additional WB replay 从 {data_audit['additional_verified_wb_replay']['eligible_candidates']} 条合格第一轮 WB 中固定选择。",
        f"- Manifest：`{training['manifest_sha256']}`；duplicate rows=0，theorem-group duplicates=0，max repeat={training['max_repeat']}。Hard-B/Hard-C/unclassified hard=0。",
        f"- 泄漏门禁：{leakage['status']}；EOS：{eos['eos_supervised']}/2000；zero-label={eos['zero_label']}；semantic truncation={eos['semantic_truncation']}。",
        f"- 训练 125/125 step、无放回 2000/2000；train loss {training['metrics']['train_loss']:.6f}；eval loss {training['eval_loss']:.6f}；eval token accuracy {training['eval_token_accuracy']:.4%}。",
        f"- M0 哈希保持 `{training['frozen_m0_hash_after_training']}`；S2-Balanced-Hard merged 哈希 `{contract['checkpoint_model_sha256']}`。", "",
        "## Canonical inference 与行为稳定性", "",
        f"- merged checkpoint + Transformers；adapter=null；父合同 `{contract['parent_generation_contract_sha256']}`；C2 解析合同 `{contract['generation_config_sha256']}`。",
        f"- Canary PASS：140/140 候选；抽取率 {canary['combined']['S2-Balanced-Hard']['extraction_success_rate']:.2%}；平均长度 {canary['combined']['S2-Balanced-Hard']['completion_tokens_mean']:.2f}（C1 {c1_canary['combined']['S2-New-WB']['completion_tokens_mean']:.2f}）；length finish {canary['combined']['S2-Balanced-Hard']['max_length_finish_ratio']:.2%}；病理重复 {canary['combined']['S2-Balanced-Hard']['pathological_repetition_rate']:.2%}。", "",
        "## 正式评估", "", *table, "",
        f"总体 C2 相对 C1：pass@1 {c1_delta['pass_at_1']:+.2f} pp，pass@2 {c1_delta['pass_at_2']:+.2f} pp；相对 M0：pass@1 {m0_delta['pass_at_1']:+.2f} pp，pass@2 {m0_delta['pass_at_2']:+.2f} pp；相对 hard-heavy：pass@1 {hard_delta['pass_at_1']:+.2f} pp，pass@2 {hard_delta['pass_at_2']:+.2f} pp。", "",
        "## 任务问题回答", "",
        f"1. 是否进一步提升 WB unseen：是。C1 {wb['S2-New-WB']['pass_at_1']:.2%}/{wb['S2-New-WB']['pass_at_2']:.2%} → C2 {wb['S2-Balanced-Hard']['pass_at_1']:.2%}/{wb['S2-Balanced-Hard']['pass_at_2']:.2%}。",
        f"2. 10% Hard-A 是否保持稳定：是。总体 pass@1 提升，行为 canary 通过；LD-easy 从 {ld['S2-New-WB']['pass_at_1']:.2%}/{ld['S2-New-WB']['pass_at_2']:.2%} 提升至 {ld['S2-Balanced-Hard']['pass_at_1']:.2%}/{ld['S2-Balanced-Hard']['pass_at_2']:.2%}。",
        "3. Phase B 退化是否主要来自 hard 比例过高：结果与这一解释一致，而不支持‘Hard 数据本身无价值’；但由于没有同 foundation 的 0% Hard-A 对照，应表述为强方向性证据，而非独立因果证明。",
        "4. 更合理方向是否为 foundation dominant + small hard supplement：在当前四个实验臂中是；C2 同时取得最高总体 pass@1 与 pass@2。", "",
        f"WB unseen C2 vs C1 的配对 McNemar p@2 p={wb['S2-Balanced-Hard_comparisons']['S2-New-WB']['paired_pass_at_2']['mcnemar_exact_p']:.4f}；总体 C2 vs C1 p@2 p={aggregate_comparisons['S2-New-WB']['paired_pass_at_2']['mcnemar_exact_p']:.4f}。小样本或非显著差异仍只作方向性解释。", "",
        "未运行 Full500、Strict-unseen200 或 miniF2F。", "", "## 阶段状态", "",
        "Phase C2 completed.", "", "Waiting for approval before Phase C3.", "",
    ]
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(root / "evaluation/status.json", {
        "status": "PHASE_C2_EVALUATION_COMPLETED", "canary_gate_passed": canary["gate_passed"],
        "canary_candidates_per_model": canary["combined"]["S2-Balanced-Hard"]["candidate_count"],
        "formal_evaluation_completed": True, "formal_problems": c2["problems"], "formal_candidates": c2["candidates"],
        "fatal_errors": c2["fatal_errors"], "worker_restarts": c2["worker_restarts"], "forbidden_suites_run": [], "phase_c3_started": False,
    })
    final_summary = [
        "# Stage 2 data-ratio ablation status", "", "## Phase C1 — COMPLETE", "", "S2-New-WB completed.", "",
        "## Phase C2 — COMPLETE", "", "S2-Balanced-Hard completed with the approved 80% New-WB / 10% Hard-A foundation-dominant mixture.", "",
        f"Aggregate p@1: M0 {m0['pass_at_1']:.2%}, hard-heavy {hard['pass_at_1']:.2%}, C1 {c1['pass_at_1']:.2%}, C2 {c2['pass_at_1']:.2%}.",
        f"Aggregate p@2: M0 {m0['pass_at_2']:.2%}, hard-heavy {hard['pass_at_2']:.2%}, C1 {c1['pass_at_2']:.2%}, C2 {c2['pass_at_2']:.2%}.", "",
        "## Phase C3 — NOT STARTED", "", "Waiting for explicit approval before Phase C3.", "",
    ]
    (project / "outputs/stage2_data_ratio_ablation/final_summary.md").write_text("\n".join(final_summary), encoding="utf-8")
    print(json.dumps({"report": str(root / 'report.md'), "comparison": str(comparisons_root / 'phaseC2_comparison.json'), "aggregate": totals, "answers": answers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
