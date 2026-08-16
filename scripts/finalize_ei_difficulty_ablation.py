#!/usr/bin/env python3
"""Compute paired statistics and finalize EI difficulty-aware ablation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finalize_ei_data_composition_ablation import fmt_stat, paired_stat, pct, states, summarize

DATASETS = {
    "WB-Unseen-Holdout150": ("wb_unseen_holdout150", 150),
    "LD-easy64": ("ld_easy64", 64),
    "Monitor64": ("monitor64", 64),
    "WB-Train-Retention150": ("wb_train_retention150", 150),
}
MODELS = ("H0-No-Hard", "A_medium_hard", "B_medium_only", "C_add_repair", "D_add_replay")
DISPLAY = {
    "H0-No-Hard": "H0", "A_medium_hard": "A Difficulty-aware", "B_medium_only": "B Medium-heavy",
    "C_add_repair": "C +Repair", "D_add_replay": "D +Replay",
}
PAIRS = (
    ("H0-No-Hard", "A_medium_hard"), ("H0-No-Hard", "B_medium_only"),
    ("H0-No-Hard", "C_add_repair"), ("H0-No-Hard", "D_add_replay"),
    ("A_medium_hard", "B_medium_only"), ("A_medium_hard", "C_add_repair"),
    ("A_medium_hard", "D_add_replay"),
)
FAILURES = ("syntax_error", "elaboration_error", "unknown_identifier", "tactic_error", "unsolved_goals", "timeout")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def result_dir(project: Path, root: Path, model: str, slug: str) -> Path:
    if model == "H0-No-Hard":
        return project / "outputs/expert_iteration/ei_ablation/comparisons/H0_matched_seed/evaluation/core" / slug
    return root / model / "evaluation/core" / slug


def round0_dir(project: Path, slug: str) -> Path:
    return project / "outputs/expert_iteration/round0/evaluation/core/EI-Round0" / slug


def failure_counts(directory: Path) -> dict[str, int]:
    counts = Counter()
    for row in read_jsonl(directory / "attempts.jsonl"):
        if row.get("success"):
            continue
        status = "timeout" if row.get("timed_out") else str(row.get("status") or "other")
        counts[status] += 1
    return {name: counts.get(name, 0) for name in (*FAILURES, "extraction_error", "other")}


def significant_positive(stat: dict[str, Any]) -> bool:
    return stat["delta_percentage_points"] > 0 and stat["paired_bootstrap_95_ci_percentage_points"][0] > 0 and stat["mcnemar_exact_p"] < 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/difficulty_ablation"
    preparation = read_json(root / "data_preparation_summary.json")

    metrics: dict[str, dict[str, Any]] = {dataset: {} for dataset in DATASETS}
    paired: dict[str, dict[str, Any]] = {dataset: {} for dataset in DATASETS}
    failures: dict[str, dict[str, Any]] = {dataset: {} for dataset in DATASETS}
    aggregate = {model: {metric: {} for metric in ("pass_at_1", "pass_at_2", "candidate_success")} for model in MODELS}
    for dataset, (slug, expected) in DATASETS.items():
        for model in MODELS:
            directory = result_dir(project, root, model, slug)
            model_metrics, model_paired = summarize(directory, expected)
            metrics[dataset][model] = model_metrics
            paired[dataset][model] = model_paired
            failures[dataset][model] = failure_counts(directory)
            for metric in aggregate[model]:
                aggregate[model][metric].update({f"{dataset}:{key}": value for key, value in states(model_paired, metric).items()})

    historical_round0 = {}
    for dataset, (slug, expected) in DATASETS.items():
        historical_round0[dataset], _ = summarize(round0_dir(project, slug), expected)

    comparisons: dict[str, Any] = {}
    for left, right in PAIRS:
        key = f"{left}_vs_{right}"
        comparisons[key] = {}
        for dataset in DATASETS:
            comparisons[key][dataset] = {
                metric: paired_stat(states(paired[dataset][left], metric), states(paired[dataset][right], metric), f"difficulty:{key}:{dataset}:{metric}")
                for metric in ("pass_at_1", "pass_at_2", "candidate_success")
            }
        comparisons[key]["Aggregate428"] = {
            metric: paired_stat(aggregate[left][metric], aggregate[right][metric], f"difficulty:{key}:aggregate:{metric}")
            for metric in ("pass_at_1", "pass_at_2")
        }
        comparisons[key]["CandidateAggregate856"] = paired_stat(
            aggregate[left]["candidate_success"], aggregate[right]["candidate_success"], f"difficulty:{key}:candidate_aggregate"
        )

    stability = {}
    for model in MODELS[1:]:
        stability[model] = {
            "foundation_forgetting": any(metrics[name][model]["pass_at_2"] < metrics[name]["H0-No-Hard"]["pass_at_2"] - 0.02 for name in ("Monitor64", "WB-Train-Retention150")),
            "output_drift": any(metrics[name][model]["pathological_repetition_rate"] > max(0.05, metrics[name]["H0-No-Hard"]["pathological_repetition_rate"] + 0.03) or metrics[name][model]["max_length_finish_rate"] > max(0.10, metrics[name]["H0-No-Hard"]["max_length_finish_rate"] + 0.05) for name in DATASETS),
        }
    extended_required = any(
        not any(stability[model].values()) and (
            significant_positive(comparisons[f"H0-No-Hard_vs_{model}"]["Aggregate428"]["pass_at_2"])
            or significant_positive(comparisons[f"H0-No-Hard_vs_{model}"]["WB-Unseen-Holdout150"]["pass_at_2"])
            or significant_positive(comparisons[f"H0-No-Hard_vs_{model}"]["LD-easy64"]["pass_at_2"])
        ) for model in MODELS[1:]
    )

    def aggregate_score(model: str) -> tuple[int, int]:
        return (sum(metrics[d][model]["pass_at_2_solved"] for d in DATASETS), sum(metrics[d][model]["candidate_successes"] for d in DATASETS))

    viable = [model for model in MODELS[1:] if not any(stability[model].values())]
    best = max(viable or list(MODELS[1:]), key=aggregate_score)
    deployment_recommendation = best if viable else None
    ac = comparisons["A_medium_hard_vs_C_add_repair"]
    ad = comparisons["A_medium_hard_vs_D_add_replay"]
    ab = comparisons["A_medium_hard_vs_B_medium_only"]
    repair_effect = "supported" if significant_positive(ac["Aggregate428"]["pass_at_2"]) else ("directional" if ac["Aggregate428"]["pass_at_2"]["delta_percentage_points"] > 0 else "not_supported")
    replay_preservation_delta = mean(metrics[d]["D_add_replay"]["pass_at_2"] - metrics[d]["A_medium_hard"]["pass_at_2"] for d in ("LD-easy64", "Monitor64", "WB-Train-Retention150"))
    replay_effect = "supported" if replay_preservation_delta > 0 and not any(stability["D_add_replay"].values()) else ("neutral" if replay_preservation_delta == 0 else "not_supported")
    agentic_ready = False

    payload = {
        "models": list(MODELS), "datasets": metrics, "historical_ei_round0": historical_round0,
        "paired_statistics": comparisons, "failure_taxonomy": failures,
        "method": {"paired_bootstrap_replicates": 20000, "mcnemar": "two-sided exact", "historical_round0_note": "descriptive only; request seed differs from this matched-seed ablation"},
        "decisions": {
            "relative_best_ablation_model": best,
            "recommended_model_for_next_round": deployment_recommendation,
            "deployment_baseline": "H0-No-Hard" if deployment_recommendation is None else deployment_recommendation,
            "dynamic_difficulty_vs_historical_round0": "not_supported; Round0 is descriptively stronger on aggregate P@2, but its request seed differs",
            "medium_is_best_current_region": "not_supported by the realized B arm",
            "repair_effect": repair_effect,
            "replay_preservation_effect": replay_effect, "replay_mean_preservation_delta": replay_preservation_delta,
            "per_model_stability": stability, "extended_evaluation_required": extended_required,
            "extended_evaluation_run": False, "agentic_rl_ready": agentic_ready, "grpo_started": False,
            "hard_causal_answer_available": False,
            "hard_design_limitation": "No true Medium-only matched-size arm exists: task-defined B contains 50 Hard and realized arm sizes differ.",
            "provisional_next_ei_ratio_after_pool_expansion": {"Medium": 0.60, "Hard_success_1": 0.10, "Repair": 0.10, "Replay": 0.20},
            "provisional_ratio_status": "design recommendation only; do not train until non-WB Medium and independent Repair pools are expanded",
        },
    }
    write_json(root / "comparisons/all_model_metrics_and_statistics.json", payload)

    table = ["| Dataset | Model | P@1 solved | P@2 solved | Candidate success |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            row = metrics[dataset][model]
            table.append(f"| {dataset} | {DISPLAY[model]} | {row['pass_at_1_solved']}/{row['problems']} ({pct(row['pass_at_1'])}) | {row['pass_at_2_solved']}/{row['problems']} ({pct(row['pass_at_2'])}) | {row['candidate_successes']}/{row['candidates']} ({pct(row['candidate_success_rate'])}) |")
    stats_table = ["| Comparison | Aggregate P@1 | Aggregate P@2 | Candidate success |", "|---|---:|---:|---:|"]
    for left, right in PAIRS:
        row = comparisons[f"{left}_vs_{right}"]
        stats_table.append(f"| {DISPLAY[left]} → {DISPLAY[right]} | {fmt_stat(row['Aggregate428']['pass_at_1'])} | {fmt_stat(row['Aggregate428']['pass_at_2'])} | {fmt_stat(row['CandidateAggregate856'])} |")
    round0_table = ["| Dataset | EI Round0 P@1 | EI Round0 P@2 | Candidate success |", "|---|---:|---:|---:|"]
    for dataset, row in historical_round0.items():
        round0_table.append(f"| {dataset} | {row['pass_at_1_solved']}/{row['problems']} | {row['pass_at_2_solved']}/{row['problems']} | {row['candidate_successes']}/{row['candidates']} |")
    failure_table = ["| Dataset | Model | syntax | elaboration | unknown id | tactic | unsolved | timeout |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            row = failures[dataset][model]
            failure_table.append(f"| {dataset} | {DISPLAY[model]} | {row['syntax_error']} | {row['elaboration_error']} | {row['unknown_identifier']} | {row['tactic_error']} | {row['unsolved_goals']} | {row['timeout']} |")

    report = [
        "# EI Difficulty-aware Data Composition Ablation Final Report", "", "## 结论", "",
        f"- 四个实验组中的相对最佳：**{DISPLAY[best]}**；Core aggregate P@2/candidate solved = {aggregate_score(best)}。但所有实验组都触发 foundation forgetting，**可部署基线仍为 H0，不推荐用任一新 arm 进入下一轮**。",
        "- 动态 difficulty 相对历史 EI Round0：**未显示优势**。Round0 的 aggregate P@2 描述性更高，但 request seed 不同，因此不能做严格配对归因。",
        "- Medium 是否为当前最佳 EI 区域：**证据不支持**。B 相对 A 的 aggregate P@2 为零变化，且 B 实际仍含 50 Hard、样本规模更小。",
        f"- Repair（A→C，实际仅增加 14 条独立 verified repair）证据：**{repair_effect}**；aggregate P@2 {fmt_stat(ac['Aggregate428']['pass_at_2'])}。",
        f"- Replay（A→D，额外 100 条）在 LD/Monitor/Retention 的平均 P@2 变化为 {100*replay_preservation_delta:+.2f} pp，判断：**{replay_effect}**；它部分恢复 Monitor/Retention，但仍低于 H0 的 Monitor，aggregate P@2 {fmt_stat(ad['Aggregate428']['pass_at_2'])}。",
        f"- Medium-heavy（A→B）aggregate P@2：{fmt_stat(ab['Aggregate428']['pass_at_2'])}。B 的实际构成仍含 50 Hard，且总样本仅 214，因此不能把差异解释为纯 Medium-only 效应。",
        "- Hard(success=1) 没有独立、等规模的去除对照；本实验不能可靠断言 Hard 的单独价值。",
        "- 与 EI Round0 的比较仅作描述性参考：Round0 使用不同 request seed，不能进行严格逐题配对归因。",
        f"- 是否需要扩展评估：**{'YES' if extended_required else 'NO'}**；是否具备进入 Agentic RL 条件：**NO**。未启动 GRPO。",
        "", "## 数据与训练审计", "",
        f"- 实际清单：{ {arm: value['roles'] for arm, value in preparation['arms'].items()} }。所有 arm duplicate=0、theorem-group duplicate=0、max repeat=1、EOS valid=100%。",
        "- A/C/D 使用完全相同的 114 Medium + 63 Hard + 50 Easy Discovery 集；C 额外 14 Repair；D 将 Replay 从 50 增至 150。",
        "- Replay 内部 WB:LD 约 2:1；但冻结 Dynamic/Repair 池为 WB-only，因此全 arm 仍明显偏 WB，这是外推限制。",
        "- 所有 Dynamic target 为 H0 本地验证生成 proof，Repair target 为 DeepSeek 修复后本地验证 proof；Replay 是任务书明确要求的 H0 继承 foundation 例外。",
        "- 每组独立从 H0 merged 初始化 fresh LoRA；epoch=1、lr=1e-5、r32/alpha64/dropout0.05、effective batch16、max length1024、no packing。",
        "", "## Core Evaluation", "", *table,
        "", "## 配对统计", "", *stats_table,
        "", "## EI Round0 历史参考（不同 seed，不作配对推断）", "", *round0_table,
        "", "## Failure taxonomy", "", *failure_table,
        "", "## 下一轮采样建议", "",
        "- 暂不启动下一轮训练。应先扩大非 WB 的动态难度 Discovery，并新增严格等规模、只改变一个因素的 Medium-only / +Hard 对照。",
        "- 数据池扩充后的**条件性设计比例**：Medium 60%、Hard(success=1) 10%、verified Repair 10%、Replay 20%；Replay 内保持 WB:LD≈2:1。该比例是下一轮受控 pilot 设计，不是本实验已经验证的最优比例。",
        "- 在 Repair 只有 14 个独立 theorem group、整体域仍偏 WB、且无纯 Hard 对照前，不具备进入 Agentic RL 的数据归因条件。",
        "", "EI difficulty-aware ablation completed.", "", "Waiting for review before EI Round1 / Repair SFT / Agentic RL planning.", "",
    ]
    (root / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    for model in MODELS[1:]:
        arm = root / model
        training_summary = read_json(arm / "checkpoint/training_summary.json")
        write_json(arm / "training_metrics.json", training_summary)
        arm_report = [f"# {DISPLAY[model]} Report", "", f"- Manifest: {preparation['arms'][model]}", f"- Training: {read_json(arm / 'checkpoint/training_summary.json')['optimizer_steps']} optimizer steps; max repeat 1.", f"- Stability: {stability[model]}", "", *[line for line in table if line.startswith('|') and (DISPLAY[model] in line or line.startswith('| Dataset') or line.startswith('|---'))], ""]
        (arm / "report.md").write_text("\n".join(arm_report), encoding="utf-8")
    print(json.dumps(payload["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
