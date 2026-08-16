"""Analyze Phase C3 LD-medium against H0/H1/H2/C2/M0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_hard_a_ablation_results import (
    aggregate,
    fmt_ci,
    paired_statistic,
    states,
    summarize,
)


DATASETS = {
    "WB-Unseen-Holdout150": ("wb_unseen_holdout150", 150),
    "LD-easy64": ("ld_easy64", 64),
    "LD-medium-holdout64": ("ld_medium_holdout64", 64),
    "Monitor64": ("monitor64", 64),
    "WB-Train-Retention150": ("wb_train_retention150", 150),
}
MODELS = ("C3", "H0", "H1", "H2", "Phase C2", "M0")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def model_dir(project: Path, model: str, slug: str) -> Path:
    if slug == "ld_medium_holdout64":
        names = {
            "C3": "S2-LD-Medium-Conservative",
            "H0": "H0-No-Hard",
            "H1": "S2-Balanced-Hard",
            "H2": "H2-20pct-Hard-A",
            "Phase C2": "S2-Balanced-Hard",
            "M0": "M0",
        }
        return (
            project
            / "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium/evaluation/full"
            / names[model]
            / slug
        )
    bases = {
        "C3": "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium/evaluation/full/S2-LD-Medium-Conservative",
        "H0": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/evaluation/full/H0-No-Hard",
        "H1": "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/evaluation/full/S2-Balanced-Hard",
        "H2": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H2_20pct_hard/evaluation/full/H2-20pct-Hard-A",
        "Phase C2": "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/evaluation/full/S2-Balanced-Hard",
        "M0": "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/full/M0",
    }
    return project / bases[model] / slug


def fmt_rate(metrics: dict[str, Any]) -> str:
    return f"{metrics['pass_at_1']:.2%}/{metrics['pass_at_2']:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium"
    comparisons = root / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)

    dataset_metrics: dict[str, dict[str, Any]] = {}
    all_pairs: dict[str, dict[str, dict[int, bool]]] = {model: {} for model in MODELS}
    legacy_pairs: dict[str, dict[str, dict[int, bool]]] = {model: {} for model in MODELS}
    auxiliary: dict[str, list[dict[str, list[float]]]] = {model: [] for model in MODELS}
    legacy_auxiliary: dict[str, list[dict[str, list[float]]]] = {model: [] for model in MODELS}

    for label, (slug, expected) in DATASETS.items():
        dataset_metrics[label] = {}
        for model in MODELS:
            metrics, paired, aux = summarize(model_dir(project, model, slug))
            if metrics["problems"] != expected or metrics["candidates"] != 2 * expected:
                raise RuntimeError(f"incomplete C3 result: {model} {label}: {metrics}")
            if metrics["fatal_errors"] or metrics["worker_restarts"]:
                raise RuntimeError(f"runtime instability: {model} {label}: {metrics}")
            dataset_metrics[label][model] = metrics
            auxiliary[model].append(aux)
            for pid, values in paired.items():
                all_pairs[model][f"{label}:{pid}"] = values
                if label != "LD-medium-holdout64":
                    legacy_pairs[model][f"{label}:{pid}"] = values
            if label != "LD-medium-holdout64":
                legacy_auxiliary[model].append(aux)

    aggregate492 = {
        model: aggregate(
            [dataset_metrics[label][model] for label in DATASETS], auxiliary[model]
        )
        for model in MODELS
    }
    legacy_labels = [label for label in DATASETS if label != "LD-medium-holdout64"]
    aggregate428 = {
        model: aggregate(
            [dataset_metrics[label][model] for label in legacy_labels],
            legacy_auxiliary[model],
        )
        for model in MODELS
    }

    comparisons_to_c3: dict[str, Any] = {}
    for baseline in ("M0", "H0", "H1", "H2", "Phase C2"):
        key = f"{baseline} -> C3"
        comparisons_to_c3[key] = {}
        for label in (*DATASETS.keys(), "Aggregate428", "Aggregate492"):
            if label == "Aggregate428":
                baseline_pairs = legacy_pairs[baseline]
                c3_pairs = legacy_pairs["C3"]
            elif label == "Aggregate492":
                baseline_pairs = all_pairs[baseline]
                c3_pairs = all_pairs["C3"]
            else:
                prefix = f"{label}:"
                baseline_pairs = {
                    key[len(prefix) :]: value
                    for key, value in all_pairs[baseline].items()
                    if key.startswith(prefix)
                }
                c3_pairs = {
                    key[len(prefix) :]: value
                    for key, value in all_pairs["C3"].items()
                    if key.startswith(prefix)
                }
            comparisons_to_c3[key][label] = {
                metric: paired_statistic(
                    states(baseline_pairs, metric),
                    states(c3_pairs, metric),
                    f"C3|{baseline}|{label}|{metric}",
                )
                for metric in ("pass_at_1", "pass_at_2")
            }

    pairwise = {
        "method": {
            "paired_bootstrap_replicates": 20000,
            "paired_bootstrap_unit": "problem",
            "mcnemar": "two-sided exact binomial McNemar",
            "direction": "C3 minus baseline",
        },
        "comparisons": comparisons_to_c3,
        "aliases": {
            "H1": "Phase C2 S2-Balanced-Hard",
            "independent_replication": False,
            "note": "H1 and Phase C2 are the same frozen experiment artifact.",
        },
    }
    write_json(comparisons / "pairwise_statistics.json", pairwise)
    write_json(
        comparisons / "focus_c3_vs_h0_c2.json",
        {
            "H0_to_C3": comparisons_to_c3["H0 -> C3"],
            "Phase_C2_to_C3": comparisons_to_c3["Phase C2 -> C3"],
        },
    )

    integrity = {
        "status": "PASSED",
        "expected_problems_per_model": 492,
        "expected_candidates_per_model": 984,
        "models": {
            model: {
                "problems": aggregate492[model]["problems"],
                "candidates": aggregate492[model]["candidates"],
                "fatal_errors": aggregate492[model]["fatal_errors"],
                "worker_restarts": aggregate492[model]["worker_restarts"],
            }
            for model in MODELS
        },
        "all_complete": all(
            aggregate492[model]["problems"] == 492
            and aggregate492[model]["candidates"] == 984
            for model in MODELS
        ),
        "forbidden_suites_run": [],
    }
    model_metrics = {
        "datasets": dataset_metrics,
        "aggregate428_existing_core": aggregate428,
        "aggregate492_with_ld_medium": aggregate492,
        "integrity": integrity,
        "aliases": pairwise["aliases"],
    }
    write_json(comparisons / "model_metrics.json", model_metrics)

    columns = ("C3", "H0", "H1", "H2", "Phase C2", "M0")
    table = [
        "| 数据集 | C3 p@1/p@2 | H0 | H1 | H2 | C2 | M0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in DATASETS:
        row = dataset_metrics[label]
        table.append(
            "| " + label + " | " + " | ".join(fmt_rate(row[model]) for model in columns) + " |"
        )
    table.append(
        "| Aggregate428 | "
        + " | ".join(fmt_rate(aggregate428[model]) for model in columns)
        + " |"
    )
    table.append(
        "| Aggregate492 | "
        + " | ".join(fmt_rate(aggregate492[model]) for model in columns)
        + " |"
    )
    (comparisons / "comparison_table.md").write_text(
        "\n".join(["# Phase C3 模型对照", "", *table, ""]), encoding="utf-8"
    )

    training = read_json(root / "training_metrics.json")
    data = read_json(root / "data_audit.json")
    leakage = read_json(root / "leakage_audit.json")
    eos = read_json(root / "eos_audit.json")
    contract = read_json(root / "evaluation/generation_contract.json")
    canary = read_json(root / "evaluation/canary/canary_gate.json")
    c3 = aggregate428["C3"]
    h0 = aggregate428["H0"]
    c2 = aggregate428["Phase C2"]
    m0 = aggregate428["M0"]
    focus_h0 = comparisons_to_c3["H0 -> C3"]
    focus_c2 = comparisons_to_c3["Phase C2 -> C3"]

    report = [
        "# Phase C3：LD-medium Conservative Curriculum 实验报告",
        "",
        "## 结论",
        "",
        "Phase C3 已完成数据构造、全部门禁、独立 LoRA 训练、canonical merged canary、五组正式评估和配对统计。未运行 Full500、Strict-unseen200、miniF2F，也未启动 Expert Iteration。",
        "",
        "C3 明显提高了 WB-Unseen，但没有在新的 LD-medium 留出集上超过 M0/C2，更低于 H0/H2；同时 LD-easy pass@1 相对 C2 明显回落。因此现有证据不支持把 10% LD-medium 直接冻结为第二轮 SFT 默认配方。Foundation dominant 仍成立，但 LD-medium curriculum 需要更保守比例、更强可学习性筛选或分阶段训练后再复验。",
        "",
        "## 数据与门禁",
        "",
        f"- 训练 2000 行：New WB {data['role_counts']['New WB']}、LD-medium {data['role_counts']['LD-medium']}、Replay {data['role_counts']['Replay']}。Replay=Stable/Core 12 + Frontier 47 + verified WB 41；Hard-A/B/C=0。",
        f"- 两轮冻结人工/Codex 语义分级共 {data['ld_medium']['manual_labels_total']} 条，其中 medium {data['ld_medium']['manual_medium_total']} 条；训练选择 200，独立 holdout 64。",
        f"- LD-medium train/holdout 六轴零重叠，source-file disjoint={data['ld_medium']['source_file_disjoint']}（holdout 20 个 source files）。holdout proof-free={data['ld_medium']['holdout_proof_free']}。",
        f"- Manifest `{data['manifest_sha256']}`；duplicate rows=0、theorem-group duplicates=0、max repeat=1、proof 修改=0。",
        f"- Leakage={leakage['status']}；EOS={eos['status']}，supervised EOS 2000/2000、zero-label=0、semantic truncation=0。",
        "",
        "## 训练与 canonical inference",
        "",
        f"- 从冻结 merged M0 新建 LoRA：r={training['resolved_config']['lora_r']}、alpha={training['resolved_config']['lora_alpha']}、dropout={training['resolved_config']['lora_dropout']}、LR={training['resolved_config']['learning_rate']}、epoch=1、effective batch=16。",
        f"- 125/125 optimizer steps；draws 2000、unique 2000、max repeat 1；train loss {training['metrics']['train_loss']:.6f}；eval loss {training['eval_loss']:.6f}；eval token accuracy {training['eval_token_accuracy']:.2%}。",
        f"- M0 训练前后哈希一致 `{training['frozen_m0_hash_after_training']}`；C3 merged hash `{contract['checkpoint_model_sha256']}`。",
        f"- 推理路径：merged checkpoint + Transformers，adapter=null；父合同 `{contract['parent_generation_contract_sha256']}`，C3 resolved contract `{contract['generation_config_sha256']}`。",
        f"- Canary PASS={canary['gate_passed']}；140/140 candidates；抽取率 {canary['combined']['S2-LD-Medium-Conservative']['extraction_success_rate']:.2%}；平均长度 {canary['combined']['S2-LD-Medium-Conservative']['completion_tokens_mean']:.2f}；length finish {canary['combined']['S2-LD-Medium-Conservative']['max_length_finish_ratio']:.2%}；病理重复 {canary['combined']['S2-LD-Medium-Conservative']['pathological_repetition_rate']:.2%}。",
        "",
        "## 正式评估",
        "",
        *table,
        "",
        f"既有四集合 Aggregate428：C3 {fmt_rate(c3)}，H0 {fmt_rate(h0)}，C2 {fmt_rate(c2)}，M0 {fmt_rate(m0)}。C3 相对 C2 为 p@1 {(c3['pass_at_1']-c2['pass_at_1'])*100:+.2f} pp、p@2 {(c3['pass_at_2']-c2['pass_at_2'])*100:+.2f} pp；相对 H0 为 p@1 {(c3['pass_at_1']-h0['pass_at_1'])*100:+.2f} pp、p@2 {(c3['pass_at_2']-h0['pass_at_2'])*100:+.2f} pp。",
        "",
        "## 重点配对统计",
        "",
        f"- C3 vs H0，Aggregate428 p@1：{fmt_ci(focus_h0['Aggregate428']['pass_at_1'])}；p@2：{fmt_ci(focus_h0['Aggregate428']['pass_at_2'])}。",
        f"- C3 vs C2，Aggregate428 p@1：{fmt_ci(focus_c2['Aggregate428']['pass_at_1'])}；p@2：{fmt_ci(focus_c2['Aggregate428']['pass_at_2'])}。",
        f"- C3 vs H0，WB-Unseen p@1：{fmt_ci(focus_h0['WB-Unseen-Holdout150']['pass_at_1'])}；p@2：{fmt_ci(focus_h0['WB-Unseen-Holdout150']['pass_at_2'])}。",
        f"- C3 vs C2，WB-Unseen p@1：{fmt_ci(focus_c2['WB-Unseen-Holdout150']['pass_at_1'])}；p@2：{fmt_ci(focus_c2['WB-Unseen-Holdout150']['pass_at_2'])}。",
        f"- C3 vs H0，LD-medium p@1：{fmt_ci(focus_h0['LD-medium-holdout64']['pass_at_1'])}；p@2：{fmt_ci(focus_h0['LD-medium-holdout64']['pass_at_2'])}。",
        f"- C3 vs C2，LD-medium p@1：{fmt_ci(focus_c2['LD-medium-holdout64']['pass_at_1'])}；p@2：{fmt_ci(focus_c2['LD-medium-holdout64']['pass_at_2'])}。",
        "",
        "全部 C3-vs-baseline、五数据集、Aggregate428/492、p@1/p@2 的 20,000 次 paired bootstrap CI 和 exact McNemar 保存在 `comparisons/pairwise_statistics.json`。H1 与 Phase C2 为同一冻结工件，不视为独立重复。",
        "",
        "## 对任务书问题的回答",
        "",
        f"1. **LD-medium 是否带来额外收益？** 未观察到专项收益。LD-medium-holdout64：C3 {fmt_rate(dataset_metrics['LD-medium-holdout64']['C3'])}，M0/C2 {fmt_rate(dataset_metrics['LD-medium-holdout64']['M0'])}，H0/H2 {fmt_rate(dataset_metrics['LD-medium-holdout64']['H0'])}。LD-easy C3 {fmt_rate(dataset_metrics['LD-easy64']['C3'])}，相对 C2 {fmt_rate(dataset_metrics['LD-easy64']['Phase C2'])} 回落。",
        f"2. **是否损害 foundation？** 没有明显损害，反而提升 WB-Unseen：C3 {fmt_rate(dataset_metrics['WB-Unseen-Holdout150']['C3'])}，C2 {fmt_rate(dataset_metrics['WB-Unseen-Holdout150']['Phase C2'])}；Retention C3 {fmt_rate(dataset_metrics['WB-Train-Retention150']['C3'])}，C2 {fmt_rate(dataset_metrics['WB-Train-Retention150']['Phase C2'])}；Monitor C3 {fmt_rate(dataset_metrics['Monitor64']['C3'])}。",
        "3. **LD-medium 与 Hard-A 哪个更值得加入？** 当前不能判定 LD-medium 更优。C3 的 WB unseen 更好、Aggregate p@2 略高，但 Aggregate p@1 与 LD-easy 更低，且 LD-medium 专项留出没有提升。Hard-A 也没有已证明的独立显著收益，因此两者都不应扩大比例。",
        "4. **是否支持 Foundation → LD-medium → Expert Iteration curriculum？** 只支持第一段 Foundation。10% LD-medium 这一步尚未通过收益门禁；建议先缩小到 2%–5%、加强 learnability/near-boundary 筛选或采用分阶段短程试验，再决定是否进入 Expert Iteration。",
        "",
        "## 停止状态",
        "",
        "Phase C3 completed.",
        "",
        "Waiting for review before Expert Iteration planning.",
        "",
    ]
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(
        root / "evaluation/status.json",
        {
            "status": "PHASE_C3_EVALUATION_COMPLETED",
            "canary_gate_passed": canary["gate_passed"],
            "formal_evaluation_completed": True,
            "c3_formal_problems": aggregate492["C3"]["problems"],
            "c3_formal_candidates": aggregate492["C3"]["candidates"],
            "all_model_comparison_problems": sum(
                aggregate492[model]["problems"] for model in MODELS
            ),
            "all_model_comparison_candidates": sum(
                aggregate492[model]["candidates"] for model in MODELS
            ),
            "fatal_errors": aggregate492["C3"]["fatal_errors"],
            "worker_restarts": aggregate492["C3"]["worker_restarts"],
            "forbidden_suites_run": [],
            "expert_iteration_started": False,
            "report": str(root / "report.md"),
        },
    )
    print(
        json.dumps(
            {
                "status": "PHASE_C3_COMPLETED",
                "integrity": integrity,
                "aggregate428": aggregate428,
                "aggregate492": aggregate492,
                "report": str(root / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
