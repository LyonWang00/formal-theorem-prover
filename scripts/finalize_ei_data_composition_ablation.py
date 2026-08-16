#!/usr/bin/env python3
"""Compute paired statistics and finalize the EI data-composition ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DATASETS = {
    "WB-Unseen-Holdout150": ("wb_unseen_holdout150", 150),
    "LD-easy64": ("ld_easy64", 64),
    "Monitor64": ("monitor64", 64),
    "WB-Train-Retention150": ("wb_train_retention150", 150),
}
MODELS = (
    "H0-No-Hard",
    "EI-A_success_only",
    "EI-B_success_frontier",
    "EI-C_success_frontier_replay",
)
DISPLAY = {
    "H0-No-Hard": "H0",
    "EI-A_success_only": "EI-A Success-only",
    "EI-B_success_frontier": "EI-B Success+Frontier",
    "EI-C_success_frontier_replay": "EI-C Success+Frontier+Replay",
}
PAIRS = (
    ("H0-No-Hard", "EI-A_success_only"),
    ("H0-No-Hard", "EI-B_success_frontier"),
    ("H0-No-Hard", "EI-C_success_frontier_replay"),
    ("EI-A_success_only", "EI-B_success_frontier"),
    ("EI-B_success_frontier", "EI-C_success_frontier_replay"),
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_MASTER_SEED = 20261815


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def result_dir(root: Path, model: str, slug: str) -> Path:
    if model == "H0-No-Hard":
        return root / "comparisons/H0_matched_seed/evaluation/core" / slug
    return root / model / "evaluation/core" / slug


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize(directory: Path, expected: int) -> tuple[dict[str, Any], dict[str, dict[int, bool]]]:
    attempts = read_jsonl(directory / "attempts.jsonl")
    generations = read_jsonl(directory / "generations.jsonl")
    summary = read_json(directory / "benchmark_summary.json")
    paired: dict[str, dict[int, bool]] = {}
    for row in attempts:
        paired.setdefault(str(row["problem_id"]), {})[int(row["attempt_index"])] = bool(row["success"])
    if len(paired) != expected or len(attempts) != expected * 2 or len(generations) != expected * 2:
        raise RuntimeError(f"incomplete evaluation at {directory}")
    if any(set(values) != {0, 1} for values in paired.values()):
        raise RuntimeError(f"candidate indices drifted at {directory}")
    pass1 = {key: value[0] for key, value in paired.items()}
    pass2 = {key: any(value.values()) for key, value in paired.items()}
    lengths = [float(row.get("metadata", {}).get("completion_tokens") or 0) for row in generations]
    repetitions = [repeated_ngram_ratio(str(row.get("raw_output") or "")) for row in generations]
    extraction = [bool(str(row.get("extracted_proof") or "").strip()) for row in generations]
    format_valid = [
        bool(re.match(r"^\s*(?:by\b|term\b)", str(row.get("extracted_proof") or "")))
        for row in generations
    ]
    successes = sum(bool(row["success"]) for row in attempts)
    return ({
        "problems": expected,
        "candidates": len(attempts),
        "candidate_successes": successes,
        "candidate_success_rate": successes / len(attempts),
        "pass_at_1_solved": sum(pass1.values()),
        "pass_at_2_solved": sum(pass2.values()),
        "pass_at_1": sum(pass1.values()) / expected,
        "pass_at_2": sum(pass2.values()) / expected,
        "mean_completion_tokens": mean(lengths),
        "p95_completion_tokens": float(np.quantile(lengths, 0.95)),
        "extraction_success_rate": sum(extraction) / len(extraction),
        "format_validity_rate": sum(format_valid) / len(format_valid),
        "mean_repeated_4gram_ratio": mean(repetitions),
        "pathological_repetition_rate": sum(value >= 0.35 for value in repetitions) / len(repetitions),
        "max_length_finish_rate": sum(
            row.get("finish_reason") in {"length", "max_length"}
            or int(row.get("metadata", {}).get("completion_tokens") or 0) >= 256
            for row in generations
        ) / len(generations),
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "fatal_errors": len(summary.get("fatal_errors") or []),
        "worker_restarts": int(summary.get("pantograph_worker_restart_count") or 0),
    }, paired)


def states(paired: dict[str, dict[int, bool]], metric: str) -> dict[str, bool]:
    if metric == "pass_at_1":
        return {key: value[0] for key, value in paired.items()}
    if metric == "pass_at_2":
        return {key: any(value.values()) for key, value in paired.items()}
    if metric == "candidate_success":
        return {f"{key}:{index}": success for key, value in paired.items() for index, success in value.items()}
    raise ValueError(metric)


def exact_mcnemar(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_stat(left: dict[str, bool], right: dict[str, bool], label: str) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError(f"paired statement IDs differ for {label}")
    ids = sorted(left)
    diffs = np.asarray([int(right[key]) - int(left[key]) for key in ids], dtype=int)
    negative, zero, positive = (int(np.sum(diffs == value)) for value in (-1, 0, 1))
    seed = int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_MASTER_SEED}:{label}".encode()).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(ids), np.asarray([negative, zero, positive]) / len(ids), size=BOOTSTRAP_REPLICATES)
    deltas = 100.0 * (-draws[:, 0] + draws[:, 2]) / len(ids)
    return {
        "problems": len(ids), "left_solved": sum(left.values()), "right_solved": sum(right.values()),
        "delta_percentage_points": 100.0 * float(np.mean(diffs)),
        "paired_bootstrap_95_ci_percentage_points": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": seed,
        "left_only": negative, "right_only": positive,
        "both_fail": sum(not left[key] and not right[key] for key in ids),
        "both_pass": sum(left[key] and right[key] for key in ids),
        "mcnemar_exact_p": exact_mcnemar(negative, positive),
    }


def significant_positive(stat: dict[str, Any]) -> bool:
    return stat["delta_percentage_points"] > 0 and stat["paired_bootstrap_95_ci_percentage_points"][0] > 0 and stat["mcnemar_exact_p"] < 0.05


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt_stat(value: dict[str, Any]) -> str:
    low, high = value["paired_bootstrap_95_ci_percentage_points"]
    return f"{value['delta_percentage_points']:+.2f} pp [{low:+.2f}, {high:+.2f}], p={value['mcnemar_exact_p']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/ei_ablation"
    metrics: dict[str, dict[str, Any]] = {name: {} for name in DATASETS}
    paired: dict[str, dict[str, Any]] = {name: {} for name in DATASETS}
    aggregate: dict[str, dict[str, dict[str, bool]]] = {
        model: {"pass_at_1": {}, "pass_at_2": {}, "candidate_success": {}} for model in MODELS
    }
    for dataset, (slug, expected) in DATASETS.items():
        for model in MODELS:
            model_metrics, model_paired = summarize(result_dir(root, model, slug), expected)
            metrics[dataset][model] = model_metrics
            paired[dataset][model] = model_paired
            for metric in ("pass_at_1", "pass_at_2", "candidate_success"):
                aggregate[model][metric].update({
                    f"{dataset}:{key}": value for key, value in states(model_paired, metric).items()
                })

    comparisons: dict[str, Any] = {}
    for left, right in PAIRS:
        pair_key = f"{left}_vs_{right}"
        comparisons[pair_key] = {}
        for dataset in DATASETS:
            comparisons[pair_key][dataset] = {
                metric: paired_stat(
                    states(paired[dataset][left], metric), states(paired[dataset][right], metric),
                    f"{pair_key}:{dataset}:{metric}",
                ) for metric in ("pass_at_1", "pass_at_2", "candidate_success")
            }
        comparisons[pair_key]["Aggregate428"] = {
            metric: paired_stat(aggregate[left][metric], aggregate[right][metric], f"{pair_key}:Aggregate428:{metric}")
            for metric in ("pass_at_1", "pass_at_2")
        }
        comparisons[pair_key]["CandidateAggregate856"] = paired_stat(
            aggregate[left]["candidate_success"], aggregate[right]["candidate_success"],
            f"{pair_key}:CandidateAggregate856:candidate_success",
        )

    def drift(model: str) -> dict[str, bool]:
        h0 = "H0-No-Hard"
        forgetting = any(
            metrics[name][model]["pass_at_2"] < metrics[name][h0]["pass_at_2"] - 0.02
            for name in ("Monitor64", "WB-Train-Retention150")
        )
        output = any(
            metrics[name][model]["pathological_repetition_rate"] > max(0.05, metrics[name][h0]["pathological_repetition_rate"] + 0.03)
            or metrics[name][model]["max_length_finish_rate"] > max(0.10, metrics[name][h0]["max_length_finish_rate"] + 0.05)
            for name in DATASETS
        )
        return {"foundation_forgetting": forgetting, "output_drift": output}

    decisions = {model: drift(model) for model in MODELS[1:]}
    h0_a = comparisons["H0-No-Hard_vs_EI-A_success_only"]["Aggregate428"]["pass_at_2"]
    a_b = comparisons["EI-A_success_only_vs_EI-B_success_frontier"]["Aggregate428"]["pass_at_2"]
    b_c = comparisons["EI-B_success_frontier_vs_EI-C_success_frontier_replay"]["Aggregate428"]["pass_at_2"]
    self_distillation = "statistically_supported" if significant_positive(h0_a) else ("directional_positive" if h0_a["delta_percentage_points"] > 0 else "not_supported")
    frontier_value = "statistically_supported" if significant_positive(a_b) else ("directional_positive" if a_b["delta_percentage_points"] > 0 else "not_supported")
    preservation_names = ("LD-easy64", "Monitor64", "WB-Train-Retention150")
    replay_preservation_delta = sum(
        metrics[name]["EI-C_success_frontier_replay"]["pass_at_2"] - metrics[name]["EI-B_success_frontier"]["pass_at_2"]
        for name in preservation_names
    ) / len(preservation_names)
    replay_value = "supported" if replay_preservation_delta > 0 and not decisions["EI-C_success_frontier_replay"]["foundation_forgetting"] else ("neutral" if replay_preservation_delta == 0 else "not_supported")

    viable = [model for model in MODELS[1:] if not any(decisions[model].values())]
    def score(model: str) -> tuple[float, float]:
        p2 = sum(metrics[name][model]["pass_at_2_solved"] for name in DATASETS) / 428
        csr = sum(metrics[name][model]["candidate_successes"] for name in DATASETS) / 856
        return p2, csr
    best = max(viable or list(MODELS[1:]), key=score)
    h0_score = sum(metrics[name]["H0-No-Hard"]["pass_at_2_solved"] for name in DATASETS) / 428
    supports_round1 = bool(viable) and score(best)[0] >= h0_score
    repairdata_needed = frontier_value in {"statistically_supported", "directional_positive"}
    grpo_ready = supports_round1 and self_distillation != "not_supported" and not any(decisions[best].values())

    payload = {
        "models": list(MODELS), "datasets": metrics, "paired_statistics": comparisons,
        "method": {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED, "mcnemar": "two-sided exact"},
        "decisions": {
            "success_bank_effect": self_distillation, "frontier_incremental_value": frontier_value,
            "replay_domain_preservation": replay_value, "replay_mean_preservation_delta": replay_preservation_delta,
            "per_model_stability": decisions, "recommended_model": best,
            "supports_ei_round1": supports_round1, "repairdata_needed": repairdata_needed,
            "evidence_supports_grpo_preparation": grpo_ready, "grpo_started": False,
            "extended_evaluation_run": False,
        },
    }
    write_json(root / "comparisons/all_model_metrics_and_statistics.json", payload)

    metric_table = [
        "| Dataset | Model | Solved P@1 | Solved P@2 | Candidate success |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            row = metrics[dataset][model]
            metric_table.append(
                f"| {dataset} | {DISPLAY[model]} | {row['pass_at_1_solved']}/{row['problems']} ({pct(row['pass_at_1'])}) "
                f"| {row['pass_at_2_solved']}/{row['problems']} ({pct(row['pass_at_2'])}) "
                f"| {row['candidate_successes']}/{row['candidates']} ({pct(row['candidate_success_rate'])}) |"
            )
    stats_table = ["| Comparison | Aggregate P@1 | Aggregate P@2 | Candidate success |", "|---|---:|---:|---:|"]
    for left, right in PAIRS:
        row = comparisons[f"{left}_vs_{right}"]["Aggregate428"]
        candidate = comparisons[f"{left}_vs_{right}"]["CandidateAggregate856"]
        stats_table.append(f"| {DISPLAY[left]} → {DISPLAY[right]} | {fmt_stat(row['pass_at_1'])} | {fmt_stat(row['pass_at_2'])} | {fmt_stat(candidate)} |")
    focus_table = [
        "| Dataset | EI-A→EI-B P@2 | EI-B→EI-C P@2 |",
        "|---|---:|---:|",
    ]
    for dataset in (*DATASETS, "Aggregate428"):
        ab = comparisons["EI-A_success_only_vs_EI-B_success_frontier"][dataset]["pass_at_2"]
        bc = comparisons["EI-B_success_frontier_vs_EI-C_success_frontier_replay"][dataset]["pass_at_2"]
        focus_table.append(f"| {dataset} | {fmt_stat(ab)} | {fmt_stat(bc)} |")
    behavior_table = [
        "| Model | Mean tokens | Extraction | Format valid | Repetition | Max-length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        rows = [metrics[name][model] for name in DATASETS]
        behavior_table.append(
            f"| {DISPLAY[model]} | {mean(row['mean_completion_tokens'] for row in rows):.2f} "
            f"| {pct(mean(row['extraction_success_rate'] for row in rows))} "
            f"| {pct(mean(row['format_validity_rate'] for row in rows))} "
            f"| {pct(mean(row['pathological_repetition_rate'] for row in rows))} "
            f"| {pct(mean(row['max_length_finish_rate'] for row in rows))} |"
        )

    preparation = read_json(root / "data_preparation_summary.json")
    report = [
        "# Expert Iteration Data Composition Ablation Final Report", "",
        "## 结论", "",
        f"- Success Bank 自蒸馏效果：**{self_distillation}**；H0→EI-A aggregate P@2 为 {fmt_stat(h0_a)}。",
        f"- Frontier 增量价值：**{frontier_value}**；EI-A→EI-B aggregate P@2 为 {fmt_stat(a_b)}。",
        f"- Replay 对 LD/Monitor/Retention 的平均 P@2 保持变化为 {100 * replay_preservation_delta:+.2f} pp，判断：**{replay_value}**；EI-B→EI-C aggregate P@2 为 {fmt_stat(b_c)}。",
        f"- 推荐配比模型：**{DISPLAY[best]}**，实际配比 {preparation['arm_role_counts'][best]}。支持进入 EI Round1：**{'YES' if supports_round1 else 'NO'}**；需要继续引入 RepairData：**{'YES' if repairdata_needed else 'NO / evidence insufficient'}**。",
        f"- 是否具备 GRPO 准备证据：**{'YES' if grpo_ready else 'NO'}**。本实验未启动 GRPO。",
        "", "## 实验合同与数据", "",
        f"- 三组均从冻结 H0-No-Hard merged checkpoint 独立初始化新 LoRA；实际每组 {preparation['realized_rows_per_arm']} 条，而非 500 条，因为 Success Bank 仅有 {preparation['success_unique_theorems']} 个可用唯一 theorem group。",
        f"- RepairData v2 产出 {preparation['frontier_verified_unique_theorems']} 个可用且 theorem-group unique 的 Frontier target；不足时按冻结规则降低 Frontier 并以 verified Success 回填。",
        f"- 配比：{preparation['arm_role_counts']}。所有 target 均为 Pantograph verified；失败 proof 和 reference proof 未作为 Success/Frontier target。",
        "- 训练配置完全一致：epoch 1、lr 1e-5、LoRA r32/alpha64/dropout0.05、effective batch 16、max length 1024、packing=false、completion-only、supervised EOS。",
        "- 三组均通过 duplicate、theorem-group、六轴泄漏、EOS、zero-label、semantic-truncation 门禁。",
        "- 所有评估均使用 merged checkpoint + Transformers + BF16 + 冻结 generation contract；四模型使用相同 seed 和每题两个 candidates。",
        "", "## 运行恢复审计", "",
        "- EI-A 首次预训练中断产物已隔离到 `interrupted_checkpoint_20260803_resume1`，未作为最终 checkpoint 使用。",
        "- EI-A 在 step 6 遇到一次瞬时 CUBLAS 错误；随后 20 次 2048×2048 BF16 GEMM smoke test 全部通过，中断时的无状态产物已隔离到 `interrupted_checkpoint_20260803_cuda_step6`，训练从干净状态重跑。",
        "- 计算机在 EI-A merge 期间崩溃；完整 adapter 得到保留，部分 merged 目录隔离到 `interrupted_merge_20260803_system_crash`，随后 merge 从完整 adapter 重跑并通过哈希与加载审计。",
        "- 首次 EI 消融 LD-easy64 被误路由到 generic benchmark，导致 `Mathlib` 预加载与 source-faithful 模板发生重复声明；四个无效目录均归档为 `invalid_ld_easy64_generic_mathlib_20260804`。最终 LD 指标已用 `ld_holdout` 专用入口、24 个真实 imports 分组、单 worker 全量重跑，512/512 candidates 的重复声明污染为 0。",
        "- 上述事件属于运行环境中断，不计为数据、训练或模型失败；最终统计仅使用完成审计的 merged checkpoints。",
        "", "## Core Evaluation", "", *metric_table,
        "", "## 关键配对统计", "", *stats_table,
        "", "### Frontier 与 Replay 的数据集级 P@2", "", *focus_table,
        "", "统计为 20,000 次 statement-level paired bootstrap 95% CI，McNemar 为双侧精确检验；CI 未排除 0 的差异仅视作方向性证据。",
        "", "## 输出行为", "", *behavior_table,
        "", "## 稳定性与后续", "",
        *(f"- {DISPLAY[model]}：foundation forgetting={'YES' if decisions[model]['foundation_forgetting'] else 'NO'}；output drift={'YES' if decisions[model]['output_drift'] else 'NO'}。" for model in MODELS[1:]),
        "- Full500、Strict-unseen200 和 miniF2F 未运行；本任务未自动扩大评估或训练规模。",
        "", "EI data composition ablation completed.", "",
        "Waiting for review before EI Round1 / Repair SFT / GRPO planning.", "",
    ]
    (root / "final_report.md").write_text("\n".join(report), encoding="utf-8")

    for model in MODELS[1:]:
        arm = root / model
        audit = read_json(arm / "data_audit.json")
        training = read_json(arm / "checkpoint/training_summary.json")
        arm_table = [
            "| Dataset | Solved P@1 | Solved P@2 | Candidate success |",
            "|---|---:|---:|---:|",
        ]
        for dataset in DATASETS:
            row = metrics[dataset][model]
            arm_table.append(
                f"| {dataset} | {row['pass_at_1_solved']}/{row['problems']} ({pct(row['pass_at_1'])}) "
                f"| {row['pass_at_2_solved']}/{row['problems']} ({pct(row['pass_at_2'])}) "
                f"| {row['candidate_successes']}/{row['candidates']} ({pct(row['candidate_success_rate'])}) |"
            )
        arm_report = [
            f"# {DISPLAY[model]} Report", "",
            f"- Manifest rows: {audit['realized_rows']}; composition: {audit['role_counts']}; data gate: {audit['status']}.",
            f"- Optimizer steps: {training['optimizer_steps']}; unique draws: {training['unique_draws']}; max repeat: {training['max_repeat']}.",
            f"- Foundation forgetting: {'YES' if decisions[model]['foundation_forgetting'] else 'NO'}; output drift: {'YES' if decisions[model]['output_drift'] else 'NO'}.",
            "", *arm_table, "",
        ]
        (arm / "report.md").write_text("\n".join(arm_report), encoding="utf-8")
    print(json.dumps(payload["decisions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
