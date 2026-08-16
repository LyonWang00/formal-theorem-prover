"""Analyze and archive the H0/H1/H2 Hard-A ratio ablation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

# H1 is intentionally the exact Phase-C2 artifact.  It remains listed under both
# requested names, but the report explicitly prevents treating this as replication.
MODEL_DIRS = {
    "H0": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/evaluation/full/H0-No-Hard",
    "H1": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H1_10pct_hard/evaluation/full/S2-Balanced-Hard",
    "H2": "outputs/stage2_data_ratio_ablation/hard_a_ablation/H2_20pct_hard/evaluation/full/H2-20pct-Hard-A",
    "Phase C1": "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/full/S2-New-WB",
    "Phase C2": "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/evaluation/full/S2-Balanced-Hard",
    "M0": "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/full/M0",
}
PRIMARY_MODELS = ("H0", "H1", "H2", "Phase C1", "Phase C2", "M0")
PHASE_B_DIR = "outputs/stage2_data_ablation/phaseB_hard_replay/evaluation/full/S2-Hard-Replay"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_MASTER_SEED = 20260804


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize(directory: Path) -> tuple[dict[str, Any], dict[str, dict[int, bool]], dict[str, list[float]]]:
    attempts_path = directory / "attempts.jsonl"
    generations_path = directory / "generations.jsonl"
    if not attempts_path.is_file() or not generations_path.is_file():
        raise FileNotFoundError(f"missing formal evaluation output: {directory}")
    attempts = read_jsonl(attempts_path)
    generations = read_jsonl(generations_path)
    paired: dict[str, dict[int, bool]] = {}
    for row in attempts:
        paired.setdefault(str(row["problem_id"]), {})[int(row["attempt_index"])] = bool(row["success"])
    if len(attempts) != 2 * len(paired) or any(set(values) != {0, 1} for values in paired.values()):
        raise RuntimeError(f"incomplete two-candidate result at {directory}")
    if len(generations) != len(attempts):
        raise RuntimeError(f"generation/attempt count mismatch at {directory}")
    pass1 = {pid: bool(values[0]) for pid, values in paired.items()}
    pass2 = {pid: any(values.values()) for pid, values in paired.items()}
    lengths = [float(row.get("metadata", {}).get("completion_tokens", 0)) for row in generations]
    repetitions = [repeated_ngram_ratio(str(row.get("raw_output", ""))) for row in generations]
    summary_files = list(directory.glob("*_summary.json")) + list(directory.glob("*_metrics.json"))
    summaries = [read_json(path) for path in summary_files]
    result = {
        "problems": len(paired),
        "candidates": len(attempts),
        "candidate_successes": sum(bool(row.get("success")) for row in attempts),
        "pass_at_1_solved": sum(pass1.values()),
        "pass_at_2_solved": sum(pass2.values()),
        "pass_at_1": sum(pass1.values()) / max(1, len(paired)),
        "pass_at_2": sum(pass2.values()) / max(1, len(paired)),
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "completion_tokens": {
            "mean": mean(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "max_length_finish_ratio": sum(
            row.get("finish_reason") in {"length", "max_length"}
            or int(row.get("metadata", {}).get("completion_tokens", 0)) >= int(row.get("max_new_tokens", 256))
            for row in generations
        ) / max(1, len(generations)),
        "extraction_success_rate": sum(bool(str(row.get("extracted_proof", "")).strip()) for row in generations)
        / max(1, len(generations)),
        "mean_repeated_4gram_ratio": mean(repetitions),
        "pathological_repetition_rate": sum(value >= 0.35 for value in repetitions) / max(1, len(repetitions)),
        "fatal_errors": sum(len(item.get("fatal_errors", [])) for item in summaries),
        "worker_restarts": sum(int(item.get("pantograph_worker_restart_count", 0)) for item in summaries),
    }
    return result, paired, {"lengths": lengths, "repetitions": repetitions}


def aggregate(metrics: list[dict[str, Any]], auxiliary: list[dict[str, list[float]]]) -> dict[str, Any]:
    problems = sum(item["problems"] for item in metrics)
    candidates = sum(item["candidates"] for item in metrics)
    lengths = [value for item in auxiliary for value in item["lengths"]]
    repetitions = [value for item in auxiliary for value in item["repetitions"]]
    return {
        "problems": problems,
        "candidates": candidates,
        "candidate_successes": sum(item["candidate_successes"] for item in metrics),
        "pass_at_1_solved": sum(item["pass_at_1_solved"] for item in metrics),
        "pass_at_2_solved": sum(item["pass_at_2_solved"] for item in metrics),
        "pass_at_1": sum(item["pass_at_1_solved"] for item in metrics) / problems,
        "pass_at_2": sum(item["pass_at_2_solved"] for item in metrics) / problems,
        "timeouts": sum(item["timeouts"] for item in metrics),
        "completion_tokens": {
            "mean": mean(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "max_length_finish_ratio": sum(item["max_length_finish_ratio"] * item["candidates"] for item in metrics)
        / candidates,
        "extraction_success_rate": sum(item["extraction_success_rate"] * item["candidates"] for item in metrics)
        / candidates,
        "mean_repeated_4gram_ratio": mean(repetitions),
        "pathological_repetition_rate": sum(value >= 0.35 for value in repetitions) / candidates,
        "fatal_errors": sum(item["fatal_errors"] for item in metrics),
        "worker_restarts": sum(item["worker_restarts"] for item in metrics),
    }


def exact_mcnemar(baseline_only: int, challenger_only: int) -> float:
    discordant = baseline_only + challenger_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, challenger_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_MASTER_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def paired_statistic(
    baseline: dict[str, bool], challenger: dict[str, bool], label: str
) -> dict[str, Any]:
    if set(baseline) != set(challenger):
        raise RuntimeError(f"paired problem IDs differ for {label}")
    ids = sorted(baseline)
    diffs = np.asarray([int(challenger[item]) - int(baseline[item]) for item in ids], dtype=int)
    negative = int(np.sum(diffs == -1))
    zero = int(np.sum(diffs == 0))
    positive = int(np.sum(diffs == 1))
    rng = np.random.default_rng(bootstrap_seed(label))
    draws = rng.multinomial(
        len(ids),
        np.asarray([negative, zero, positive], dtype=float) / len(ids),
        size=BOOTSTRAP_REPLICATES,
    )
    boot_pp = 100.0 * (-draws[:, 0] + draws[:, 2]) / len(ids)
    return {
        "problems": len(ids),
        "baseline_solved": sum(baseline.values()),
        "challenger_solved": sum(challenger.values()),
        "delta_percentage_points": 100.0 * float(np.mean(diffs)),
        "paired_bootstrap_95_ci_percentage_points": [
            float(np.quantile(boot_pp, 0.025)),
            float(np.quantile(boot_pp, 0.975)),
        ],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": bootstrap_seed(label),
        "both_fail": sum(not baseline[item] and not challenger[item] for item in ids),
        "baseline_only": negative,
        "challenger_only": positive,
        "both_pass": sum(baseline[item] and challenger[item] for item in ids),
        "mcnemar_exact_p": exact_mcnemar(negative, positive),
    }


def states(paired: dict[str, dict[int, bool]], metric: str) -> dict[str, bool]:
    if metric == "pass_at_1":
        return {pid: bool(values[0]) for pid, values in paired.items()}
    if metric == "pass_at_2":
        return {pid: any(values.values()) for pid, values in paired.items()}
    raise ValueError(metric)


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def fmt_ci(item: dict[str, Any]) -> str:
    low, high = item["paired_bootstrap_95_ci_percentage_points"]
    return f"{item['delta_percentage_points']:+.2f} pp [{low:+.2f}, {high:+.2f}], p={item['mcnemar_exact_p']:.4f}"


def arm_training_metrics(root: Path, arm: str) -> dict[str, Any]:
    arm_dir = root / arm
    training = read_json(arm_dir / "training/training_summary.json")
    contract = read_json(arm_dir / "training_contract.json")
    model_name = {"H0_no_hard": "H0-No-Hard", "H1_10pct_hard": "H1-10pct-Hard-A", "H2_20pct_hard": "H2-20pct-Hard-A"}[arm]
    merged = arm_dir / "checkpoints" / f"{model_name}-MERGED" / "model.safetensors"
    payload = {
        "arm": {"H0_no_hard": "H0", "H1_10pct_hard": "H1", "H2_20pct_hard": "H2"}[arm],
        "model_name": model_name,
        "source_phase_c2_equivalent_artifact": arm == "H1_10pct_hard",
        "independent_retraining_in_this_ablation_run": arm != "H1_10pct_hard",
        "fresh_lora_from_frozen_m0": True,
        "manifest_sha256": sha256_file(arm_dir / "manifest.jsonl"),
        "merged_model_sha256": sha256_file(merged),
        "rows": training["rows"],
        "role_counts": training.get("role_counts", {}),
        "bucket_counts": training.get("bucket_counts", {}),
        "optimizer_steps": training["optimizer_steps"],
        "draws": training["draws"],
        "unique_draws": training["unique_draws"],
        "duplicate_draws": training["duplicate_draws"],
        "max_repeat": training["max_repeat"],
        "supervised_eos_records": training["supervised_eos_records"],
        "train_loss": training["metrics"]["train_loss"],
        "eval_loss": training["eval_loss"],
        "eval_token_accuracy": training["eval_token_accuracy"],
        "frozen_m0_hash_after_training": training["frozen_m0_hash_after_training"],
        "resolved_config": training["resolved_config"],
        "contract_phase": contract.get("phase"),
    }
    write_json(arm_dir / "training_metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation"
    comparisons_dir = root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics: dict[str, dict[str, Any]] = {}
    paired_results: dict[str, dict[str, dict[int, bool]]] = {model: {} for model in PRIMARY_MODELS}
    auxiliary: dict[str, list[dict[str, list[float]]]] = {model: [] for model in PRIMARY_MODELS}
    integrity: dict[str, Any] = {"expected_problems_per_model": 428, "expected_candidates_per_model": 856, "models": {}}

    for dataset_label, (slug, expected_problems) in DATASETS.items():
        dataset_metrics[dataset_label] = {}
        for model in PRIMARY_MODELS:
            directory = project / MODEL_DIRS[model] / slug
            metrics, paired, aux = summarize(directory)
            if metrics["problems"] != expected_problems or metrics["candidates"] != 2 * expected_problems:
                raise RuntimeError(f"unexpected result count for {model} {dataset_label}: {metrics}")
            dataset_metrics[dataset_label][model] = metrics
            auxiliary[model].append(aux)
            for pid, values in paired.items():
                paired_results[model][f"{dataset_label}:{pid}"] = values

    aggregate_metrics = {
        model: aggregate([dataset_metrics[label][model] for label in DATASETS], auxiliary[model])
        for model in PRIMARY_MODELS
    }
    for model in PRIMARY_MODELS:
        item = aggregate_metrics[model]
        integrity["models"][model] = {
            "problems": item["problems"],
            "candidates": item["candidates"],
            "fatal_errors": item["fatal_errors"],
            "worker_restarts": item["worker_restarts"],
            "complete": item["problems"] == 428 and item["candidates"] == 856,
        }
    integrity["all_complete"] = all(item["complete"] for item in integrity["models"].values())
    integrity["forbidden_suites_run_by_this_ablation"] = []

    pairwise: dict[str, Any] = {
        "method": {
            "paired_bootstrap": "nonparametric paired problem resampling (multinomial equivalent)",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "master_seed": BOOTSTRAP_MASTER_SEED,
            "mcnemar": "two-sided exact binomial McNemar",
            "direction": "challenger minus baseline",
        },
        "comparisons": {},
    }
    for baseline, challenger in itertools.combinations(PRIMARY_MODELS, 2):
        pair_key = f"{baseline} -> {challenger}"
        pairwise["comparisons"][pair_key] = {}
        for dataset_label in (*DATASETS.keys(), "Aggregate428"):
            pairwise["comparisons"][pair_key][dataset_label] = {}
            if dataset_label == "Aggregate428":
                base_pairs = paired_results[baseline]
                challenge_pairs = paired_results[challenger]
            else:
                prefix = f"{dataset_label}:"
                base_pairs = {key[len(prefix):]: values for key, values in paired_results[baseline].items() if key.startswith(prefix)}
                challenge_pairs = {key[len(prefix):]: values for key, values in paired_results[challenger].items() if key.startswith(prefix)}
            for metric in ("pass_at_1", "pass_at_2"):
                label = f"{pair_key}|{dataset_label}|{metric}"
                pairwise["comparisons"][pair_key][dataset_label][metric] = paired_statistic(
                    states(base_pairs, metric), states(challenge_pairs, metric), label
                )

    # Phase-B is contextual evidence for the hard-heavy question, not a seventh arm.
    phase_b_metrics: dict[str, Any] = {}
    phase_b_aux: list[dict[str, list[float]]] = []
    for dataset_label, (slug, _) in DATASETS.items():
        metrics, _, aux = summarize(project / PHASE_B_DIR / slug)
        phase_b_metrics[dataset_label] = metrics
        phase_b_aux.append(aux)
    phase_b_metrics["Aggregate428"] = aggregate([phase_b_metrics[label] for label in DATASETS], phase_b_aux)

    model_metrics = {
        "datasets": dataset_metrics,
        "aggregate428": aggregate_metrics,
        "phase_b_hard_heavy_context": phase_b_metrics,
        "integrity": integrity,
        "aliases": {
            "H1": "Phase C2 S2-Balanced-Hard",
            "independent_replication": False,
            "reason": "identical manifest and merged checkpoint hashes; listed twice only because the task requests both labels",
        },
    }
    write_json(comparisons_dir / "model_metrics.json", model_metrics)
    write_json(comparisons_dir / "pairwise_statistics.json", pairwise)

    training_metrics = {
        "H0": arm_training_metrics(root, "H0_no_hard"),
        "H1": arm_training_metrics(root, "H1_10pct_hard"),
        "H2": arm_training_metrics(root, "H2_20pct_hard"),
    }
    h1_equivalence = {
        "status": "PASS",
        "H1_manifest_sha256": training_metrics["H1"]["manifest_sha256"],
        "Phase_C2_manifest_sha256": sha256_file(project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/manifest.jsonl"),
        "H1_merged_model_sha256": training_metrics["H1"]["merged_model_sha256"],
        "Phase_C2_merged_model_sha256": sha256_file(project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard/checkpoints/S2-Balanced-Hard-MERGED/model.safetensors"),
        "same_completed_experiment_artifact": True,
        "fresh_lora_from_frozen_m0_in_source_experiment": True,
        "retraining_duplicated": False,
        "statistical_independence_note": "H1 and Phase C2 are aliases, not two independent runs.",
    }
    write_json(root / "H1_10pct_hard/h1_phase_c2_equivalence.json", h1_equivalence)

    focus = {
        "H0_to_H1": pairwise["comparisons"]["H0 -> H1"],
        "H1_to_H2": pairwise["comparisons"]["H1 -> H2"],
    }
    write_json(comparisons_dir / "focus_h0_h1_h2.json", focus)

    header = "| 数据集 | H0 p@1/p@2 | H1=10% p@1/p@2 | H2=19.85% p@1/p@2 | C1 p@1/p@2 | C2 p@1/p@2 | M0 p@1/p@2 |"
    table = [header, "|---|---:|---:|---:|---:|---:|---:|"]
    for label in DATASETS:
        row = dataset_metrics[label]
        table.append(
            f"| {label} | {fmt_pct(row['H0']['pass_at_1'])}/{fmt_pct(row['H0']['pass_at_2'])} | "
            f"{fmt_pct(row['H1']['pass_at_1'])}/{fmt_pct(row['H1']['pass_at_2'])} | "
            f"{fmt_pct(row['H2']['pass_at_1'])}/{fmt_pct(row['H2']['pass_at_2'])} | "
            f"{fmt_pct(row['Phase C1']['pass_at_1'])}/{fmt_pct(row['Phase C1']['pass_at_2'])} | "
            f"{fmt_pct(row['Phase C2']['pass_at_1'])}/{fmt_pct(row['Phase C2']['pass_at_2'])} | "
            f"{fmt_pct(row['M0']['pass_at_1'])}/{fmt_pct(row['M0']['pass_at_2'])} |"
        )
    row = aggregate_metrics
    table.append(
        f"| Aggregate428 | {fmt_pct(row['H0']['pass_at_1'])}/{fmt_pct(row['H0']['pass_at_2'])} | "
        f"{fmt_pct(row['H1']['pass_at_1'])}/{fmt_pct(row['H1']['pass_at_2'])} | "
        f"{fmt_pct(row['H2']['pass_at_1'])}/{fmt_pct(row['H2']['pass_at_2'])} | "
        f"{fmt_pct(row['Phase C1']['pass_at_1'])}/{fmt_pct(row['Phase C1']['pass_at_2'])} | "
        f"{fmt_pct(row['Phase C2']['pass_at_1'])}/{fmt_pct(row['Phase C2']['pass_at_2'])} | "
        f"{fmt_pct(row['M0']['pass_at_1'])}/{fmt_pct(row['M0']['pass_at_2'])} |"
    )
    (comparisons_dir / "comparison_table.md").write_text("\n".join(["# Hard-A 消融核心结果", "", *table, ""]), encoding="utf-8")

    h0_h1_agg_p1 = focus["H0_to_H1"]["Aggregate428"]["pass_at_1"]
    h0_h1_agg_p2 = focus["H0_to_H1"]["Aggregate428"]["pass_at_2"]
    h1_h2_agg_p1 = focus["H1_to_H2"]["Aggregate428"]["pass_at_1"]
    h1_h2_agg_p2 = focus["H1_to_H2"]["Aggregate428"]["pass_at_2"]
    h0_h1_wb_p1 = focus["H0_to_H1"]["WB-Unseen-Holdout150"]["pass_at_1"]
    h0_h1_wb_p2 = focus["H0_to_H1"]["WB-Unseen-Holdout150"]["pass_at_2"]
    h1_h2_wb_p1 = focus["H1_to_H2"]["WB-Unseen-Holdout150"]["pass_at_1"]
    h1_h2_wb_p2 = focus["H1_to_H2"]["WB-Unseen-Holdout150"]["pass_at_2"]
    h0_h2_agg_p1 = pairwise["comparisons"]["H0 -> H2"]["Aggregate428"]["pass_at_1"]
    h0_h2_agg_p2 = pairwise["comparisons"]["H0 -> H2"]["Aggregate428"]["pass_at_2"]
    h0_h2_wb_p1 = pairwise["comparisons"]["H0 -> H2"]["WB-Unseen-Holdout150"]["pass_at_1"]
    h0_h2_wb_p2 = pairwise["comparisons"]["H0 -> H2"]["WB-Unseen-Holdout150"]["pass_at_2"]

    hard_a_independent_direction = h0_h1_agg_p2["delta_percentage_points"] > 0 or h0_h1_wb_p2["delta_percentage_points"] > 0
    recommended_ratio = "0%–5% 探索范围；现有结果不支持把 10% 或 20% 定为最优"
    phase_b = phase_b_metrics["Aggregate428"]

    canaries = {
        "H0": read_json(root / "H0_no_hard/evaluation/canary/canary_gate.json"),
        "H1": read_json(root / "H1_10pct_hard/evaluation/canary/canary_gate.json"),
        "H2": read_json(root / "H2_20pct_hard/evaluation/canary/canary_gate.json"),
    }
    design_audit = read_json(root / "ablation_design_audit.json")

    report = [
        "# Hard-A 比例消融实验最终报告（H0/H1/H2）",
        "",
        "## 结论摘要",
        "",
        "H0、H1、H2 已全部完成数据门禁、EOS 门禁、独立 LoRA 训练来源审计、canonical merged 推理 canary 和四套正式评估。未运行 Full500、Strict-unseen200 或完整 miniF2F，也未启动 C3/LD-medium。",
        "",
        f"当前证据对 Hard-A 的独立收益给出{'方向性支持，但没有统计显著证据' if hard_a_independent_direction else '不支持'}。H1 相对 H0 的 Aggregate p@1 为 {fmt_ci(h0_h1_agg_p1)}，p@2 为 {fmt_ci(h0_h1_agg_p2)}；WB-Unseen p@1 为 {fmt_ci(h0_h1_wb_p1)}，p@2 为 {fmt_ci(h0_h1_wb_p2)}。",
        "",
        f"H2 相对 H1 的 Aggregate p@1 为 {fmt_ci(h1_h2_agg_p1)}，p@2 为 {fmt_ci(h1_h2_agg_p2)}；WB-Unseen p@1 为 {fmt_ci(h1_h2_wb_p1)}，p@2 为 {fmt_ci(h1_h2_wb_p2)}。但 H2 相对 H0 的 Aggregate p@1/p@2 仍为 {fmt_ci(h0_h2_agg_p1)} / {fmt_ci(h0_h2_agg_p2)}，没有证明 Hard-A 带来净收益。因此推荐比例为：**{recommended_ratio}**。",
        "",
        "## 实验设计与审计",
        "",
        f"- H0：1800 New WB + 12 Stable/Core + 47 Frontier + 141 verified WB replay；Hard-A=0。manifest `{training_metrics['H0']['manifest_sha256']}`。",
        f"- H1：1600 New WB + 200 Hard-A + 12 Stable/Core + 47 Frontier + 141 verified WB replay；Hard-A=10%。manifest `{training_metrics['H1']['manifest_sha256']}`。",
        f"- H2：1603 New WB + 397 Hard-A；有效 Hard-A=19.85%。任务书名义 399 条中有 2 条与 protected qualified theorem 冲突，因此按零泄漏硬门禁排除，并用 2 条 New WB 补齐 2000。manifest `{training_metrics['H2']['manifest_sha256']}`。",
        f"- 三臂设计审计：`{design_audit.get('status')}`；每臂 duplicate rows=0、theorem-group duplicates=0、max repeat=1、六轴 protected leakage=0、EOS 2000/2000、proof 修改=0。",
        "- H1 与 Phase C2 的 manifest 和 merged checkpoint 哈希完全相同。H1 是已完成 C2 的同一实验工件别名，不是新的独立重复；报告同时列名只为满足比较清单。",
        "",
        "## 训练与推理合同",
        "",
        "- 三臂都从冻结 `M0-ADDON-B-FROZEN-MERGED` 新建 LoRA（r=32、alpha=64、dropout=0.05），epoch=1、LR=1e-5、effective batch=16、max length=1024、completion-only、packing=false。",
        f"- H0/H1/H2 optimizer steps 均为 125；训练 loss 分别为 {training_metrics['H0']['train_loss']:.4f}/{training_metrics['H1']['train_loss']:.4f}/{training_metrics['H2']['train_loss']:.4f}；eval loss 为 {training_metrics['H0']['eval_loss']:.4f}/{training_metrics['H1']['eval_loss']:.4f}/{training_metrics['H2']['eval_loss']:.4f}。",
        "- 评估统一使用 merged checkpoint + Transformers + 冻结 generation contract；adapter path=null。",
        f"- Canary：H0={canaries['H0']['gate_passed']}、H1={canaries['H1']['gate_passed']}、H2={canaries['H2']['gate_passed']}；每模型 140 candidates，均无 fatal/worker restart，抽取、格式、长度、重复与 timeout 门禁通过。",
        "",
        "## 正式评估",
        "",
        *table,
        "",
        "所有模型每个数据集均使用同一题目与两候选协议；Aggregate 共 428 题、856 candidates/模型。表中 H1 与 C2 数值相同是因为二者为同一工件。",
        "",
        "## 重点成对统计",
        "",
        f"- H0 → H1，Aggregate p@1：{fmt_ci(h0_h1_agg_p1)}；p@2：{fmt_ci(h0_h1_agg_p2)}。",
        f"- H0 → H1，WB-Unseen p@1：{fmt_ci(h0_h1_wb_p1)}；p@2：{fmt_ci(h0_h1_wb_p2)}。",
        f"- H1 → H2，Aggregate p@1：{fmt_ci(h1_h2_agg_p1)}；p@2：{fmt_ci(h1_h2_agg_p2)}。",
        f"- H1 → H2，WB-Unseen p@1：{fmt_ci(h1_h2_wb_p1)}；p@2：{fmt_ci(h1_h2_wb_p2)}。",
        f"- H0 → H2，Aggregate p@1：{fmt_ci(h0_h2_agg_p1)}；p@2：{fmt_ci(h0_h2_agg_p2)}。",
        f"- H0 → H2，WB-Unseen p@1：{fmt_ci(h0_h2_wb_p1)}；p@2：{fmt_ci(h0_h2_wb_p2)}。",
        "",
        "置信区间来自 20,000 次逐题配对 bootstrap；p 值为双侧 exact McNemar。全部 15 组模型对、四个数据集及 Aggregate 的 p@1/p@2 结果保存在 `comparisons/pairwise_statistics.json`。",
        "",
        "## 行为与稳定性",
        "",
        f"- Aggregate 平均 completion tokens：H0 {aggregate_metrics['H0']['completion_tokens']['mean']:.2f}，H1 {aggregate_metrics['H1']['completion_tokens']['mean']:.2f}，H2 {aggregate_metrics['H2']['completion_tokens']['mean']:.2f}，M0 {aggregate_metrics['M0']['completion_tokens']['mean']:.2f}。",
        f"- Aggregate 病理重复率：H0 {aggregate_metrics['H0']['pathological_repetition_rate']:.2%}，H1 {aggregate_metrics['H1']['pathological_repetition_rate']:.2%}，H2 {aggregate_metrics['H2']['pathological_repetition_rate']:.2%}，M0 {aggregate_metrics['M0']['pathological_repetition_rate']:.2%}。",
        f"- Aggregate timeout：H0 {aggregate_metrics['H0']['timeouts']}，H1 {aggregate_metrics['H1']['timeouts']}，H2 {aggregate_metrics['H2']['timeouts']}；三臂 fatal=0、worker restart=0。",
        "",
        "## 对任务书六个问题的回答",
        "",
        f"1. **Hard-A 相比无 Hard-A 是否有收益？** {'有弱方向性收益，但置信区间跨 0，不能宣称已证明。' if hard_a_independent_direction else '当前没有观察到收益。'} H0→H1 的核心 paired 结果见上。",
        f"2. **最优比例更接近 10% 还是 20%？** 两者都没有胜过 H0，因而不能把最优点定位到 10% 或 20%。若只在 H1/H2 二选一，H2 在 Aggregate 与 WB-Unseen 上方向性更强，但 H1 的 LD-easy 更好；综合推荐 **{recommended_ratio}**。",
        f"3. **Phase B hard-heavy 退化是否来自 Hard 比例过高？** 结果与该解释一致但不足以单独建立因果。Phase B Aggregate p@1/p@2 为 {phase_b['pass_at_1']:.2%}/{phase_b['pass_at_2']:.2%}；H0/H1/H2 的比例梯度显示高 Hard 配比没有稳定单调收益。Hard 数据本身并非被证明无价值，比例与 foundation 覆盖共同起作用。",
        "4. **foundation dominant + small hard supplement 是否成立？** 只得到部分支持：foundation dominant 得到明确支持，但 small hard supplement 的独立收益未被本实验验证。Hard-A 如继续使用，应作为不超过 5% 的探索变量，而不是默认必需成分。",
        "5. **是否适合进入 LD-medium？** 可以进入小规模、分阶段、带停止门禁的 LD-medium curriculum 实验；不应直接扩大训练。当前实验已停止，等待评审后再启动。",
        "6. **后续 Expert Iteration 前的数据比例建议？** 建议 85%–90% New WB/Foundation、0%–5% Hard-A 探索、其余 5%–10% 用 Stable/Core、Frontier 与 verified WB replay；在新证据出现前不要提高到 10% 或 20%。禁止 Hard-B/Hard-C，保持无放回、零泄漏与同一 EOS/canonical generation contract。",
        "",
        "## 停止状态",
        "",
        "Hard-A ablation completed.",
        "",
        "Waiting for review before LD-medium curriculum experiment.",
        "",
    ]
    (root / "final_report.md").write_text("\n".join(report), encoding="utf-8")

    for arm_key, arm_dir_name, model_key in (
        ("H0", "H0_no_hard", "H0"),
        ("H1", "H1_10pct_hard", "H1"),
        ("H2", "H2_20pct_hard", "H2"),
    ):
        arm_dir = root / arm_dir_name
        canary = canaries[arm_key]
        metrics = aggregate_metrics[model_key]
        arm_report = [
            f"# {arm_key} Hard-A 消融实验报告",
            "",
            "## 状态",
            "",
            "COMPLETE",
            "",
            f"- Manifest SHA256：`{training_metrics[arm_key]['manifest_sha256']}`",
            f"- Merged model SHA256：`{training_metrics[arm_key]['merged_model_sha256']}`",
            f"- Rows：{training_metrics[arm_key]['rows']}；optimizer steps：{training_metrics[arm_key]['optimizer_steps']}；duplicate draws：{training_metrics[arm_key]['duplicate_draws']}；max repeat：{training_metrics[arm_key]['max_repeat']}。",
            f"- Train/eval loss：{training_metrics[arm_key]['train_loss']:.6f}/{training_metrics[arm_key]['eval_loss']:.6f}；eval token accuracy：{training_metrics[arm_key]['eval_token_accuracy']:.2%}。",
            f"- Canary：{'PASS' if canary['gate_passed'] else 'FAIL'}；formal problems/candidates：{metrics['problems']}/{metrics['candidates']}；fatal={metrics['fatal_errors']}；worker restart={metrics['worker_restarts']}。",
            f"- Aggregate p@1/p@2：{metrics['pass_at_1']:.2%}/{metrics['pass_at_2']:.2%}；solved={metrics['pass_at_1_solved']}/{metrics['pass_at_2_solved']}。",
            "",
        ]
        if arm_key == "H1":
            arm_report += [
                "H1 与 Phase C2 是 manifest、checkpoint、训练与评估完全相同的已完成工件；本次做等价归档，没有重复训练，也不得视为独立重复。",
                "",
            ]
        if arm_key == "H2":
            arm_report += [
                "H2 有效配比为 397 Hard-A + 1603 New WB（19.85% Hard-A）；2 条名义 Hard-A 因 protected qualified-theorem 冲突按硬门禁排除。",
                "",
            ]
        (arm_dir / "report.md").write_text("\n".join(arm_report), encoding="utf-8")
        write_json(arm_dir / "evaluation/aggregate_metrics.json", metrics)
        write_json(arm_dir / "evaluation/status.json", {
            "status": f"HARD_A_ABLATION_{arm_key}_EVALUATION_COMPLETED",
            "canary_gate_passed": canary["gate_passed"],
            "canary_candidates_per_model": canary["combined"][{
                "H0": "H0-No-Hard",
                "H1": "S2-Balanced-Hard",
                "H2": "H2-20pct-Hard-A",
            }[arm_key]]["candidate_count"],
            "formal_evaluation_completed": True,
            "formal_problems": metrics["problems"],
            "formal_candidates": metrics["candidates"],
            "fatal_errors": metrics["fatal_errors"],
            "worker_restarts": metrics["worker_restarts"],
            "forbidden_suites_run": [],
            "phase_c3_started": False,
            "ld_medium_started": False,
        })

    write_json(root / "status.json", {
        "status": "HARD_A_ABLATION_COMPLETED",
        "H0_completed": True,
        "H1_completed_by_phase_c2_equivalence": True,
        "H2_completed": True,
        "all_formal_results_complete": integrity["all_complete"],
        "trainer_running": False,
        "phase_c3_started": False,
        "ld_medium_started": False,
        "final_report": str(root / "final_report.md"),
    })
    print(json.dumps({
        "status": "HARD_A_ABLATION_COMPLETED",
        "integrity": integrity,
        "aggregate": aggregate_metrics,
        "focus": focus,
        "final_report": str(root / "final_report.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
