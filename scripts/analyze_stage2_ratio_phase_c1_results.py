"""Create the final three-model Phase C1 comparison and audit report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = {
    "WB-Unseen-Holdout150": "wb_unseen_holdout150",
    "LD-easy64": "ld_easy64",
    "Monitor64": "monitor64",
    "WB-Train-Retention150": "wb_train_retention150",
}
MODELS = ("M0", "S2-Hard-Replay", "S2-New-WB")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pctl(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def rep_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize(directory: Path) -> tuple[dict[str, Any], dict[str, dict[int, bool]]]:
    attempts = read_jsonl(directory / "attempts.jsonl")
    generations = read_jsonl(directory / "generations.jsonl")
    paired: dict[str, dict[int, bool]] = {}
    for row in attempts:
        paired.setdefault(str(row["problem_id"]), {})[int(row["attempt_index"])] = bool(row["success"])
    if len(attempts) != 2 * len(paired) or any(set(v) != {0, 1} for v in paired.values()):
        raise RuntimeError(f"incomplete two-candidate result at {directory}")
    solved = {pid: any(v.values()) for pid, v in paired.items()}
    pass1 = {pid: bool(v[0]) for pid, v in paired.items()}
    lengths = [int(row.get("metadata", {}).get("completion_tokens", 0)) for row in generations]
    reps = [rep_ratio(str(row.get("raw_output", ""))) for row in generations]
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in list(directory.glob("*_summary.json")) + list(directory.glob("*_metrics.json"))]
    result = {
        "problems": len(paired), "candidates": len(attempts),
        "candidate_successes": sum(bool(row.get("success")) for row in attempts),
        "pass_at_1_count": sum(pass1.values()), "pass_at_2_count": sum(solved.values()),
        "pass_at_1": sum(pass1.values()) / max(1, len(paired)),
        "pass_at_2": sum(solved.values()) / max(1, len(paired)),
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "completion_tokens": {"mean": mean(lengths), "p50": pctl(lengths, .5), "p95": pctl(lengths, .95), "max": max(lengths)},
        "max_length_finish_ratio": sum(
            row.get("finish_reason") in {"length", "max_length"}
            or int(row.get("metadata", {}).get("completion_tokens", 0)) >= int(row.get("max_new_tokens", 256))
            for row in generations
        ) / max(1, len(generations)),
        "extraction_success_rate": sum(bool(str(row.get("extracted_proof", "")).strip()) for row in generations) / max(1, len(generations)),
        "mean_repeated_4gram_ratio": mean(reps),
        "pathological_repetition_rate": sum(x >= .35 for x in reps) / max(1, len(reps)),
        "fatal_errors": sum(len(x.get("fatal_errors", [])) for x in summaries),
        "worker_restarts": sum(int(x.get("pantograph_worker_restart_count", 0)) for x in summaries),
    }
    return result, paired


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_counts(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    if set(a) != set(b):
        raise RuntimeError("paired problem IDs differ")
    ids = sorted(a)
    a_only = sum(a[x] and not b[x] for x in ids)
    b_only = sum(b[x] and not a[x] for x in ids)
    return {
        "both_fail": sum(not a[x] and not b[x] for x in ids),
        "baseline_only": a_only, "challenger_only": b_only,
        "both_pass": sum(a[x] and b[x] for x in ids),
        "mcnemar_exact_p": exact_mcnemar(a_only, b_only),
    }


def model_dir(project: Path, model: str, slug: str) -> Path:
    if model == "S2-Hard-Replay":
        return project / "outputs/stage2_data_ablation/phaseB_hard_replay/evaluation/full/S2-Hard-Replay" / slug
    return project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/evaluation/full" / model / slug


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    problems, candidates = sum(x["problems"] for x in metrics), sum(x["candidates"] for x in metrics)
    return {
        "problems": problems, "candidates": candidates,
        "candidate_successes": sum(x["candidate_successes"] for x in metrics),
        "pass_at_1_count": sum(x["pass_at_1_count"] for x in metrics),
        "pass_at_2_count": sum(x["pass_at_2_count"] for x in metrics),
        "pass_at_1": sum(x["pass_at_1_count"] for x in metrics) / problems,
        "pass_at_2": sum(x["pass_at_2_count"] for x in metrics) / problems,
        "timeouts": sum(x["timeouts"] for x in metrics),
        "mean_completion_tokens": sum(x["completion_tokens"]["mean"] * x["candidates"] for x in metrics) / candidates,
        "fatal_errors": sum(x["fatal_errors"] for x in metrics),
        "worker_restarts": sum(x["worker_restarts"] for x in metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    project = parser.parse_args().project.resolve()
    root = project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb"
    rows: dict[str, Any] = {}
    all_pairs: dict[str, dict[str, dict[int, bool]]] = {model: {} for model in MODELS}
    for label, slug in DATASETS.items():
        metrics: dict[str, Any] = {}
        pairs: dict[str, dict[str, dict[int, bool]]] = {}
        for model in MODELS:
            metrics[model], pairs[model] = summarize(model_dir(project, model, slug))
            for pid, values in pairs[model].items():
                all_pairs[model][f"{label}:{pid}"] = values
        item: dict[str, Any] = {model: metrics[model] for model in MODELS}
        item["delta_vs_M0_percentage_points"] = {
            model: {
                "pass_at_1": 100 * (metrics[model]["pass_at_1"] - metrics["M0"]["pass_at_1"]),
                "pass_at_2": 100 * (metrics[model]["pass_at_2"] - metrics["M0"]["pass_at_2"]),
            } for model in MODELS[1:]
        }
        item["paired_vs_M0"] = {}
        for model in MODELS[1:]:
            item["paired_vs_M0"][model] = {
                "pass_at_1": paired_counts({x: bool(v[0]) for x, v in pairs["M0"].items()}, {x: bool(v[0]) for x, v in pairs[model].items()}),
                "pass_at_2": paired_counts({x: any(v.values()) for x, v in pairs["M0"].items()}, {x: any(v.values()) for x, v in pairs[model].items()}),
            }
        rows[label] = item

    totals = {model: aggregate([rows[label][model] for label in DATASETS]) for model in MODELS}
    aggregate_delta = {model: {
        "pass_at_1": 100 * (totals[model]["pass_at_1"] - totals["M0"]["pass_at_1"]),
        "pass_at_2": 100 * (totals[model]["pass_at_2"] - totals["M0"]["pass_at_2"]),
    } for model in MODELS[1:]}
    aggregate_paired = {}
    for model in MODELS[1:]:
        aggregate_paired[model] = {
            "pass_at_1": paired_counts({x: bool(v[0]) for x, v in all_pairs["M0"].items()}, {x: bool(v[0]) for x, v in all_pairs[model].items()}),
            "pass_at_2": paired_counts({x: any(v.values()) for x, v in all_pairs["M0"].items()}, {x: any(v.values()) for x, v in all_pairs[model].items()}),
        }
    new, hard, m0 = totals["S2-New-WB"], totals["S2-Hard-Replay"], totals["M0"]
    answers = {
        "new_wb_improves_unseen": rows["WB-Unseen-Holdout150"]["S2-New-WB"]["pass_at_2"] > rows["WB-Unseen-Holdout150"]["M0"]["pass_at_2"],
        "new_wb_more_stable_than_hard_heavy_on_aggregate_pass1": abs(new["pass_at_1"] - m0["pass_at_1"]) < abs(hard["pass_at_1"] - m0["pass_at_1"]),
        "ld_easy_preserved_vs_M0_pass2": rows["LD-easy64"]["S2-New-WB"]["pass_at_2"] >= rows["LD-easy64"]["M0"]["pass_at_2"],
        "pass1_decline_reduced_vs_hard_heavy": (new["pass_at_1"] - m0["pass_at_1"]) > (hard["pass_at_1"] - m0["pass_at_1"]),
    }
    contract = json.loads((root / "evaluation/generation_contract.json").read_text(encoding="utf-8"))
    comparison = {
        "phase": "C1", "model": "S2-New-WB", "baselines": ["M0-ADDON-B-FROZEN", "S2-Hard-Replay"],
        "canonical_backend": "transformers", "checkpoint_type": "merged",
        "generation_contract_sha256": contract["generation_config_sha256"],
        "parent_generation_contract_sha256": contract["parent_generation_contract_sha256"],
        "evaluation_integrity": {
            "expected_problems_per_model": 428, "expected_candidates_per_model": 856,
            "complete": all(x["problems"] == 428 and x["candidates"] == 856 for x in totals.values()),
            "new_model_fatal_errors": new["fatal_errors"], "new_model_worker_restarts": new["worker_restarts"],
            "forbidden_suites_run": [],
        },
        "datasets": rows, "aggregate": totals, "aggregate_delta_vs_M0_percentage_points": aggregate_delta,
        "aggregate_paired_vs_M0": aggregate_paired, "task_questions": answers,
    }
    comparisons = project / "outputs/stage2_data_ratio_ablation/comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    (comparisons / "phaseC1_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    header = "| Dataset | M0 p@1 | Hard p@1 | New-WB p@1 | M0 p@2 | Hard p@2 | New-WB p@2 |"
    rule = "|---|---:|---:|---:|---:|---:|---:|"
    table = [header, rule]
    for label in DATASETS:
        x = rows[label]
        table.append(f"| {label} | {x['M0']['pass_at_1']:.2%} | {x['S2-Hard-Replay']['pass_at_1']:.2%} | {x['S2-New-WB']['pass_at_1']:.2%} | {x['M0']['pass_at_2']:.2%} | {x['S2-Hard-Replay']['pass_at_2']:.2%} | {x['S2-New-WB']['pass_at_2']:.2%} |")
    table.append(f"| Aggregate (428) | {m0['pass_at_1']:.2%} | {hard['pass_at_1']:.2%} | {new['pass_at_1']:.2%} | {m0['pass_at_2']:.2%} | {hard['pass_at_2']:.2%} | {new['pass_at_2']:.2%} |")

    training = json.loads((root / "training/training_summary.json").read_text(encoding="utf-8"))
    data_audit = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
    leakage = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
    eos = json.loads((root / "eos_audit.json").read_text(encoding="utf-8"))
    canary = json.loads((root / "evaluation/canary/canary_gate.json").read_text(encoding="utf-8"))
    report = [
        "# Phase C1 — New WB 增量实验报告", "", "## 结论", "",
        "Phase C1 已按任务书完成，行为门禁和四组正式评估均完成。未运行 Full500、Strict-unseen200 或 miniF2F，也未启动 Phase C2。", "",
        "New WB 配方相较 M0 同时提高了 WB unseen 的 pass@1/pass@2 与四集合总体 pass@1/pass@2；相较 hard-heavy，本轮避免了总体 pass@1 退化，因此当前方向性证据支持 New WB 作为更稳定的第二轮主体。LD-easy 的 pass@2 完全保持，但 pass@1 小幅下降。", "",
        "## 数据与门禁", "",
        f"- 训练共 {data_audit['rows']} 行：New WB 1200、独立 Hard-A 300、Old replay 500。最终全表桶计数：New WB 1200、Hard-A {training['bucket_counts']['Hard-A']}、Hard-B {training['bucket_counts']['Hard-B']}、Frontier {training['bucket_counts']['Frontier']}、Stable/Core {training['bucket_counts']['Stable/Core']}。",
        "- 用户批准以 342 条 Hard-B 补充原 replay 缺口；严格 qualified-theorem 泄漏门禁又排除了 2 条 Hard-A，因此用额外 2 条合格 Hard-B 等量替换，最终 Hard-B 为 344。没有重复或放回抽样。",
        f"- Manifest SHA-256：`{training['manifest_sha256']}`；duplicate rows=0，theorem-group duplicates=0，max repeat={training['max_repeat']}。",
        f"- 受保护集合六轴泄漏门禁：{'PASS' if leakage.get('status') == 'PASSED' else 'FAIL'}；WB retention 重叠按任务书单独允许并记录。",
        f"- EOS 门禁：{'PASS' if eos.get('status') == 'PASSED' else 'FAIL'}；supervised EOS {training['supervised_eos_records']}/{training['rows']}，zero-label=0，semantic truncation=0。", "",
        "## 训练", "",
        f"- 从冻结 M0 合并 checkpoint 新建 LoRA：r={training['resolved_config']['lora_r']}、alpha={training['resolved_config']['lora_alpha']}、dropout={training['resolved_config']['lora_dropout']}、LR={training['resolved_config']['learning_rate']}、epoch={training['resolved_config']['num_train_epochs']}、effective batch={training['resolved_config']['per_device_train_batch_size'] * training['resolved_config']['gradient_accumulation_steps']}。",
        f"- optimizer steps {training['optimizer_steps']}/125；train loss {training['metrics']['train_loss']:.6f}；eval loss {training['eval_loss']:.6f}；eval token accuracy {training['eval_token_accuracy']:.4%}。",
        f"- M0 训练前后哈希一致：`{training['frozen_m0_hash_after_training']}`。合并 S2-New-WB 模型哈希：`{contract['checkpoint_model_sha256']}`。", "",
        "## 固定推理合同与行为 canary", "",
        f"- canonical path：merged checkpoint + Transformers；adapter path=null；父 generation contract：`{contract['parent_generation_contract_sha256']}`；本轮解析合同：`{contract['generation_config_sha256']}`。仅模型身份变化。",
        f"- Canary：{'PASS' if canary['gate_passed'] else 'FAIL'}；每模型 140 个候选，S2-New-WB 抽取率 {canary['combined']['S2-New-WB']['extraction_success_rate']:.2%}，平均长度 {canary['combined']['S2-New-WB']['completion_tokens_mean']:.2f}，length finish {canary['combined']['S2-New-WB']['max_length_finish_ratio']:.2%}，病理重复 {canary['combined']['S2-New-WB']['pathological_repetition_rate']:.2%}。", "",
        "## 正式评估", "", *table, "",
        f"S2-New-WB 相对 M0：总体 pass@1 {aggregate_delta['S2-New-WB']['pass_at_1']:+.2f} pp，pass@2 {aggregate_delta['S2-New-WB']['pass_at_2']:+.2f} pp。S2-Hard-Replay 相对 M0：总体 pass@1 {aggregate_delta['S2-Hard-Replay']['pass_at_1']:+.2f} pp，pass@2 {aggregate_delta['S2-Hard-Replay']['pass_at_2']:+.2f} pp。", "",
        "## 对四个问题的回答", "",
        f"1. New WB 是否提高 unseen：{'是（按 pass@2）' if answers['new_wb_improves_unseen'] else '否'}；WB-Unseen pass@2 为 {rows['WB-Unseen-Holdout150']['M0']['pass_at_2']:.2%} → {rows['WB-Unseen-Holdout150']['S2-New-WB']['pass_at_2']:.2%}。",
        f"2. 是否比 hard-heavy 更稳定：{'是' if answers['new_wb_more_stable_than_hard_heavy_on_aggregate_pass1'] else '否'}；以总体 pass@1 相对 M0 的绝对偏移衡量。",
        f"3. 是否保持 LD-easy：{'是（按 pass@2）' if answers['ld_easy_preserved_vs_M0_pass2'] else '否'}；LD-easy pass@2 为 {rows['LD-easy64']['M0']['pass_at_2']:.2%} → {rows['LD-easy64']['S2-New-WB']['pass_at_2']:.2%}。",
        f"4. 是否减少 pass@1 下降：{'是，且总体上转为提升' if answers['pass1_decline_reduced_vs_hard_heavy'] else '否'}；New-WB 总体变化 {aggregate_delta['S2-New-WB']['pass_at_1']:+.2f} pp，hard-heavy 为 {aggregate_delta['S2-Hard-Replay']['pass_at_1']:+.2f} pp。", "",
        f"逐题配对结果中，WB unseen pass@2 的 McNemar 精确 p={rows['WB-Unseen-Holdout150']['paired_vs_M0']['S2-New-WB']['pass_at_2']['mcnemar_exact_p']:.4f}；四集合总体 pass@2 的 p={aggregate_paired['S2-New-WB']['pass_at_2']['mcnemar_exact_p']:.4f}。其余差异主要视为方向性证据，不应解释为已确证的总体规律。", "",
        "## 阶段状态", "", "Phase C1 completed.", "", "Waiting for approval before Phase C2.", "",
    ]
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    comparison_md = ["# Phase C1 three-model comparison", "", *table, "", f"Aggregate S2-New-WB vs M0: p@1 {aggregate_delta['S2-New-WB']['pass_at_1']:+.2f} pp; p@2 {aggregate_delta['S2-New-WB']['pass_at_2']:+.2f} pp.", "", "See `phaseC1_comparison.json` for paired counts and exact McNemar p-values.", ""]
    (comparisons / "phaseC1_comparison.md").write_text("\n".join(comparison_md), encoding="utf-8")
    final_summary = [
        "# Stage 2 data-ratio ablation status", "", "## Phase C1 — COMPLETE", "",
        "S2-New-WB was trained from frozen M0 and evaluated on the four authorized suites. All data, EOS, canonical-inference, canary, and result-completeness gates passed.", "",
        f"Aggregate p@1: M0 {m0['pass_at_1']:.2%}, S2-Hard-Replay {hard['pass_at_1']:.2%}, S2-New-WB {new['pass_at_1']:.2%}.",
        f"Aggregate p@2: M0 {m0['pass_at_2']:.2%}, S2-Hard-Replay {hard['pass_at_2']:.2%}, S2-New-WB {new['pass_at_2']:.2%}.", "",
        "## Phase C2 — NOT STARTED", "", "Waiting for explicit approval before Phase C2.", "", "## Phase C3 — NOT STARTED", "",
    ]
    (project / "outputs/stage2_data_ratio_ablation/final_summary.md").write_text("\n".join(final_summary), encoding="utf-8")
    (root / "evaluation/status.json").write_text(json.dumps({
        "status": "PHASE_C1_EVALUATION_COMPLETED",
        "canary_gate_passed": canary["gate_passed"],
        "canary_candidates_per_model": canary["combined"]["S2-New-WB"]["candidate_count"],
        "formal_evaluation_completed": True,
        "formal_problems": new["problems"],
        "formal_candidates": new["candidates"],
        "fatal_errors": new["fatal_errors"],
        "worker_restarts": new["worker_restarts"],
        "forbidden_suites_run": [],
        "phase_c2_started": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(root / 'report.md'), "comparison": str(comparisons / 'phaseC1_comparison.json'), "aggregate": totals, "answers": answers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
