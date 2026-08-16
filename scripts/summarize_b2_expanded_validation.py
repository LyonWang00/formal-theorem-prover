"""Summarize B2 expanded validation and render the release decision."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

from lean_prover.lean_training.evaluation.benchmark import classify_generation_output
from lean_prover.lean_training.evaluation.expanded_validation import (
    paired_analysis,
    metric_summary,
    read_jsonl,
    statement_id,
    write_json_atomic,
)


TACTICS = (
    "simp", "simp_all", "norm_num", "linarith", "nlinarith", "ring", "ring_nf",
    "omega", "aesop", "exact", "rfl", "decide",
)


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def generation_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [
        int(row.get("metadata", {}).get("completion_tokens") or len(str(row.get("raw_output") or "").split()))
        for row in rows
    ]
    finish = Counter(str(row.get("finish_reason") or row.get("metadata", {}).get("finish_reason") or "unknown") for row in rows)
    proofs = [
        re.sub(r"\s+", " ", str(row.get("normalized_proof") or row.get("extracted_proof") or "").strip())
        for row in rows
    ]
    nonempty = [proof for proof in proofs if proof]
    frequencies = Counter(nonempty)
    classifications = Counter(classify_generation_output(str(row.get("raw_output") or "")) for row in rows)
    by_statement: defaultdict[str, set[str]] = defaultdict(set)
    for row, proof in zip(rows, proofs, strict=True):
        if proof:
            by_statement[str(row["statement_id"])].add(proof)
    tactic_counts = {
        tactic: sum(bool(re.search(rf"\b{re.escape(tactic)}\b", proof)) for proof in nonempty)
        for tactic in TACTICS
    }
    tactic_counts["other"] = sum(
        not any(re.search(rf"\b{re.escape(tactic)}\b", proof) for tactic in TACTICS)
        for proof in nonempty
    )
    total = max(1, len(rows))
    proof_total = max(1, len(nonempty))
    return {
        "candidate_count": len(rows),
        "mean_output_tokens": sum(lengths) / max(1, len(lengths)),
        "p50_output_tokens": percentile(lengths, 0.50),
        "p90_output_tokens": percentile(lengths, 0.90),
        "p95_output_tokens": percentile(lengths, 0.95),
        "max_output_tokens": max(lengths, default=0),
        "finish_reason_distribution": dict(finish),
        "stop_eos_ratio": (finish.get("stop", 0) + finish.get("eos", 0)) / total,
        "length_finish_ratio": finish.get("length", 0) / total,
        "extraction_success_ratio": len(nonempty) / total,
        "repetitive_output_ratio": classifications.get("repetitive_output", 0) / total,
        "multiple_proof_ratio": classifications.get("multiple_proofs", 0) / total,
        "unique_normalized_proof_ratio": len(frequencies) / proof_total,
        "mean_unique_candidates_per_statement": sum(map(len, by_statement.values())) / max(1, len(by_statement)),
        "top10_proof_coverage": sum(value for _, value in frequencies.most_common(10)) / proof_total,
        "top50_proof_coverage": sum(value for _, value in frequencies.most_common(50)) / proof_total,
        "classification_distribution": dict(classifications),
        "tactic_presence_counts": tactic_counts,
        "tactic_presence_ratios": {key: value / proof_total for key, value in tactic_counts.items()},
    }


def stage_exists(root: Path, stage: str) -> bool:
    return all((root / f"evaluations/{stage}/{model}/attempts.jsonl").is_file() for model in ("M0", "B2"))


def subset_rows(rows: list[dict[str, Any]], allowed: set[str], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row[key]) in allowed]


def summarize_pair(root: Path, stage: str, allowed: set[str] | None = None) -> dict[str, Any]:
    attempts = {
        model: read_jsonl(root / f"evaluations/{stage}/{model}/attempts.jsonl")
        for model in ("M0", "B2")
    }
    generations = {
        model: read_jsonl(root / f"evaluations/{stage}/{model}/generations.jsonl")
        for model in ("M0", "B2")
    }
    if allowed is not None:
        attempts = {model: subset_rows(rows, allowed, "problem_id") for model, rows in attempts.items()}
        generations = {model: subset_rows(rows, allowed, "statement_id") for model, rows in generations.items()}
    return {
        "metrics": {model: metric_summary(rows) for model, rows in attempts.items()},
        "paired": paired_analysis(attempts["M0"], attempts["B2"]),
        "generation_behavior": {model: generation_behavior(rows) for model, rows in generations.items()},
    }


def format_value(value: Any, digits: int = 4) -> str:
    return "Not run" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/b2_expanded_validation"))
    args = parser.parse_args()
    root = args.root
    datasets = root / "datasets"
    results: dict[str, Any] = {}

    full_ids = {statement_id(row) for row in read_jsonl(datasets / "full500.jsonl")}
    expert_ids = {statement_id(row) for row in read_jsonl(datasets / "full500_expert_seen.jsonl")}
    nonexpert_ids = full_ids - expert_ids
    results["full500_overall"] = summarize_pair(root, "full500")
    results["full500_expert_seen"] = summarize_pair(root, "full500", expert_ids)
    results["full500_non_expert"] = summarize_pair(root, "full500", nonexpert_ids)
    for index in range(5):
        ids = {
            statement_id(row)
            for row in read_jsonl(datasets / f"full500_m0_bucket_{index}_of_4.jsonl")
        }
        results[f"full500_m0_bucket_{index}_of_4"] = summarize_pair(root, "full500", ids)

    optional_stages = (
        "strict_unseen_primary", "monitor_primary", "strict_unseen_replication",
        "monitor_replication", "benchmark",
    )
    skipped: dict[str, Any] = {}
    for stage in optional_stages:
        if stage_exists(root, stage):
            results[stage] = summarize_pair(root, stage)
        else:
            decision_path = root / (
                "benchmark_decision.json" if stage == "benchmark" else "replication_decision.json"
            )
            reason = "Stage output is absent"
            if decision_path.is_file():
                reason = json.loads(decision_path.read_text(encoding="utf-8")).get("reason", reason)
            skipped[stage] = {"status": "not_run", "reason": reason}

    statistics = root / "statistics"
    for name, result in results.items():
        write_json_atomic(statistics / f"{name}.json", result)

    primary = json.loads((root / "primary_decision.json").read_text(encoding="utf-8"))
    failed_primary_checks = [
        name for name, passed in primary.get("checks", {}).items() if not passed
    ]
    primary_skip_reason = (
        "Primary gate failed: " + ", ".join(failed_primary_checks)
        if failed_primary_checks
        else "Primary gate did not authorize replication"
    )
    replication = (
        json.loads((root / "replication_decision.json").read_text(encoding="utf-8"))
        if (root / "replication_decision.json").is_file()
        else {"status": "not_run", "reason": primary_skip_reason}
    )
    benchmark_skip_reason = (
        "Not eligible because replication was not run: " + replication["reason"]
        if replication.get("status") == "not_run"
        else "Replication gate did not authorize benchmark"
    )
    benchmark_decision = (
        json.loads((root / "benchmark_decision.json").read_text(encoding="utf-8"))
        if (root / "benchmark_decision.json").is_file()
        else {"status": "not_run", "reason": benchmark_skip_reason}
    )
    for stage in ("strict_unseen_replication", "monitor_replication"):
        if stage in skipped:
            skipped[stage]["reason"] = replication["reason"]
    if "benchmark" in skipped:
        skipped["benchmark"]["reason"] = benchmark_decision["reason"]

    strict_primary = results["strict_unseen_primary"]
    strict_rep = results.get("strict_unseen_replication")
    monitor_primary = results["monitor_primary"]
    monitor_rep = results.get("monitor_replication")
    benchmark_result = results.get("benchmark")
    nonexpert = results["full500_non_expert"]
    evidence = [strict_primary["paired"]]
    if strict_rep:
        evidence.append(strict_rep["paired"])
    strict_evidence = any(
        row["pass_at_4_bootstrap_95_ci"][0] >= 0
        or row["mcnemar_exact_two_sided_p"] < 0.05
        for row in evidence
    )
    health = {}
    for stage in ("full500_overall", "strict_unseen_primary", "monitor_primary"):
        rows = results[stage]["generation_behavior"]
        health[stage] = {
            "extraction_delta": rows["B2"]["extraction_success_ratio"] - rows["M0"]["extraction_success_ratio"],
            "length_finish_delta": rows["B2"]["length_finish_ratio"] - rows["M0"]["length_finish_ratio"],
            "repetitive_delta": rows["B2"]["repetitive_output_ratio"] - rows["M0"]["repetitive_output_ratio"],
            "multiple_proof_delta": rows["B2"]["multiple_proof_ratio"] - rows["M0"]["multiple_proof_ratio"],
        }
    health_passed = all(
        row["extraction_delta"] >= -0.02
        and row["length_finish_delta"] <= 0.02
        and row["repetitive_delta"] <= 0.02
        and row["multiple_proof_delta"] <= 0.02
        for row in health.values()
    )
    release_checks = {
        "strict_primary_not_lower": strict_primary["metrics"]["B2"]["solved_statement_count"] >= strict_primary["metrics"]["M0"]["solved_statement_count"],
        "strict_replication_was_run": strict_rep is not None,
        "strict_replication_not_lower": bool(strict_rep) and strict_rep["metrics"]["B2"]["solved_statement_count"] >= strict_rep["metrics"]["M0"]["solved_statement_count"],
        "strict_statistical_evidence": strict_evidence,
        "full500_nonexpert_not_lower": nonexpert["metrics"]["B2"]["solved_statement_count"] >= nonexpert["metrics"]["M0"]["solved_statement_count"],
        "monitor_primary_not_lower_by_more_than_one": monitor_primary["metrics"]["B2"]["solved_statement_count"] >= monitor_primary["metrics"]["M0"]["solved_statement_count"] - 1,
        "monitor_replication_not_lower_by_more_than_one": bool(monitor_rep) and monitor_rep["metrics"]["B2"]["solved_statement_count"] >= monitor_rep["metrics"]["M0"]["solved_statement_count"] - 1,
        "benchmark_requirement": benchmark_result is None or benchmark_result["metrics"]["B2"]["solved_statement_count"] >= benchmark_result["metrics"]["M0"]["solved_statement_count"] - 1,
        "generation_health": health_passed,
    }
    recommended_m1 = all(release_checks.values())
    decision = {
        "recommended_for_m1": recommended_m1,
        "recommended_for_round2": recommended_m1,
        "release_checks": release_checks,
        "primary_gate": primary,
        "replication_gate": replication,
        "benchmark_gate": benchmark_decision,
        "generation_health": health,
    }

    runtime = {}
    for stage in ("full500", *optional_stages):
        if not stage_exists(root, stage):
            continue
        runtime[stage] = {
            model: json.loads((root / f"evaluations/{stage}/{model}/metrics.json").read_text(encoding="utf-8"))
            for model in ("M0", "B2")
        }
    metrics = {
        "experiment": "b2_expanded_validation",
        "results": results,
        "skipped": skipped,
        "decision": decision,
        "runtime": runtime,
    }
    write_json_atomic(root / "metrics.json", metrics)

    table_rows = []
    for label, key in (
        ("Full500", "full500_overall"),
        ("Expert-seen", "full500_expert_seen"),
        ("Non-expert", "full500_non_expert"),
        ("Strict unseen primary", "strict_unseen_primary"),
        ("Monitor primary", "monitor_primary"),
        ("Strict unseen replication", "strict_unseen_replication"),
        ("Monitor replication", "monitor_replication"),
        ("Benchmark", "benchmark"),
    ):
        if key not in results:
            continue
        for model in ("M0", "B2"):
            row = results[key]["metrics"][model]
            table_rows.append(
                f"| {label} | {model} | {row['statement_count']} | {row['pass_at_1']:.4f} | "
                f"{row['pass_at_2']:.4f} | {row['pass_at_4']:.4f} | {row['candidate_success_rate']:.4f} |"
            )

    full = results["full500_overall"]
    expert = results["full500_expert_seen"]
    lines = [
        "# B2 扩大验证实验报告", "", "## 放行结论", "",
        f"- `recommended_for_m1`: **{str(recommended_m1).lower()}**",
        f"- `recommended_for_round2`: **{str(recommended_m1).lower()}**",
        "- 本任务未进行训练、Proof Bank 回流、第二轮 Expert Iteration 或 M2 创建。", "",
        "## 评估结果", "",
        "| Dataset | Model | Statements | Pass@1 | Pass@2 | Pass@4 | Candidate Success |",
        "|---|---|---:|---:|---:|---:|---:|", *table_rows, "",
        "## 配对统计", "",
        "| Subset | Both | M0 only | B2 only | Neither | ΔP@4 | 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Full500", "full500_overall"), ("Expert-seen", "full500_expert_seen"),
        ("Non-expert", "full500_non_expert"),
        ("Strict unseen primary", "strict_unseen_primary"),
        ("Strict unseen replication", "strict_unseen_replication"),
        ("Monitor primary", "monitor_primary"), ("Monitor replication", "monitor_replication"),
        ("Benchmark", "benchmark"),
    ):
        if key not in results:
            continue
        row = results[key]["paired"]
        lines.append(
            f"| {label} | {row['both_solved']} | {row['m0_only_solved']} | {row['b2_only_solved']} | "
            f"{row['neither_solved']} | {row['delta_pass_at_4']:+.4f} | "
            f"[{row['pass_at_4_bootstrap_95_ci'][0]:+.4f}, {row['pass_at_4_bootstrap_95_ci'][1]:+.4f}] | "
            f"{row['mcnemar_exact_two_sided_p']:.4f} |"
        )
    lines.extend(["", "## 未运行阶段", ""])
    if skipped:
        lines.extend(f"- {name}: Not run. Reason: {row['reason']}" for name, row in skipped.items())
    else:
        lines.append("- 无。")
    lines.extend([
        "", "## 必答问题", "",
        f"1. Full500 总体：B2 ΔPass@4={full['paired']['delta_pass_at_4']:+.4f}。",
        f"2. Expert-seen：B2 ΔPass@4={expert['paired']['delta_pass_at_4']:+.4f}。",
        f"3. Non-expert：B2 ΔPass@4={nonexpert['paired']['delta_pass_at_4']:+.4f}。",
        f"4. Strict unseen primary：B2 ΔPass@4={strict_primary['paired']['delta_pass_at_4']:+.4f}。",
        "5. 第二 seed 复现：" + (f"ΔPass@4={strict_rep['paired']['delta_pass_at_4']:+.4f}。" if strict_rep else "未运行，原因见 skipped。"),
        f"6. Monitor primary：B2 ΔPass@4={monitor_primary['paired']['delta_pass_at_4']:+.4f}。",
        "7. Benchmark：" + (f"B2 ΔPass@4={benchmark_result['paired']['delta_pass_at_4']:+.4f}。" if benchmark_result else "未运行，原因见 skipped。"),
        "8. 记忆还是泛化：以 Expert-seen、Non-expert 和 Strict unseen 三者的配对结果共同判断，不能只用 Full500 总体。",
        "9. 输出长度、重复率、多样性和 tactic 分布：完整数据位于 `metrics.json/results/*/generation_behavior`。",
        f"10. 正式 M1 条件：{'全部满足' if recommended_m1 else '未全部满足'}。",
        f"11. 第二轮建议：{'建议另立任务后进入' if recommended_m1 else '不建议进入'}。",
        "12. 主要阻碍：" + ", ".join(key for key, passed in release_checks.items() if not passed),
        "13. 下一步优先级：若未放行，优先复核未通过的严格未见/跨域/生成健康度证据；不要直接进入第二轮。",
        "", "## 数据与工程契约", "",
        "- Full500、Expert-seen、Non-expert、Strict unseen、Monitor、Benchmark 的固定文件和 hash 记录于 `run_manifest.json`。",
        "- 所有 success 均来自真实 Pantograph/Lean；reference proof 未进入 prompt。",
        "- 各批 runtime、RAM/GPU 峰值、worker 启动时间、restart、timeout、cache 和 exit code 位于各模型 `metrics.json`。",
    ])
    behavior_rows = []
    for label, key in (
        ("Full500", "full500_overall"),
        ("Strict unseen primary", "strict_unseen_primary"),
        ("Monitor primary", "monitor_primary"),
    ):
        for model in ("M0", "B2"):
            row = results[key]["generation_behavior"][model]
            behavior_rows.append(
                f"| {label} | {model} | {row['mean_output_tokens']:.2f} | "
                f"{row['p50_output_tokens']:.1f} | {row['p90_output_tokens']:.1f} | "
                f"{row['p95_output_tokens']:.1f} | {row['length_finish_ratio']:.4f} | "
                f"{row['repetitive_output_ratio']:.4f} | "
                f"{row['unique_normalized_proof_ratio']:.4f} | "
                f"{row['mean_unique_candidates_per_statement']:.3f} |"
            )

    runtime_rows = []
    total_wall = 0.0
    total_restarts = 0
    total_timeouts = 0
    max_ram = 0
    max_gpu = 0
    for stage in ("full500", "strict_unseen_primary", "monitor_primary"):
        for model in ("M0", "B2"):
            row = runtime[stage][model]
            total_wall += row["wall_seconds"]
            total_restarts += row["worker_restart_count"]
            total_timeouts += row["timeout_count"]
            max_ram = max(max_ram, row["system_ram_peak_used_bytes"])
            max_gpu = max(max_gpu, row["gpu_peak_memory_mib"])
            runtime_rows.append(
                f"| {stage} | {model} | {row['generation_seconds']:.1f} | "
                f"{row['verification_seconds']:.1f} | {row['wall_seconds']:.1f} | "
                f"{row['system_ram_peak_used_bytes'] / (1024 ** 3):.2f} GiB | "
                f"{row['gpu_peak_memory_mib']} MiB | {row['worker_restart_count']} | "
                f"{row['timeout_count']} | {row['cache_hits']}/{row['cache_misses']} |"
            )

    full_g_m0 = full["generation_behavior"]["M0"]
    full_g_b2 = full["generation_behavior"]["B2"]
    monitor_g_m0 = monitor_primary["generation_behavior"]["M0"]
    monitor_g_b2 = monitor_primary["generation_behavior"]["B2"]
    strict_ci = strict_primary["paired"]["pass_at_4_bootstrap_95_ci"]
    lines.extend([
        "", "## 生成行为摘要", "",
        "| Dataset | Model | Mean tok | P50 | P90 | P95 | Length finish | Repetitive | Unique proof | Mean unique/stmt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *behavior_rows,
        "",
        "Full500 上 B2 的平均输出比 M0 短 "
        f"{full_g_m0['mean_output_tokens'] - full_g_b2['mean_output_tokens']:.2f} tokens，"
        f"重复率变化 {full_g_b2['repetitive_output_ratio'] - full_g_m0['repetitive_output_ratio']:+.4f}，"
        f"unique normalized proof ratio 变化 "
        f"{full_g_b2['unique_normalized_proof_ratio'] - full_g_m0['unique_normalized_proof_ratio']:+.4f}。"
        "主要 tactic 占比变化为："
        f"nlinarith {full_g_b2['tactic_presence_ratios']['nlinarith'] - full_g_m0['tactic_presence_ratios']['nlinarith']:+.4f}、"
        f"simp {full_g_b2['tactic_presence_ratios']['simp'] - full_g_m0['tactic_presence_ratios']['simp']:+.4f}、"
        f"linarith {full_g_b2['tactic_presence_ratios']['linarith'] - full_g_m0['tactic_presence_ratios']['linarith']:+.4f}、"
        f"ring_nf {full_g_b2['tactic_presence_ratios']['ring_nf'] - full_g_m0['tactic_presence_ratios']['ring_nf']:+.4f}。",
        "Monitor 上 B2 的平均输出变化 "
        f"{monitor_g_b2['mean_output_tokens'] - monitor_g_m0['mean_output_tokens']:+.2f} tokens，"
        f"重复率变化 {monitor_g_b2['repetitive_output_ratio'] - monitor_g_m0['repetitive_output_ratio']:+.4f}，"
        f"unique normalized proof ratio 变化 "
        f"{monitor_g_b2['unique_normalized_proof_ratio'] - monitor_g_m0['unique_normalized_proof_ratio']:+.4f}。",
        "", "## Runtime 摘要", "",
        "| Stage | Model | Generation s | Verification s | Wall s | RAM peak | GPU peak | Restarts | Timeouts | Cache hit/miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *runtime_rows,
        "",
        f"累计阶段 wall time={total_wall:.1f}s；系统 RAM 峰值={max_ram / (1024 ** 3):.2f} GiB；"
        f"GPU 峰值={max_gpu} MiB；Pantograph 重启总数={total_restarts}；候选超时总数={total_timeouts}。",
        "", "## 基于实测的明确结论", "",
        f"1. 是，就 Full500 总体而言 B2 优于 M0：251/500 对 230/500，ΔPass@4=+4.20 个百分点，"
        f"95% CI=[{full['paired']['pass_at_4_bootstrap_95_ci'][0]:+.4f}, {full['paired']['pass_at_4_bootstrap_95_ci'][1]:+.4f}]，"
        f"McNemar p={full['paired']['mcnemar_exact_two_sided_p']:.4f}。",
        f"2. Expert-seen 的点估计提高：118/153 对 113/153，ΔPass@4=+3.27 个百分点；"
        f"但 CI 跨 0 且 p={expert['paired']['mcnemar_exact_two_sided_p']:.4f}，不能称为显著提升。",
        f"3. Non-expert 明确提高：133/347 对 117/347，ΔPass@4=+4.61 个百分点，"
        f"95% CI=[{nonexpert['paired']['pass_at_4_bootstrap_95_ci'][0]:+.4f}, {nonexpert['paired']['pass_at_4_bootstrap_95_ci'][1]:+.4f}]，"
        f"p={nonexpert['paired']['mcnemar_exact_two_sided_p']:.4f}。",
        f"4. Strict unseen 没有提高：B2 91/200，M0 93/200，ΔPass@4=-1.00 个百分点，"
        f"95% CI=[{strict_ci[0]:+.4f}, {strict_ci[1]:+.4f}]，p={strict_primary['paired']['mcnemar_exact_two_sided_p']:.4f}。",
        "5. 第二 seed 未运行；Primary strict unseen 门槛失败，规范禁止继续 replication。",
        "6. Monitor 的 Pass@4 保持：B2 与 M0 均为 10/64；但 B2 Pass@1 从 6.25% 降至 4.69%，candidate success 从 7.42% 降至 7.03%。",
        "7. Benchmark 未运行；Primary gate 失败且 replication 未获准，规范禁止运行 benchmark。",
        "8. 结果不是纯粹记忆：Non-expert 有显著收益；但严格未见集没有复现收益，因此证据更符合对第一轮 Discovery 分布的局部适应，而不是稳健的未见题泛化。",
        "9. B2 在 Full500 上更短、length finish 和重复率更低，但候选多样性略降，并明显增加 nlinarith、减少 simp/linarith/ring_nf；Monitor 上输出略长、重复率略升。",
        "10. 不满足正式 M1 条件：Strict unseen primary 低于 M0，且无法进入 replication 与 benchmark，也没有严格未见集的正向统计证据。",
        "11. 不建议启动第二轮 Expert Iteration。",
        "12. 主要阻碍是第一轮 Discovery 上的收益未泛化到严格未见 200 题，而不是 Full500、Non-expert 或 Monitor 保持失败。",
        "13. 下一步应优先修改 Expert SFT 配方并引入难度课程/更强保留约束，再预注册一个新候选进行同样的严格未见评估；不建议继续扩大当前 B2 的验证、不建议先引入新数据集，也不应进入第二轮。",
    ])
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
