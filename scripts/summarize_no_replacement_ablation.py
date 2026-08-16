"""Summarize the fixed-manifest B/C Expert SFT ablation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import comb
import json
from pathlib import Path
import random
from typing import Any

from compare_generation_drift import summarize as summarize_generations


ARMS = ("M0", "C0", "B1", "B2", "C1", "C2", "C3")
TRAIN_DIRS = {
    "B1": "B1_anchor_1000",
    "B2": "B2_anchor_expert_1000",
    "C1": "C1_anchor_micro",
    "C2": "C2_expert_micro",
    "C3": "C3_expert_anchor_micro",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def statement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: defaultdict[str, int] = defaultdict(int)
    seen: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        key = str(row["problem_id"])
        seen[key] += 1
        grouped[key] += int(bool(row["success"]))
    if len(grouped) == 0 or any(count != 4 for count in seen.values()):
        raise ValueError("each fixed-gate statement must have exactly four attempts")
    return dict(grouped)


def gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = statement_counts(rows)
    first = {str(row["problem_id"]): bool(row["success"]) for row in rows if int(row["attempt_index"]) == 0}
    successes = sum(counts.values())
    return {
        "statements": len(counts),
        "candidates": len(rows),
        "verified_candidates": successes,
        "candidate_success_rate": successes / len(rows),
        "solved_statements": sum(value > 0 for value in counts.values()),
        "pass_at_1": sum(first.values()) / len(counts),
        "pass_at_4": sum(value > 0 for value in counts.values()) / len(counts),
        "success_at_4_distribution": {str(index): sum(value == index for value in counts.values()) for index in range(5)},
        "timeout_attempts": sum(bool(row.get("timed_out")) for row in rows),
        "success_count_by_statement": counts,
    }


def mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    return min(1.0, 2.0 * sum(comb(discordant, value) for value in range(lower + 1)) / (2**discordant))


def paired(left: dict[str, int], right: dict[str, int], seed: int = 20260721) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    differences = [int(right[key] > 0) - int(left[key] > 0) for key in ids]
    both = sum(left[key] > 0 and right[key] > 0 for key in ids)
    left_only = sum(left[key] > 0 and right[key] == 0 for key in ids)
    right_only = sum(left[key] == 0 and right[key] > 0 for key in ids)
    rng = random.Random(seed)
    draws = 10_000
    estimates = sorted(
        sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
        for _ in range(draws)
    )
    return {
        "statements": len(ids),
        "both_solved": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": len(ids) - both - left_only - right_only,
        "delta_pass_at_4": sum(differences) / len(ids),
        "bootstrap_95_ci": [estimates[249], estimates[9749]],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "mcnemar_exact_two_sided_p": mcnemar_exact(left_only, right_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/expert_sft_no_replacement_ablation"))
    parser.add_argument("--round1", type=Path, default=Path("outputs/expert_iteration_round1"))
    args = parser.parse_args()
    root = args.root
    release = read_json(root / "release_gate_results.json")
    selection = read_json(root / "checkpoint_selection.json")
    manifest = read_json(root / "manifests/experiment_manifest.json")
    c0 = read_json(root / "evaluations/C0_control/c0_gate_report.json")

    discovery = {
        arm: gate_summary(read_jsonl(root / f"evaluations/discovery_gate/{arm}/attempts.jsonl"))
        for arm in ARMS
    }
    monitor = {
        arm: gate_summary(read_jsonl(root / f"evaluations/monitor/{arm}/attempts.jsonl"))
        for arm in ("M0", "C0", "B2", "B1")
    }
    behavior = {
        arm: summarize_generations(read_jsonl(root / f"evaluations/discovery_gate/{arm}/generations.jsonl"))
        for arm in ARMS
    }

    comparisons = {}
    for arm in ARMS[1:]:
        comparisons[f"M0_vs_{arm}"] = paired(
            discovery["M0"]["success_count_by_statement"],
            discovery[arm]["success_count_by_statement"],
        )
    for left, right in (("B1", "B2"), ("C1", "C2"), ("C1", "C3"), ("C2", "C3")):
        comparisons[f"{left}_vs_{right}"] = paired(
            discovery[left]["success_count_by_statement"],
            discovery[right]["success_count_by_statement"],
        )

    training: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    for arm, directory in TRAIN_DIRS.items():
        training[arm] = read_json(root / f"checkpoints/{directory}/training_metrics.json")
        traces[arm] = read_json(root / f"traces/{arm}_sampling_trace.json")

    previous_eval = read_json(args.round1 / "evaluation/eval_loss.json")
    m0_eval = previous_eval["M0"]
    evaluation = {
        "M0": {"eval_loss": m0_eval["eval_loss"], "eval_token_accuracy": m0_eval["eval_token_accuracy"]},
        "C0": {"eval_loss": m0_eval["eval_loss"], "eval_token_accuracy": m0_eval["eval_token_accuracy"], "reason": "bit-identical merged model hash"},
    }
    for arm, row in training.items():
        evaluation[arm] = {"eval_loss": row.get("eval_loss"), "eval_token_accuracy": row.get("eval_token_accuracy")}

    runtime_rows = [*release["discovery_gate"].values(), *release["monitor"].values()]
    runtime = {
        "evaluation_generation_seconds": sum(float(row["generation_seconds"]) for row in runtime_rows),
        "evaluation_verification_seconds": sum(float(row["verification_seconds"]) for row in runtime_rows),
        "evaluation_total_seconds": sum(float(row["total_seconds"]) for row in runtime_rows),
        "pantograph_worker_restarts": sum(int(row["pantograph_worker_restart_count"]) for row in runtime_rows),
        "verification_timeouts": sum(int(row["timeout_attempts"]) for row in runtime_rows),
        "cache_hits": sum(int(row["cache_hits"]) for row in runtime_rows),
        "cache_misses": sum(int(row["cache_misses"]) for row in runtime_rows),
        "training_runtime_seconds": {arm: row.get("train_runtime") or row.get("training_runtime_seconds") for arm, row in training.items()},
        "known_incidents": ["B1 step_5 Early Gate first attempt hit transient cudaErrorUnknown; health check passed and the fixed retry completed."],
    }
    metrics = {
        "experiment": "fixed_manifest_without_replacement_BC_ablation",
        "manifest": manifest,
        "checkpoint_selection": selection,
        "zero_step_control": c0,
        "sampling_traces": traces,
        "training": training,
        "eval": evaluation,
        "discovery_gate": discovery,
        "monitor": monitor,
        "generation_behavior": behavior,
        "paired_discovery_gate": comparisons,
        "runtime": runtime,
        "full_replay": {"status": "not_run", "reason": "No trained model reached 98/150 on Discovery Gate."},
        "benchmark": {"status": "not_run", "reason": "Release gates, including Full Replay, were not all met."},
    }
    atomic_json(root / "metrics.json", metrics)

    lines = [
        "# 无放回 Expert SFT B/C 消融实验报告", "",
        "## 结论", "",
        "固定 manifest 与 DataLoader 无放回契约已经实现并由 trace 验证：每个训练臂的物理记录只抽取一次，duplicate draw 为 0，最大重复次数为 1。公平重跑的固定 150 题 Gate 上，B2 最优（86/150），较 M0（76/150）提高 10 题；Monitor 为 12/64，与 M0 持平。但 B2 未达到预设的 98/150 Full Replay 门槛，因此不能将它发布为新 M1，也没有运行 Full Replay 或 benchmark。", "",
        "## 模型对比", "",
        "| Model | 数据 | 最佳 step | Eval loss | Eval token acc | Discovery P@1 | Discovery P@4 | ΔP@4 vs M0 | Monitor P@4 |", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    descriptions = {"M0":"baseline", "C0":"zero-step", "B1":"1000 anchor", "B2":"847 anchor + 153 expert", "C1":"160 anchor", "C2":"153 expert", "C3":"39 anchor + 153 expert"}
    for arm in ARMS:
        selected = "-" if arm in ("M0", "C0") else str(selection["arms"][arm]["best_step"])
        mon = monitor.get(arm)
        lines.append(
            f"| {arm} | {descriptions[arm]} | {selected} | {evaluation[arm]['eval_loss']:.6f} | {evaluation[arm]['eval_token_accuracy']:.6f} | "
            f"{discovery[arm]['pass_at_1']:.4f} | {discovery[arm]['pass_at_4']:.4f} | {discovery[arm]['pass_at_4']-discovery['M0']['pass_at_4']:+.4f} | "
            f"{mon['pass_at_4']:.4f}" if mon else
            f"| {arm} | {descriptions[arm]} | {selected} | {evaluation[arm]['eval_loss']:.6f} | {evaluation[arm]['eval_token_accuracy']:.6f} | {discovery[arm]['pass_at_1']:.4f} | {discovery[arm]['pass_at_4']:.4f} | {discovery[arm]['pass_at_4']-discovery['M0']['pass_at_4']:+.4f} | 未运行 |"
        )
        if mon:
            lines[-1] += " |"

    lines.extend(["", "## 无放回契约", "", "| Arm | Rows/draws | Unique | Duplicate draws | Max repeat | Anchor/Expert draws |", "|---|---:|---:|---:|---:|---:|"])
    for arm in TRAIN_DIRS:
        trace = traces[arm]
        lines.append(f"| {arm} | {trace['physical_rows']}/{trace['total_draws']} | {trace['unique_records_seen']} | {trace['duplicate_draw_count']} | {trace['max_draws_per_record']} | {trace['anchor_draws']}/{trace['expert_draws']} |")

    lines.extend(["", "## Discovery Gate 配对统计", "", "| Comparison | Both | Left only | Right only | ΔP@4 | Bootstrap 95% CI | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in comparisons.items():
        lines.append(f"| {name} | {row['both_solved']} | {row['left_only']} | {row['right_only']} | {row['delta_pass_at_4']:+.4f} | [{row['bootstrap_95_ci'][0]:+.4f}, {row['bootstrap_95_ci'][1]:+.4f}] | {row['mcnemar_exact_two_sided_p']:.4f} |")

    b2_pair = comparisons["M0_vs_B2"]
    lines.extend([
        "", "## 根因问题回答", "",
        "1. **无放回修正是否生效？** 是。五个训练臂均为 draws=physical rows、unique=draws、duplicate=0、max repeat=1。",
        "2. **旧实验退化能否完全归因于有放回采样？** 不能。修正后 B1 不再退化且 B2 提升，说明重复采样是重要混杂因素；但缺少同一代码其余条件完全一致的重复/无重复随机对照，不能声称唯一因果。",
        "3. **Anchor continuation 必然破坏 M0 吗？** 否。本轮 B1 在 Discovery 为 83/150、Monitor 12/64，没有重现先前大幅退化。",
        "4. **Frontier expert 是否带来额外收益？** B2 比 B1 多 3 题，点估计为正；是否稳健取决于 B1_vs_B2 配对区间和 p 值，不能仅凭点估计断言。",
        "5. **极少 step 是否立即退化？** C1 step_5 为 77/150，接近 M0；C2/C3 为 75/150。没有发现统一的立即崩坏，但纯 expert 微更新没有收益。",
        "6. **checkpoint/LoRA merge-load 是否造成漂移？** 没有权重层面的证据。C0 merged hash 与 M0 完全一致、greedy 10/10 一致；采样 Gate 的一题差异属于随机执行波动。",
        "7. **当前最小不退化方案是什么？** B1 step_10 是最保守的训练方案；若看收益，B2 step_63 最优，但尚未过发布门槛。",
        "8. **B 轮与 C 轮说明什么？** 完整单 epoch B 轮优于极小 C 轮；退化并非简单由 step 数单调决定，数据组合和足够优化量共同重要。",
        "9. **是否已证明 B2 显著优于 M0？** 配对结果为 " + f"Δ={b2_pair['delta_pass_at_4']:+.4f}, 95% CI={b2_pair['bootstrap_95_ci']}, McNemar p={b2_pair['mcnemar_exact_two_sided_p']:.4f}" + "；按区间/p 值解释，不夸大点估计。",
        "10. **是否继续第二轮 Expert Iteration？** 暂不继续。B2 未达到 98/150 Full Replay gate，应先复现实验或扩大固定 Gate 以确认收益。",
        "", "## 生成行为与运行时", "",
        "各模型的长度、finish reason、重复、多 proof、多样性、Top-10/Top-50 coverage 与 tactic presence 已完整保存在 `metrics.json` 的 `generation_behavior`。",
        f"本轮 release evaluations 总计 generation {runtime['evaluation_generation_seconds']:.1f}s、verification {runtime['evaluation_verification_seconds']:.1f}s；Pantograph restart={runtime['pantograph_worker_restarts']}，timeout={runtime['verification_timeouts']}，cache hit/miss={runtime['cache_hits']}/{runtime['cache_misses']}。",
        "", "## Gate 决策", "",
        "- Full Replay：未运行；所有训练模型均低于 98/150。",
        "- Benchmark：未运行；前置 Gate 未全部满足。",
        "- 新 M1/M2：未创建。所有训练 checkpoint 仍是消融候选。",
        "- 推荐：保留 B2 step_63 作为候选，先在另一个预注册 seed 或更大固定集合上复核；在通过发布门槛前不要进入第二轮 Expert Iteration。",
    ])
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(root / 'metrics.json'), "report": str(root / 'final_report.md')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
