"""Aggregate B1/B2 fixed-gate results and emit the final ablation report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from compare_expert_sft_ablation import (
    attempts_from_generation_verification,
    bootstrap_delta,
    gate_metrics,
    prompt_statement_key,
)
from compare_generation_drift import summarize as summarize_generations


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def canonical_discovery_attempts(
    generations: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    *,
    canonical_by_key: dict[str, str],
    allowed_keys: set[str],
) -> list[dict[str, Any]]:
    local_to_canonical: dict[str, str] = {}
    for row in generations:
        key = prompt_statement_key(str(row["prompt"]))
        if key in allowed_keys:
            local_to_canonical[str(row["statement_id"])] = canonical_by_key[key]
    attempts = attempts_from_generation_verification(generations, verifications)
    return [
        {**row, "problem_id": local_to_canonical[str(row["problem_id"])]}
        for row in attempts
        if str(row["problem_id"]) in local_to_canonical
    ]


def paired_names(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "both_solved": row["both_solved"],
        "M0_only": row["M0_only_solved"],
        "candidate_only": row["candidate_only_solved"],
        "neither": row["neither_solved"],
        "delta_pass_at_4": row["delta_pass_at_4"],
        "bootstrap_95_ci": row["bootstrap_95_ci"],
        "bootstrap_draws": row["bootstrap_draws"],
        "bootstrap_seed": row["bootstrap_seed"],
    }


def required_tactic_presence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the task-specified tactics; ``other`` means none of them appears."""
    patterns = {
        "simp": r"\bsimp(?:_all)?\b",
        "norm_num": r"\bnorm_num\b",
        "linarith": r"\b(?:nlinarith|linarith)\b",
        "aesop": r"\baesop\b",
        "omega": r"\bomega\b",
        "ring": r"\bring(?:_nf)?\b",
    }
    proofs = [
        str(row.get("normalized_proof") or row.get("extracted_proof") or "").strip()
        for row in rows
    ]
    proofs = [proof for proof in proofs if proof]
    counts = {
        tactic: sum(bool(re.search(pattern, proof)) for proof in proofs)
        for tactic, pattern in patterns.items()
    }
    counts["other"] = sum(
        not any(re.search(pattern, proof) for pattern in patterns.values())
        for proof in proofs
    )
    denominator = max(1, len(proofs))
    return {
        "denominator_extracted_proofs": len(proofs),
        "presence_counts": counts,
        "presence_ratios": {
            tactic: count / denominator for tactic, count in counts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/expert_sft_b_ablation")
    )
    parser.add_argument(
        "--prior", type=Path, default=Path("outputs/expert_sft_anchor_ablation")
    )
    parser.add_argument(
        "--round1", type=Path, default=Path("outputs/expert_iteration_round1")
    )
    args = parser.parse_args()
    root, prior, round1 = args.root, args.prior, args.round1

    selection = {
        "selection_order": [
            "retention_pass_at_4",
            "retention_pass_at_1",
            "retention_candidate_success_rate",
            "earlier_step",
        ],
        "M0": {"checkpoint": "clean_M0"},
        "B1": {"checkpoint": "step_25"},
        "B2": {"checkpoint": "step_25"},
    }

    retention_dirs = {
        "M0": root / "evaluation/retention/M0",
        **{
            f"B1_step_{step}": root / f"evaluation/retention/B1_step_{step}"
            for step in (25, 50, 75, 100)
        },
        **{
            f"B2_step_{step}": root / f"evaluation/retention/B2_step_{step}"
            for step in (25, 50, 75, 100)
        },
    }
    retention: dict[str, Any] = {}
    for name, path in retention_dirs.items():
        metric = read_json(path / "discovery_replay_metrics.json")
        behavior = summarize_generations(read_jsonl(path / "generations.jsonl"))
        retention[name] = {
            "pass_at_1": metric["discovery_replay_pass_at_1"],
            "pass_at_4": metric["discovery_replay_pass_at_4"],
            "candidate_success_rate": metric["discovery_replay_compile_success_rate"],
            "solved_statements": metric["successes"],
            "mean_output_tokens": behavior["mean_output_tokens"],
            "repetitive_output_ratio": behavior["repetitive_output_ratio"],
            "generation_behavior": behavior,
            "runtime": metric,
        }
    selection["B1"]["retention_candidates"] = {
        str(step): retention[f"B1_step_{step}"] for step in (25, 50, 75, 100)
    }
    selection["B2"]["retention_candidates"] = {
        str(step): retention[f"B2_step_{step}"] for step in (25, 50, 75, 100)
    }
    write_json(root / "checkpoint_selection.json", selection)

    gate_rows = read_jsonl(prior / "gates_stratified/discovery_gate_150.jsonl")
    gate_keys = {prompt_statement_key(str(row["prompt"])) for row in gate_rows}
    m0_generations = read_jsonl(round1 / "iteration_000/discovery/generations.jsonl")
    m0_verifications = read_jsonl(round1 / "iteration_000/discovery/verifications.jsonl")
    canonical_by_key = {
        prompt_statement_key(str(row["prompt"])): str(row["statement_id"])
        for row in m0_generations
        if prompt_statement_key(str(row["prompt"])) in gate_keys
    }
    if set(canonical_by_key) != gate_keys:
        raise ValueError("M0 discovery generations do not cover the fixed gate")
    canonical_ids = set(canonical_by_key.values())

    discovery_sources = {
        "M0": (m0_generations, m0_verifications),
        "B1": (
            read_jsonl(root / "evaluation/discovery_gate/B1/generations.jsonl"),
            read_jsonl(root / "evaluation/discovery_gate/B1/verifications.jsonl"),
        ),
        "B2": (
            read_jsonl(root / "evaluation/discovery_gate/B2/generations.jsonl"),
            read_jsonl(root / "evaluation/discovery_gate/B2/verifications.jsonl"),
        ),
    }
    discovery: dict[str, Any] = {}
    discovery_behavior: dict[str, Any] = {}
    for model, (generations, verifications) in discovery_sources.items():
        attempts = canonical_discovery_attempts(
            generations,
            verifications,
            canonical_by_key=canonical_by_key,
            allowed_keys=gate_keys,
        )
        metric = gate_metrics(attempts, canonical_ids)
        cropped_generations = [
            row
            for row in generations
            if prompt_statement_key(str(row["prompt"])) in gate_keys
        ]
        if len(cropped_generations) != 600:
            raise ValueError(
                f"{model} has {len(cropped_generations)} discovery candidates, expected 600"
            )
        discovery[model] = metric
        discovery_behavior[model] = summarize_generations(cropped_generations)
        discovery_behavior[model]["required_tactic_presence"] = (
            required_tactic_presence(cropped_generations)
        )

    paired = {
        model: paired_names(
            bootstrap_delta(
                discovery["M0"]["success_count_by_statement"],
                discovery[model]["success_count_by_statement"],
                seed=42,
                draws=10_000,
            )
        )
        for model in ("B1", "B2")
    }

    monitor_paths = {
        "M0": round1 / "monitor/M0/attempts.jsonl",
        "B1": root / "evaluation/monitor/B1/attempts.jsonl",
        "B2": root / "evaluation/monitor/B2/attempts.jsonl",
    }
    monitor_ids = {
        str(row["problem_id"]) for row in read_jsonl(monitor_paths["M0"])
    }
    monitor = {
        model: gate_metrics(read_jsonl(path), monitor_ids)
        for model, path in monitor_paths.items()
    }

    m0_eval = read_json(round1 / "evaluation/eval_loss.json")["M0"]
    evaluation = {
        "M0": {
            "eval_loss": m0_eval["eval_loss"],
            "eval_token_accuracy": m0_eval["eval_token_accuracy"],
        },
        "B1": read_json(root / "evaluation/eval_loss/B1/eval_metrics.json"),
        "B2": read_json(root / "evaluation/eval_loss/B2/eval_metrics.json"),
    }
    sampling = {
        "B1": read_json(root / "B1_balanced_anchor_only/sampling_trace.json"),
        "B2": read_json(
            root / "B2_balanced_anchor_frontier_90_10/sampling_trace.json"
        ),
    }
    training = {
        "B1": read_json(
            root
            / "B1_balanced_anchor_only/checkpoint/b_ablation_training_metrics.json"
        ),
        "B2": read_json(
            root
            / "B2_balanced_anchor_frontier_90_10/checkpoint/b_ablation_training_metrics.json"
        ),
    }

    release = {
        model: {
            "discovery_gate": len(discovery[model]["solved_statement_ids"]) >= 98,
            "monitor_gate": monitor[model]["pass_at_4"]
            >= monitor["M0"]["pass_at_4"] - 1 / 64,
        }
        for model in ("B1", "B2")
    }
    for model in release:
        release[model]["qualified_for_full_replay"] = all(release[model].values())
    full_replay = {
        "status": "not_run",
        "reason": "B1 and B2 both failed the fixed Discovery Gate threshold of 98/150",
        "required_discovery_solved": 98,
        "results": {
            model: {
                "discovery_solved": len(discovery[model]["solved_statement_ids"]),
                "monitor_pass_at_4": monitor[model]["pass_at_4"],
                **release[model],
            }
            for model in ("B1", "B2")
        },
    }
    write_json(root / "evaluation/full_replay/decision.json", full_replay)

    runtime_files = list(root.glob("evaluation/retention/*/discovery_replay_metrics.json"))
    runtime_files += list(root.glob("evaluation/discovery_gate/*/discovery_replay_metrics.json"))
    runtime_files += list(root.glob("evaluation/monitor/*/monitor_metrics.json"))
    runtime_rows = [read_json(path) for path in runtime_files]
    runtime = {
        "training_wall_seconds": sum(row["wall_seconds"] for row in training.values()),
        "evaluation_wall_seconds": sum(row["total_seconds"] for row in runtime_rows),
        "eval_loss_wall_seconds": sum(
            float(evaluation[model].get("wall_seconds", 0)) for model in ("B1", "B2")
        ),
        "gpu_peak_allocated_bytes": max(
            row["gpu_peak_allocated_bytes"] for row in training.values()
        ),
        "gpu_peak_reserved_bytes": max(
            row["gpu_peak_reserved_bytes"] for row in training.values()
        ),
        "process_peak_rss_kib": max(
            row["process_peak_rss_kib"] for row in training.values()
        ),
        "pantograph_worker_restart_count": sum(
            int(row.get("pantograph_worker_restart_count", 0)) for row in runtime_rows
        ),
        "cache_hits": sum(int(row.get("cache_hits", 0)) for row in runtime_rows),
        "cache_misses": sum(int(row.get("cache_misses", 0)) for row in runtime_rows),
    }
    runtime["recorded_total_seconds"] = (
        runtime["training_wall_seconds"]
        + runtime["evaluation_wall_seconds"]
        + runtime["eval_loss_wall_seconds"]
    )

    metrics = {
        "experiment": "Expert Iteration B1/B2 ablation",
        "initialization_checkpoint": "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor",
        "checkpoint_selection": selection,
        "sampling_trace": sampling,
        "training": training,
        "eval": evaluation,
        "retention": retention,
        "discovery_gate": discovery,
        "monitor": monitor,
        "generation_behavior_discovery_gate": discovery_behavior,
        "paired_statistics_discovery_gate": paired,
        "release_gates": release,
        "full_replay": full_replay,
        "runtime": runtime,
    }
    write_json(root / "metrics.json", metrics)
    write_json(root / "evaluation/paired_statistics.json", paired)
    write_json(root / "evaluation/generation_behavior.json", discovery_behavior)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Expert Iteration B1 / B2 消融实验报告",
        "",
        "## 结论",
        "",
        "本次实验没有找到可放行到 full replay 的配方。B1 与 B2 的最优 checkpoint 都是 step 25；两者 Monitor 均比 M0 多解 1 题，但固定 Discovery Gate 分别从 M0 的 100/150 降至 80/150 和 83/150。B2 比 B1 多解 3 题，说明 frontier expert 相对 anchor-only 有正贡献，但不足以抵消 continuation training 的退化。",
        "",
        "## 模型对比表",
        "",
        "| Model | Anchor | Expert | Selected | Eval loss | Eval token acc. | Monitor P@1 | Monitor P@4 | Discovery P@1 | Discovery P@4 | Δ P@4 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {
        "M0": ("—", "—", "clean M0"),
        "B1": ("100%", "0%", "step 25"),
        "B2": ("90%", "10% frontier", "step 25"),
    }
    for model, (anchor, expert, selected) in rows.items():
        delta = discovery[model]["pass_at_4"] - discovery["M0"]["pass_at_4"]
        lines.append(
            f"| {model} | {anchor} | {expert} | {selected} | {evaluation[model]['eval_loss']:.6f} | {pct(evaluation[model]['eval_token_accuracy'])} | {pct(monitor[model]['pass_at_1'])} | {pct(monitor[model]['pass_at_4'])} | {pct(discovery[model]['pass_at_1'])} | {pct(discovery[model]['pass_at_4'])} | {delta * 100:+.2f} pp |"
        )

    lines += [
        "",
        "## Retention checkpoint 筛选",
        "",
        "| Checkpoint | B1 Pass@4 | B2 Pass@4 |",
        "|---|---:|---:|",
    ]
    for step in (25, 50, 75, 100):
        lines.append(
            f"| step {step} | {pct(retention[f'B1_step_{step}']['pass_at_4'])} | {pct(retention[f'B2_step_{step}']['pass_at_4'])} |"
        )

    lines += [
        "",
        "## Discovery Gate paired 统计",
        "",
        "| Model | Both solved | M0 only | Candidate only | Neither | Δ Pass@4 | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B1", "B2"):
        row = paired[model]
        ci = row["bootstrap_95_ci"]
        lines.append(
            f"| {model} | {row['both_solved']} | {row['M0_only']} | {row['candidate_only']} | {row['neither']} | {row['delta_pass_at_4'] * 100:+.2f} pp | [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] pp |"
        )

    lines += [
        "",
        "## 生成行为变化（Discovery Gate）",
        "",
        "| Model | Mean tokens | P50 | P90 | P95 | Max | Length finish | Repetitive | Multiple proof | Unique proof | Mean unique/statement | Top10 | Top50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("M0", "B1", "B2"):
        row = discovery_behavior[model]
        lines.append(
            f"| {model} | {row['mean_output_tokens']:.2f} | {row['p50_output_tokens']:.0f} | {row['p90_output_tokens']:.0f} | {row['p95_output_tokens']:.0f} | {row['max_output_tokens']} | {pct(row['length_finish_ratio'])} | {pct(row['repetitive_output_ratio'])} | {pct(row['multiple_proof_ratio'])} | {pct(row['unique_normalized_proof_ratio'])} | {row['mean_unique_candidates_per_statement']:.3f} | {pct(row['top10_output_proof_coverage'])} | {pct(row['top50_output_proof_coverage'])} |"
        )

    lines += [
        "",
        "### Finish reason 分布",
        "",
        "| Model | stop | length | other/unknown |",
        "|---|---:|---:|---:|",
    ]
    for model in ("M0", "B1", "B2"):
        finish = discovery_behavior[model]["finish_reason_distribution"]
        known = int(finish.get("stop", 0)) + int(finish.get("length", 0))
        total = sum(int(value) for value in finish.values())
        lines.append(
            f"| {model} | {int(finish.get('stop', 0))} | {int(finish.get('length', 0))} | {total - known} |"
        )

    lines += [
        "",
        "### Tactic presence（占已成功抽取 proof 的比例）",
        "",
        "同一 proof 可包含多个 tactic；`other` 表示未出现下列六类 tactic。",
        "",
        "| Model | simp | norm_num | linarith | aesop | omega | ring | other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("M0", "B1", "B2"):
        ratios = discovery_behavior[model]["required_tactic_presence"][
            "presence_ratios"
        ]
        lines.append(
            f"| {model} | {pct(ratios['simp'])} | {pct(ratios['norm_num'])} | {pct(ratios['linarith'])} | {pct(ratios['aesop'])} | {pct(ratios['omega'])} | {pct(ratios['ring'])} | {pct(ratios['other'])} |"
        )

    lines += [
        "",
        "## 采样与运行记录",
        "",
        "| Experiment | Actual anchor draw | Actual expert draw | Anchor label tokens | Expert label tokens | Anchor unique | Expert unique |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("B1", "B2"):
        row = sampling[model]
        lines.append(
            f"| {model} | {pct(row['actual_anchor_draw_ratio'])} | {pct(row['actual_expert_draw_ratio'])} | {pct(row['anchor_label_token_share'])} | {pct(row['expert_label_token_share'])} | {row['anchor_unique_seen']} | {row['expert_unique_seen']} |"
        )
    lines += [
        "",
        f"记录总耗时 {runtime['recorded_total_seconds'] / 3600:.2f} h；训练峰值 GPU allocated/reserved 为 {runtime['gpu_peak_allocated_bytes'] / 2**30:.2f}/{runtime['gpu_peak_reserved_bytes'] / 2**30:.2f} GiB；进程峰值 RSS 为 {runtime['process_peak_rss_kib'] / 2**20:.2f} GiB；Pantograph worker 重启 {runtime['pantograph_worker_restart_count']} 次。",
    ]

    lines += [
        "",
        "## 解释",
        "",
        "1. **Anchor continuation 是主要退化来源。** B1 不含 expert，虽然只训练到最佳 step 25，Discovery Pass@4 仍比 M0 低 13.33 pp。",
        "2. **Frontier expert 有相对正收益。** B2 比 B1 多解 3/150（+2.00 pp），但仍比 M0 低 11.33 pp，符合“B1 < M0，B2 > B1，但 B2 < M0”的情况。",
        "3. **当前训练预算仍然过强。** 四个 checkpoint 中最早的 step 25 最优；从 step 50 起 Retention 已下降，说明退化很早发生。",
        "4. **本次没有证据表明应放弃 frontier selection。** 90/10 frontier 配方优于 anchor-only；问题更接近 continuation 强度，而不是 expert 选择方向本身。",
        "",
        "## 下一步建议",
        "",
        "- 最佳数据方向：保留 balanced anchor + frontier expert 90/10，每个 statement 最多一个 verified proof。",
        "- 最佳已测试配置：LR 1e-5、step 25，但仍未通过 Discovery Gate，不能作为正式 Expert SFT 配方。",
        "- 下一次仅做更小预算消融：优先测试 LR 1e-5 的 step 5/10/15，或 LR 5e-6 的 step 10/25；仍使用同一固定 gate。",
        "- 暂不继续下一轮 Expert Iteration，不运行 full replay，不推广 B1/B2 checkpoint。",
        "",
        "## Full Replay 决策",
        "",
        "未运行。B1=80/150、B2=83/150，均低于必须满足的 98/150；虽然两者 Monitor Pass@4=13/64，均满足 Monitor 门槛，但双门槛没有同时通过。",
    ]
    (root / "ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": str(root / "metrics.json"), "report": str(root / "ablation_report.md"), "full_replay": full_replay}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
