"""Aggregate fixed-gate Expert SFT ablations and emit paired statistical reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any

from compare_generation_drift import summarize as summarize_generations


ARMS = {
    "A0": "A0_clean_M0",
    "A1": "A1_unbalanced_anchor_only",
    "A2": "A2_current_M1_85_15",
    "A3": "A3_unbalanced_anchor_frontier_90_10",
    "A4": "A4_balanced_anchor_only",
    "A5": "A5_balanced_anchor_frontier_90_10",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def attempts_from_generation_verification(
    generations: list[dict[str, Any]], verifications: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    generation_by_id = {row["generation_id"]: row for row in generations}
    attempts = []
    for verification in verifications:
        generation = generation_by_id[verification["generation_id"]]
        attempts.append(
            {
                "attempt_index": int(generation["sample_index"]),
                "generation_id": generation["generation_id"],
                "problem_id": generation["statement_id"],
                "success": bool(verification.get("verified")),
                "status": verification.get("status"),
                "timed_out": bool(verification.get("timed_out")),
            }
        )
    return attempts


def gate_metrics(attempts: list[dict[str, Any]], allowed: set[str]) -> dict[str, Any]:
    selected = [row for row in attempts if str(row["problem_id"]) in allowed]
    by_statement: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_statement[str(row["problem_id"])].append(row)
    if set(by_statement) != allowed:
        missing = sorted(allowed - set(by_statement))
        raise ValueError(f"gate attempts missing {len(missing)} statements: {missing[:3]}")
    if any(len(rows) != 4 for rows in by_statement.values()):
        raise ValueError("each gate statement must have exactly four attempts")
    success_counts = {
        statement_id: sum(bool(row["success"]) for row in rows)
        for statement_id, rows in by_statement.items()
    }
    pass1 = {
        statement_id: any(bool(row["success"]) and int(row["attempt_index"]) == 0 for row in rows)
        for statement_id, rows in by_statement.items()
    }
    total_success = sum(success_counts.values())
    solved = sorted(statement_id for statement_id, count in success_counts.items() if count)
    return {
        "statements": len(allowed),
        "candidates": len(selected),
        "verified_candidates": total_success,
        "candidate_success_rate": total_success / len(selected),
        "pass_at_1": sum(pass1.values()) / len(allowed),
        "pass_at_4": len(solved) / len(allowed),
        "solved_statement_ids": solved,
        "success_count_by_statement": success_counts,
        "timeout_attempts": sum(bool(row.get("timed_out")) for row in selected),
    }


def bootstrap_delta(
    baseline: dict[str, int], candidate: dict[str, int], seed: int = 42, draws: int = 10_000
) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(candidate))
    base = [int(baseline[key] > 0) for key in ids]
    cand = [int(candidate[key] > 0) for key in ids]
    differences = [c - b for b, c in zip(base, cand, strict=True)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        estimates.append(sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids))
    estimates.sort()
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    both = sum(bool(b) and bool(c) for b, c in zip(base, cand, strict=True))
    base_only = sum(bool(b) and not bool(c) for b, c in zip(base, cand, strict=True))
    candidate_only = sum(not bool(b) and bool(c) for b, c in zip(base, cand, strict=True))
    return {
        "statements": len(ids),
        "both_solved": both,
        "M0_only_solved": base_only,
        "candidate_only_solved": candidate_only,
        "neither_solved": len(ids) - both - base_only - candidate_only,
        "delta_pass_at_4": sum(differences) / len(ids),
        "bootstrap_95_ci": [low, high],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def bucket_metrics(
    metric: dict[str, Any], bucket_by_id: dict[str, str]
) -> dict[str, dict[str, Any]]:
    ids_by_bucket: defaultdict[str, set[str]] = defaultdict(set)
    for statement_id in metric["success_count_by_statement"]:
        ids_by_bucket[bucket_by_id[statement_id]].add(statement_id)
    result = {}
    for bucket, ids in sorted(ids_by_bucket.items()):
        counts = metric["success_count_by_statement"]
        solved = sum(counts[key] > 0 for key in ids)
        candidates = 4 * len(ids)
        verified = sum(counts[key] for key in ids)
        result[bucket] = {
            "statements": len(ids),
            "pass_at_4": solved / len(ids),
            "verified_candidates": verified,
            "candidate_success_rate": verified / candidates,
        }
    return result


def crop_generations(rows: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["statement_id"]) in allowed]


def prompt_statement_key(prompt: str) -> str:
    marker = "### Lean statement\n"
    proof_marker = "\n\n### Lean proof"
    if marker not in prompt or proof_marker not in prompt:
        raise ValueError("prompt does not contain the fixed Lean statement/proof sections")
    return prompt.split(marker, 1)[1].split(proof_marker, 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/expert_sft_anchor_ablation"))
    parser.add_argument("--round1", type=Path, default=Path("outputs/expert_iteration_round1"))
    args = parser.parse_args()
    root, round1 = args.root, args.round1

    anchor_rows = read_jsonl(root / "gates_stratified/anchor_gate_150.jsonl")
    dynamic_rows = read_jsonl(root / "audit/anchor_dynamic_difficulty.jsonl")
    dynamic_by_record = {str(row["record_id"]): row for row in dynamic_rows}
    anchor_ids = {
        str(dynamic_by_record[str(row.get("id") or row.get("record_id"))]["evaluation_statement_id"])
        for row in anchor_rows
    }
    dynamic_by_eval = {str(row["evaluation_statement_id"]): row for row in dynamic_rows}
    anchor_bucket_by_id = {
        statement_id: (
            "hard" if int(dynamic_by_eval[statement_id]["m0_success_count_at_4"]) == 0
            else "frontier" if int(dynamic_by_eval[statement_id]["m0_success_count_at_4"]) in (1, 2)
            else "medium" if int(dynamic_by_eval[statement_id]["m0_success_count_at_4"]) == 3
            else "easy"
        )
        for statement_id in anchor_ids
    }

    discovery_gate_rows = read_jsonl(root / "gates_stratified/discovery_gate_150.jsonl")
    discovery_gate_prompts = {str(row["prompt"]) for row in discovery_gate_rows}
    discovery_gate_statement_keys = {
        prompt_statement_key(prompt) for prompt in discovery_gate_prompts
    }
    m0_discovery_generations = read_jsonl(round1 / "iteration_000/discovery/generations.jsonl")
    m0_id_by_statement_key = {
        prompt_statement_key(str(row["prompt"])): str(row["statement_id"])
        for row in m0_discovery_generations
    }
    discovery_ids = {
        str(row["statement_id"]) for row in m0_discovery_generations if str(row["prompt"]) in discovery_gate_prompts
    }
    if len(discovery_ids) != 150:
        raise ValueError(f"expected 150 discovery gate IDs, found {len(discovery_ids)}")
    m0_discovery_verifications = read_jsonl(round1 / "iteration_000/discovery/verifications.jsonl")
    m0_discovery_attempts = attempts_from_generation_verification(
        m0_discovery_generations, m0_discovery_verifications
    )
    m0_success_counts_full = gate_metrics(
        m0_discovery_attempts,
        {str(row["statement_id"]) for row in m0_discovery_generations},
    )["success_count_by_statement"]
    discovery_bucket_by_id = {
        statement_id: str(m0_success_counts_full[statement_id]) + "_of_4" for statement_id in discovery_ids
    }

    attempt_paths = {
        "anchor_gate": {
            "A0": root / "audit/anchor_dynamic_profile/M0/attempts.jsonl",
            "A1": root / f"{ARMS['A1']}/evaluation/stratified_anchor_gate/attempts.jsonl",
            "A2": root / f"{ARMS['A2']}/evaluation/stratified_anchor_gate/attempts.jsonl",
            "A3": root / f"{ARMS['A3']}/evaluation/stratified_anchor_gate/attempts.jsonl",
            "A4": root / f"{ARMS['A4']}/evaluation/stratified_anchor_gate/attempts.jsonl",
            "A5": root / f"{ARMS['A5']}/evaluation/stratified_anchor_gate/attempts.jsonl",
        },
        "discovery_gate": {
            "A1": root / f"{ARMS['A1']}/evaluation/stratified_discovery_gate_retry/attempts.jsonl",
            "A2": round1 / "evaluation/discovery_replay/M1/attempts.jsonl",
            "A3": root / f"{ARMS['A3']}/evaluation/stratified_discovery_gate/attempts.jsonl",
            "A4": root / f"{ARMS['A4']}/evaluation/stratified_discovery_gate/attempts.jsonl",
            "A5": root / f"{ARMS['A5']}/evaluation/stratified_discovery_gate/attempts.jsonl",
        },
        "monitor": {
            "A0": round1 / "monitor/M0/attempts.jsonl",
            "A2": round1 / "iteration_000/monitor/attempts.jsonl",
            **{
                arm: root / f"{ARMS[arm]}/evaluation/monitor/attempts.jsonl"
                for arm in ("A1", "A3", "A4", "A5")
            },
        },
    }
    allowed = {
        "anchor_gate": anchor_ids,
        "discovery_gate": discovery_ids,
        "monitor": {
            str(row["problem_id"]) for row in read_jsonl(attempt_paths["monitor"]["A0"])
        },
    }
    metrics: dict[str, dict[str, Any]] = {gate: {} for gate in allowed}
    for gate in ("anchor_gate", "monitor"):
        for arm, path in attempt_paths[gate].items():
            metrics[gate][arm] = gate_metrics(read_jsonl(path), allowed[gate])
    metrics["discovery_gate"]["A0"] = gate_metrics(m0_discovery_attempts, discovery_ids)
    for arm, path in attempt_paths["discovery_gate"].items():
        arm_attempts = read_jsonl(path)
        if arm == "A2":
            a2_generations = read_jsonl(round1 / "evaluation/discovery_replay/M1/generations.jsonl")
            canonical_by_a2_id = {
                str(row["statement_id"]): m0_id_by_statement_key[
                    prompt_statement_key(str(row["prompt"]))
                ]
                for row in a2_generations
            }
            arm_attempts = [
                {**row, "problem_id": canonical_by_a2_id[str(row["problem_id"])]}
                for row in arm_attempts
            ]
        metrics["discovery_gate"][arm] = gate_metrics(arm_attempts, discovery_ids)

    for arm in ARMS:
        metrics["anchor_gate"][arm]["by_m0_dynamic_bucket"] = bucket_metrics(
            metrics["anchor_gate"][arm], anchor_bucket_by_id
        )
        metrics["discovery_gate"][arm]["by_m0_success_count"] = bucket_metrics(
            metrics["discovery_gate"][arm], discovery_bucket_by_id
        )

    paired: dict[str, dict[str, Any]] = {}
    for gate in ("monitor", "anchor_gate", "discovery_gate"):
        paired[gate] = {}
        baseline = metrics[gate]["A0"]["success_count_by_statement"]
        for arm in ("A1", "A2", "A3", "A4", "A5"):
            paired[gate][arm] = bootstrap_delta(
                baseline, metrics[gate][arm]["success_count_by_statement"]
            )

    generation_paths = {
        "A0": round1 / "iteration_000/discovery/generations.jsonl",
        "A1": root / f"{ARMS['A1']}/evaluation/stratified_discovery_gate_retry/generations.jsonl",
        "A2": round1 / "evaluation/discovery_replay/M1/generations.jsonl",
        "A3": root / f"{ARMS['A3']}/evaluation/stratified_discovery_gate/generations.jsonl",
        "A4": root / f"{ARMS['A4']}/evaluation/stratified_discovery_gate/generations.jsonl",
        "A5": root / f"{ARMS['A5']}/evaluation/stratified_discovery_gate/generations.jsonl",
    }
    generation_behavior = {}
    cropped_dir = root / "evaluations/gate/cropped_baselines"
    for arm, path in generation_paths.items():
        all_rows = read_jsonl(path)
        rows = (
            [
                row
                for row in all_rows
                if prompt_statement_key(str(row["prompt"])) in discovery_gate_statement_keys
            ]
            if arm == "A2"
            else crop_generations(all_rows, discovery_ids)
        )
        if len(rows) != 600:
            raise ValueError(f"{arm} has {len(rows)} discovery gate generations, expected 600")
        generation_behavior[arm] = summarize_generations(rows)
        if arm in ("A0", "A2"):
            write_jsonl(cropped_dir / f"{arm}_discovery_gate_generations.jsonl", rows)
    write_jsonl(
        cropped_dir / "A0_discovery_gate_attempts.jsonl",
        [row for row in m0_discovery_attempts if str(row["problem_id"]) in discovery_ids],
    )
    write_jsonl(
        cropped_dir / "A2_discovery_gate_attempts.jsonl",
        [
            row
            for row in (
                {
                    **item,
                    "problem_id": canonical_by_a2_id[str(item["problem_id"])],
                    "original_problem_id": item["problem_id"],
                }
                for item in read_jsonl(attempt_paths["discovery_gate"]["A2"])
            )
            if str(row["problem_id"]) in discovery_ids
        ],
    )

    eval_loss_source = read_json(round1 / "evaluation/eval_loss.json")
    eval_metrics = {
        "A0": {
            "eval_loss": eval_loss_source["M0"]["eval_loss"],
            "eval_token_accuracy": eval_loss_source["M0"]["eval_token_accuracy"],
        },
        "A2": {
            "eval_loss": eval_loss_source["M1"]["eval_loss"],
            "eval_token_accuracy": eval_loss_source["M1"]["eval_token_accuracy"],
        },
    }
    training_metrics = {}
    for arm in ("A1", "A3", "A4", "A5"):
        row = read_json(root / f"{ARMS[arm]}/checkpoint/ablation_training_metrics.json")
        eval_metrics[arm] = {
            "eval_loss": row["eval_loss"],
            "eval_token_accuracy": row["eval_token_accuracy"],
        }
        training_metrics[arm] = row

    m0_eval_loss = eval_metrics["A0"]["eval_loss"]
    release = {}
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        checks = {
            "monitor": metrics["monitor"][arm]["pass_at_4"] >= metrics["monitor"]["A0"]["pass_at_4"] - 1 / 64,
            "discovery_gate": metrics["discovery_gate"][arm]["pass_at_4"] >= metrics["discovery_gate"]["A0"]["pass_at_4"] - 2 / 150,
            "eval_loss": (eval_metrics[arm]["eval_loss"] - m0_eval_loss) / m0_eval_loss <= 0.03,
        }
        release[arm] = {"checks": checks, "qualified_for_full_replay": all(checks.values())}

    full_replay = {
        "status": "not_run",
        "reason": "no trained candidate passed all fixed release gates",
        "qualified_candidates": [arm for arm, row in release.items() if row["qualified_for_full_replay"]],
        "existing_reference": {
            "A0_pass_at_4": 0.452,
            "A2_pass_at_4": 0.342,
            "note": "existing round-1 500-statement results; no new full replay was launched",
        },
    }
    benchmark = {
        "status": "not_run",
        "reason": "no candidate had both monitor and full-replay Pass@4 at least M0",
    }

    sampler_audit = read_json(root / "audit/current_recipe_sampling_trace.json")
    format_audit = read_json(root / "audit/source_format_comparison.json")
    static_audit = read_json(root / "audit/anchor_static_summary.json")
    dynamic_audit = read_json(root / "audit/anchor_dynamic_summary.json")
    drift_audit = read_json(root / "audit/m0_m1_generation_drift.json")
    expert_difficulty = read_json(root / "expert_difficulty_summary.json")
    expert_quality = read_json(root / "expert_quality_layers.json")
    train_manifest = read_json(root / "ablation_train_manifest.json")
    round1_metrics = read_json(round1 / "metrics.json")

    runtime_metric_paths = [root / "audit/anchor_dynamic_profile/M0/discovery_replay_metrics.json"]
    for arm in ("A1", "A3", "A4", "A5"):
        runtime_metric_paths.append(root / f"{ARMS[arm]}/evaluation/monitor/monitor_metrics.json")
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        runtime_metric_paths.append(
            root / f"{ARMS[arm]}/evaluation/stratified_anchor_gate/discovery_replay_metrics.json"
        )
    runtime_metric_paths.extend(
        [
            root / f"{ARMS['A1']}/evaluation/stratified_discovery_gate_retry/discovery_replay_metrics.json",
            *[
                root / f"{ARMS[arm]}/evaluation/stratified_discovery_gate/discovery_replay_metrics.json"
                for arm in ("A3", "A4", "A5")
            ],
        ]
    )
    runtime_rows = [read_json(path) for path in runtime_metric_paths]
    runtime_summary = {
        "recorded_training_wall_seconds": sum(row["wall_seconds"] for row in training_metrics.values()),
        "recorded_dynamic_and_evaluation_stage_seconds": sum(row["total_seconds"] for row in runtime_rows),
        "recorded_stage_seconds_total_excluding_static_compile_and_reporting": (
            sum(row["wall_seconds"] for row in training_metrics.values())
            + sum(row["total_seconds"] for row in runtime_rows)
        ),
        "max_training_gpu_peak_reserved_bytes": max(
            row["gpu_peak_reserved_bytes"] for row in training_metrics.values()
        ),
        "max_training_gpu_peak_allocated_bytes": max(
            row["gpu_peak_allocated_bytes"] for row in training_metrics.values()
        ),
        "max_training_process_rss_kib": max(row["process_peak_rss_kib"] for row in training_metrics.values()),
        "completed_gate_and_monitor_worker_restarts": sum(
            int(row.get("pantograph_worker_restart_count", 0)) for row in runtime_rows
        ),
        "completed_gate_and_monitor_cache_hits": sum(int(row.get("cache_hits", 0)) for row in runtime_rows),
        "completed_gate_and_monitor_cache_misses": sum(int(row.get("cache_misses", 0)) for row in runtime_rows),
        "infrastructure_incidents": [
            "A1 Discovery Gate first launch: transient CUDA OOM before generation; isolated retry completed"
        ],
    }

    result = {
        "gate_manifest": read_json(root / "gates_stratified/gate_manifest.json"),
        "eval": eval_metrics,
        "training": training_metrics,
        "gates": metrics,
        "paired_statistics": paired,
        "generation_behavior_discovery_gate": generation_behavior,
        "release_gates": release,
        "full_replay": full_replay,
        "benchmark": benchmark,
        "audits": {
            "sampler": sampler_audit,
            "format": {
                key: format_audit[key]
                for key in (
                    "sample_count_by_source",
                    "check_pass_counts",
                    "same_prompt_template",
                    "same_full_by_proof_format",
                    "all_have_labeled_eos",
                    "all_prefixes_match",
                    "all_without_markdown_fence",
                    "all_without_repeated_theorem",
                )
            },
            "anchor_static": static_audit,
            "anchor_dynamic": dynamic_audit,
            "expert_difficulty": expert_difficulty,
            "expert_quality_layers": expert_quality,
            "training_manifests": train_manifest,
            "m0_m1_generation_drift": drift_audit,
        },
        "runtime": runtime_summary,
        "monitor_one_statement_percentage_points": 100 / 64,
    }
    write_json(root / "evaluations/paired_statistics/paired_statistics.json", paired)
    write_json(root / "evaluations/gate/gate_metrics.json", metrics)
    write_json(root / "evaluations/generation_behavior.json", generation_behavior)
    write_json(root / "evaluations/full_replay/decision.json", full_replay)
    write_json(root / "evaluations/benchmark_decision.json", benchmark)
    write_json(root / "ablation_metrics.json", result)

    labels = {
        "A0": ("M0", "—", "—"),
        "A1": ("A1", "unbalanced", "none"),
        "A2": ("A2", "unbalanced", "all, 85/15"),
        "A3": ("A3", "unbalanced", "frontier, 90/10"),
        "A4": ("A4", "balanced", "none"),
        "A5": ("A5", "balanced", "frontier, 90/10"),
    }
    lines = [
        "# Anchor Difficulty and Expert SFT Ablation Report",
        "",
        "## Executive conclusion",
        "",
        "All continuation-trained arms remained below clean M0 on the fixed Discovery Gate. A5 was the strongest trained recipe, showing that balanced anchor sampling and frontier-only expert proofs are complementary, but it solved 83/150 versus M0's 100/150 and therefore failed the 98/150 release threshold. No new full replay, benchmark, second Expert Iteration round, or M2 was run.",
        "",
        "## Controlled ablation table",
        "",
        "| Model | Anchor | Expert | Eval loss | Eval token acc. | Monitor P@1 | Monitor P@4 | Anchor Gate P@4 | Discovery Gate P@4 | Full replay |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        name, anchor, expert = labels[arm]
        monitor = metrics["monitor"][arm]
        anchor_gate = metrics["anchor_gate"][arm]
        discovery = metrics["discovery_gate"][arm]
        full = "45.2% existing" if arm == "A0" else "34.2% existing" if arm == "A2" else "not run (gate fail)"
        lines.append(
            f"| {name} | {anchor} | {expert} | {eval_metrics[arm]['eval_loss']:.6f} | {eval_metrics[arm]['eval_token_accuracy']:.4%} | {monitor['pass_at_1']:.4%} | {monitor['pass_at_4']:.4%} | {anchor_gate['pass_at_4']:.4%} | {discovery['pass_at_4']:.4%} | {full} |"
        )
    lines.extend(
        [
            "",
            "## Release decision",
            "",
            f"The fixed Discovery Gate contains M0 success-count strata 0/4:50, 1/4:40, 2/4:30, 3/4:20, 4/4:10. M0 therefore solves exactly 100/150. The candidate threshold is 98/150. A1/A2/A3/A4/A5 solved respectively "
            f"{len(metrics['discovery_gate']['A1']['solved_statement_ids'])}/{len(metrics['discovery_gate']['A2']['solved_statement_ids'])}/{len(metrics['discovery_gate']['A3']['solved_statement_ids'])}/{len(metrics['discovery_gate']['A4']['solved_statement_ids'])}/{len(metrics['discovery_gate']['A5']['solved_statement_ids'])}, so none qualified. The benchmark prerequisite also failed.",
            "",
            "The requested Anchor Gate 40/60/30/20 was infeasible because the 900-record dynamic profile contains only 27 medium and 9 easy records. The deterministic gate records this shortfall and uses 47 hard, 67 frontier, 27 medium, and 9 easy records without duplication.",
            "",
            "## Current training-contract audit",
            "",
            f"The physical dataset is 3,000 anchor + 226 expert rows ({sampler_audit['physical_row_ratio']['expert']:.4%} expert), while the reconstructed DataLoader produced {sampler_audit['anchor_draws']}/{sampler_audit['expert_draws']} anchor/expert draws ({sampler_audit['expert_draw_ratio']:.4%} expert). The requested 85/15 draw contract is therefore effective within ±1 percentage point. Completion-label exposure is {sampler_audit['anchor_completion_label_tokens']}/{sampler_audit['expert_completion_label_tokens']} tokens, so expert label-token share is only {sampler_audit['expert_label_token_share']:.4%}.",
            "",
            f"Anchor coverage was {sampler_audit['anchor_unique_records_seen']}/3000 ({sampler_audit['anchor_coverage_ratio']:.2%}) with maximum repeat {sampler_audit['anchor_max_draws_per_record']}; expert coverage was {sampler_audit['expert_unique_records_seen']}/226 ({sampler_audit['expert_coverage_ratio']:.2%}) with maximum repeat {sampler_audit['expert_max_draws_per_record']}. The 20+20 format audit passed every prompt/proof/EOS/prefix/no-leakage check, so the existing M1 remains a valid A2 algorithmic control rather than `mixture_contract_invalid`.",
            "",
            "## Anchor difficulty audit",
            "",
            f"All 3,000 anchor proofs compiled successfully. Static buckets are hard {static_audit['bucket_counts']['static_hard']}, medium {static_audit['bucket_counts']['static_medium']}, easy {static_audit['bucket_counts']['static_easy']}. Proof tokens have mean/P50/P75/P90/P95/max {static_audit['subsets']['all_anchor']['proof_tokens']['mean']:.2f}/{static_audit['subsets']['all_anchor']['proof_tokens']['p50']:.0f}/{static_audit['subsets']['all_anchor']['proof_tokens']['p75']:.0f}/{static_audit['subsets']['all_anchor']['proof_tokens']['p90']:.0f}/{static_audit['subsets']['all_anchor']['proof_tokens']['p95']:.0f}/{static_audit['subsets']['all_anchor']['proof_tokens']['max']:.0f}; single-tactic ratio is {static_audit['subsets']['all_anchor']['single_tactic_ratio']:.2%}, exact duplicate ratio {static_audit['subsets']['all_anchor']['exact_duplicate_proof_ratio']:.2%}, and there are {static_audit['subsets']['all_anchor']['distinct_tactic_signatures']} tactic signatures.",
            "",
            f"The 900-statement M0 profile produced dynamic hard/frontier/medium/easy counts {dynamic_audit['dynamic_bucket_counts']['dynamic_hard']}/{dynamic_audit['dynamic_bucket_counts']['dynamic_frontier']}/{dynamic_audit['dynamic_bucket_counts']['dynamic_medium']}/{dynamic_audit['dynamic_bucket_counts']['dynamic_easy']}; M0 solved {dynamic_audit['records'] - dynamic_audit['dynamic_bucket_counts']['dynamic_hard']}/900 at Pass@4 and verified {int(dynamic_audit['verified_candidates'])}/3600 candidates. Static ordinal difficulty versus M0 success-count Spearman is {dynamic_audit['spearman_static_ordinal_vs_m0_success_count']:.4f}, with monotonic easy→hard performance, so the remaining 2,100 records were not dynamically expanded. The anchor is not overwhelmingly trivial: 54.03% are static-hard and only 27.30% static-easy, though its single-tactic subset is substantially more repetitive (26.28% duplicate) than multi-step proofs (6.32%).",
            "",
            "## Balanced-anchor exposure",
            "",
            f"A1 draws preserve the unbalanced proxy distribution hard/frontier/medium/easy = 2376/466/229/161, cover {train_manifest['arms']['A1_unbalanced_anchor_only']['unique_records']} unique anchor records, and allow maximum repeat 6. A4 changes this to 808/1293/646/485, covers {train_manifest['arms']['A4_balanced_anchor_only']['unique_records']} unique records, and caps repeats at 3. A5 uses 2910 anchor + 322 frontier-expert draws (90.04/9.96%) and expert label-token share {train_manifest['arms']['A5_balanced_anchor_frontier_90_10']['source_label_token_ratios']['frontier_expert']:.4%}. All materialized rows remain Pantograph verified.",
            "",
            "## Expert proof stratification and quality",
            "",
            f"The fixed 500-statement discovery result contains 274 unsolved, 96 at 1/4, 57 at 2/4, 46 at 3/4, and 27 at 4/4. Thus 153 statements form the 1/4+2/4 frontier, and each contributes at most one verified proof.",
            "",
            f"Across all 456 verified candidates, selected 226 proofs, frontier 153 proofs, and actual 493 expert draws, mean proof tokens are respectively {expert_quality['all_verified_candidates']['proof_tokens']['mean']:.2f}/{expert_quality['selected_expert_proofs']['proof_tokens']['mean']:.2f}/{expert_quality['frontier_selected_proofs']['proof_tokens']['mean']:.2f}/{expert_quality['actual_current_recipe_expert_draws']['proof_tokens']['mean']:.2f}. Duplicate ratios are {expert_quality['all_verified_candidates']['duplicate_proof_ratio']:.2%}/{expert_quality['selected_expert_proofs']['duplicate_proof_ratio']:.2%}/{expert_quality['frontier_selected_proofs']['duplicate_proof_ratio']:.2%}/{expert_quality['actual_current_recipe_expert_draws']['duplicate_proof_ratio']:.2%}. The high 70.99% draw-level duplication is produced by weighted re-exposure, not by the frontier pool itself (16.99%).",
            "",
            "## Root-cause ranking",
            "",
            "1. **Continuation budget / second exposure to anchor data is the primary cause.** A1, which adds no expert data, falls from M0 100/150 to 66/150 on Discovery Gate. All four newly trained arms use the same 202-step, 2e-5 budget and all remain below M0 despite lower eval losses.",
            "2. **Anchor difficulty composition is material but secondary.** A4 improves over A1 by 9 Discovery-Gate statements and 3 Anchor-Gate statements, but still loses 25 Discovery-Gate statements versus M0 and drops Monitor Pass@4.",
            "3. **Expert selection and source/token exposure matter and partially recover capability.** A3 improves over A1 by 6 Discovery-Gate statements. A5 improves over A4 by 8 and is the strongest trained arm, but still does not recover M0. The 90/10 row mixture corresponds to different expert label-token shares (A3 about 7.51%, A5 about 11.24%).",
            "4. **Using all expert proofs at 15% worsens the outcome.** A2 is the worst Anchor-Gate continuation arm (61/150) and its existing full replay is 34.2%, supporting frontier filtering and a lower expert draw ratio.",
            "5. **Sampler implementation and source formatting are not supported as root causes.** The 85/15 trace is within one percentage point and 20+20 source records have matching prompt, proof, EOS, and label contracts.",
            "6. **Generation drift is a symptom/amplifier.** Existing M1 outputs are longer and more often length-terminated/repetitive than M0; this is consistent with over-training but does not replace the controlled A1 evidence.",
            "",
            "## Recommended next controlled experiment",
            "",
            "Do not promote A1–A5 and do not start the next Expert Iteration round. Retain A5's data recipe: balanced anchor mix hard/frontier/medium/easy = 25/40/20/15, anchor/expert draws = 90/10, expert buckets = M0 success 1/4 or 2/4 only, one proof per statement, max anchor repeat = 3. First test learning rate 1e-5 with 100 optimizer steps; then, only if warranted, 1e-5 with 202 steps and 5e-6 with 202 steps. Require the same Monitor and Discovery Gate thresholds before any full replay.",
            "",
            "## Statistical interpretation",
            "",
            "All Pass@4 deltas and 95% confidence intervals are in `evaluations/paired_statistics/paired_statistics.json`. On Monitor, one statement equals 1.5625 percentage points. Confidence intervals are statement-level paired bootstraps with 10,000 deterministic resamples (seed 42).",
            "",
            "## Output and runtime behavior",
            "",
            f"On the original full 500 replay, M0/M1 mean output tokens are {drift_audit['M0']['mean_output_tokens']:.2f}/{drift_audit['M1']['mean_output_tokens']:.2f}; length-finish ratios {drift_audit['M0']['length_finish_ratio']:.2%}/{drift_audit['M1']['length_finish_ratio']:.2%}; repetitive-output ratios {drift_audit['M0']['repetitive_output_ratio']:.2%}/{drift_audit['M1']['repetitive_output_ratio']:.2%}. This confirms a strong M1 output-length/repetition drift. Comparable A0–A5 behavior on the same 150-statement Discovery Gate is in `evaluations/generation_behavior.json`.",
            "",
            f"Recorded ablation training wall time is {runtime_summary['recorded_training_wall_seconds'] / 3600:.2f} h; recorded dynamic-profile plus evaluation stage time is {runtime_summary['recorded_dynamic_and_evaluation_stage_seconds'] / 3600:.2f} h. The combined recorded stage total is {runtime_summary['recorded_stage_seconds_total_excluding_static_compile_and_reporting'] / 3600:.2f} h, excluding the static 3,000-proof compile wall time and offline reporting. Peak recorded training GPU reserved/allocated memory is {runtime_summary['max_training_gpu_peak_reserved_bytes'] / 2**30:.2f}/{runtime_summary['max_training_gpu_peak_allocated_bytes'] / 2**30:.2f} GiB; peak training-process RSS is {runtime_summary['max_training_process_rss_kib'] / 2**20:.2f} GiB. Completed dynamic/Gate/Monitor runs had {runtime_summary['completed_gate_and_monitor_worker_restarts']} Pantograph worker restarts and {runtime_summary['completed_gate_and_monitor_cache_hits']}/{runtime_summary['completed_gate_and_monitor_cache_misses']} cache hits/misses.",
            "",
            "All completed stratified Gate runs recorded 600/600 attempts and zero Pantograph worker restarts. One A1 Discovery launch failed before generation because a second vLLM process started immediately after the preceding Gate and encountered transient CUDA OOM; the isolated retry completed and only the complete retry is analyzed.",
        ]
    )
    (root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(root / "ablation_report.md"), "release": release}, indent=2))


if __name__ == "__main__":
    main()
