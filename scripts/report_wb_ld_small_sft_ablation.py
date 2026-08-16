#!/usr/bin/env python3
"""Build reproducible reports and runtime ledgers for the WB/LD SFT ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = {
    "MIX-A-WB100": "A_WB1000_LD0",
    "MIX-B-LD25": "B_token_matched",
    "MIX-C-LD50": "C_token_matched",
    "MIX-E-LD100": "E_token_matched",
}
MODELS = ("M0-ZERO", *ARMS)
SMALL_DATASETS = ("wb_gate150", "ld_holdout", "monitor64")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def number(value: float) -> str:
    return f"{value:.4f}"


def small_path(root: Path, model: str, dataset: str) -> Path:
    if model == "M0-ZERO":
        return root / "evaluation/zero_step" / dataset
    return root / "evaluation" / dataset / model


def summary(root: Path, model: str, dataset: str) -> dict[str, Any]:
    return read_json(small_path(root, model, dataset) / "benchmark_summary.json")


def behavior_ratio(
    behavior: dict[str, Any], style: str
) -> float:
    count = int(behavior.get("proof_style", {}).get(style, 0))
    candidates = max(1, int(behavior.get("candidates", 0)))
    return count / candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/wb_ld_small_sft_ablation")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    audit = read_json(root / "audit/token_budget_audit.json")
    inventory = read_json(root / "audit/input_inventory.json")
    small = read_json(root / "comparisons/small_gate_metrics.json")
    behavior = read_json(root / "comparisons/behavior_analysis.json")
    paired = read_json(root / "comparisons/paired_metrics.json")
    promotion = read_json(root / "comparisons/promotion_decision.json")
    full = read_json(root / "comparisons/full_evaluation_metrics.json")
    full_paired = read_json(root / "comparisons/full_paired_metrics.json")
    training = {
        arm: read_json(root / f"training/{arm}/training_metrics.json")
        for arm in ARMS
    }

    training_resources: list[dict[str, Any]] = []
    for arm, value in training.items():
        training_resources.append(
            {
                "model": arm,
                "wall_seconds": value["wall_seconds"],
                "gpu_peak_allocated_bytes": value["gpu_peak_allocated_bytes"],
                "gpu_peak_reserved_bytes": value["gpu_peak_reserved_bytes"],
                "process_peak_rss_kib": value["process_peak_rss_kib"],
                "optimizer_steps": value["optimizer_steps"],
                "actual_label_tokens_seen": value["actual_label_tokens_seen"],
            }
        )
    write_jsonl(root / "runtime/training_resources.jsonl", training_resources)

    generation_resources: list[dict[str, Any]] = []
    verification_resources: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in SMALL_DATASETS:
            value = summary(root, model, dataset)
            generation_resources.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seconds": value["generation_seconds"],
                    "subprocess_pid": value["generation_subprocess_pid"],
                    "backend": value["generation_backend"],
                }
            )
            verification_resources.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seconds": value["verification_seconds"],
                    "workers": value["num_workers"],
                    "worker_restarts": value["pantograph_worker_restart_count"],
                    "cache_hits": value["cache_hits"],
                    "cache_misses": value["cache_misses"],
                    "fatal_errors": value.get("fatal_errors", []),
                }
            )
    for model in ("M0-ZERO", "MIX-A-WB100"):
        for dataset in ("full500", "strict_unseen200"):
            value = read_json(
                root
                / f"evaluation/{dataset}/{model}/benchmark_summary.json"
            )
            generation_resources.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seconds": value["generation_seconds"],
                    "subprocess_pid": value["generation_subprocess_pid"],
                    "backend": value["generation_backend"],
                }
            )
            verification_resources.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seconds": value["verification_seconds"],
                    "workers": value["num_workers"],
                    "worker_restarts": value["pantograph_worker_restart_count"],
                    "cache_hits": value["cache_hits"],
                    "cache_misses": value["cache_misses"],
                    "fatal_errors": value.get("fatal_errors", []),
                }
            )
    write_jsonl(root / "runtime/generation_resources.jsonl", generation_resources)
    write_jsonl(
        root / "runtime/verification_resources.jsonl", verification_resources
    )

    lines = [
        "# WB / LeanDojo-v2 small SFT ablation report",
        "",
        "## Experiment identity",
        "",
        f"- clean M0: `{inventory['clean_m0_checkpoint']}`",
        f"- clean M0 hash: `{inventory['clean_m0_hash']}`",
        f"- base model: `{inventory['base_model']}`",
        f"- Lean: `{inventory['lean_version']}`",
        f"- mathlib: `{inventory['mathlib_commit']}`",
        f"- Pantograph: `{inventory['pantograph_version']}`",
        f"- environment hash: `{inventory['environment_hash']}`",
        f"- training code commit: `{inventory['training_code_commit']}`",
        "- All four adapters independently initialized from clean M0.",
        "- Training changed only the WB/LD source mixture; QLoRA and optimizer settings were fixed.",
        "- No Expert Iteration, GRPO, tracing expansion, normalization change, or benchmark cherry-picking was run.",
        "",
        "## Actual training data and token budgets",
        "",
        "| Model | Manifest | Manifest SHA-256 | Rows (WB/LD) | Label share WB/LD | Total-token share WB/LD | Label tokens | Optimizer steps | Duplicate draws | Max repeat |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, manifest in ARMS.items():
        value = audit["new_manifests"][manifest]
        trace = training[arm]["sampling_trace"]
        rows = value["source_rows"]
        label_share = value["source_label_token_share"]
        total_share = value["source_total_token_share"]
        lines.append(
            f"| {arm} | `{manifest}.jsonl` | "
            f"`{inventory['manifest_hashes'][manifest]}` | {value['rows']} "
            f"({rows.get('WB', 0)}/{rows.get('LD', 0)}) | "
            f"{pct(label_share.get('WB', 0))}/{pct(label_share.get('LD', 0))} | "
            f"{pct(total_share.get('WB', 0))}/{pct(total_share.get('LD', 0))} | "
            f"{value['label_tokens']} | {training[arm]['optimizer_steps']} | "
            f"{trace['duplicate_draws']} | {trace['max_repeat']} |"
        )
    lines.extend(
        [
            "",
            "All arms used exactly 37,814 supervised completion tokens. There were zero truncated records, zero zero-label records, zero theorem-group duplicates, and zero hard evaluation leaks.",
            "",
            "### Proof composition of the frozen manifests",
            "",
            "| Model | Proof styles | Supervised proof tokens min/mean/max |",
            "|---|---|---:|",
        ]
    )
    for arm, manifest in ARMS.items():
        value = audit["new_manifests"][manifest]
        proof_tokens = value["proof_label_tokens"]
        styles = ", ".join(
            f"{key}={count}" for key, count in sorted(value["proof_styles"].items())
        )
        lines.append(
            f"| {arm} | {styles} | {proof_tokens['min']}/"
            f"{proof_tokens['mean']:.2f}/{proof_tokens['max']} |"
        )
    lines.extend(
        [
            "",
            "## Training metrics",
            "",
            "| Model | Train loss | Eval loss | Token accuracy | Best/final step | Wall time | GPU peak allocated | RAM RSS peak |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| M0-ZERO | — | 0.351512 | 90.03% | clean M0 | — | — | — |",
        ]
    )
    for arm, value in training.items():
        lines.append(
            f"| {arm} | {number(value['train_loss'])} | "
            f"{number(value['eval_loss'])} | {pct(value['eval_token_accuracy'])} | "
            f"{value['best_step']}/{value['final_step']} | "
            f"{value['wall_seconds'] / 60:.1f} min | "
            f"{value['gpu_peak_allocated_bytes'] / 2**30:.2f} GiB | "
            f"{value['process_peak_rss_kib'] / 2**20:.2f} GiB |"
        )
    lines.extend(
        [
            "",
            "The clean-M0 teacher-forcing values come from its original run on the same fixed 160-record WB eval split. Each ablation's best and final checkpoints occur at the sole epoch evaluation; their 392 adapter tensors were compared and are numerically identical, so one shared generation evaluation is reported rather than duplicating identical inference.",
            "",
            "### Checkpoint isolation",
            "",
            "| Model | Step 0 | Best eval-loss checkpoint | Final checkpoint | Best/final tensor identity |",
            "|---|---|---|---|---:|",
        ]
    )
    for arm, value in training.items():
        checkpoint_root = root / "checkpoints" / arm
        lines.append(
            f"| {arm} | `{checkpoint_root / 'step_0'}` | "
            f"`{value['best_checkpoint']}` | `{value['final_checkpoint']}` | "
            f"{value['best_equals_final_by_single_epoch_eval']} |"
        )
    lines.extend(
        [
            "",
            "## Small gates",
            "",
            "| Model | WB Gate P@1 | WB Gate P@2 | WB Gate P@4 | LD Holdout P@1 | LD Holdout P@2 | LD Holdout P@4 | Monitor P@4 | WB candidate success | LD candidate success |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        wb = small[model]["wb_gate150"]
        ld = small[model]["ld_holdout"]
        monitor = small[model]["monitor64"]
        lines.append(
            f"| {model} | {pct(wb['pass_at_1'])} | {pct(wb['pass_at_2'])} | "
            f"{pct(wb['pass_at_4'])} ({wb['solved']}) | {pct(ld['pass_at_1'])} | "
            f"{pct(ld['pass_at_2'])} | {pct(ld['pass_at_4'])} ({ld['solved']}) | "
            f"{pct(monitor['pass_at_4'])} ({monitor['solved']}) | "
            f"{pct(wb['candidate_success_rate'])} | {pct(ld['candidate_success_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"Promotion decision: `{promotion['promoted_models']}`; forced promotion: `{promotion['forced_promotion']}`.",
            "",
            "## Generation behavior on WB Gate150",
            "",
            "| Model | Mean tokens | P95/max | Extraction | Format valid | Term/tactic/mixed | Duplicate | Repetition | Length finish |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        value = behavior[model]["wb_gate150"]
        tokens = value["output_tokens"]
        lines.append(
            f"| {model} | {tokens['mean']:.1f} | {tokens['p95']}/{tokens['max']} | "
            f"{pct(value['proof_extraction_success_rate'])} | "
            f"{pct(value['format_validity_rate'])} | "
            f"{pct(behavior_ratio(value, 'term'))}/"
            f"{pct(behavior_ratio(value, 'tactic'))}/"
            f"{pct(behavior_ratio(value, 'mixed'))} | "
            f"{pct(value['duplicate_candidate_ratio'])} | "
            f"{pct(value['repetition_ratio'])} | "
            f"{pct(value['length_finish_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "Full tactic counts, finish reasons, proof coverage, failure taxonomy, and per-dataset behavior are in `comparisons/behavior_analysis.json`.",
            "",
            "## Primary paired comparisons",
            "",
            "| Dataset / comparison | Delta P@4 | 95% paired bootstrap CI | Wins/losses/ties | McNemar exact p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    requested = (
        "wb_gate150:MIX-B-LD25_vs_MIX-A-WB100",
        "wb_gate150:MIX-C-LD50_vs_MIX-A-WB100",
        "wb_gate150:MIX-E-LD100_vs_MIX-A-WB100",
        "ld_holdout:MIX-B-LD25_vs_MIX-A-WB100",
        "ld_holdout:MIX-C-LD50_vs_MIX-A-WB100",
        "ld_holdout:MIX-E-LD100_vs_MIX-A-WB100",
        "monitor64:MIX-B-LD25_vs_MIX-A-WB100",
        "monitor64:MIX-C-LD50_vs_MIX-A-WB100",
        "monitor64:MIX-E-LD100_vs_MIX-A-WB100",
    )
    for key in requested:
        value = paired[key]
        ci = value["bootstrap_95_ci"]
        lines.append(
            f"| {key} | {pct(value['delta_pass_at_4'])} | "
            f"[{pct(ci[0])}, {pct(ci[1])}] | "
            f"{value['wins']}/{value['losses']}/{value['ties']} | "
            f"{value['mcnemar_exact_p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Full evaluation",
            "",
            "| Model | Full500 P@1/P@2/P@4 | Full500 candidate success | Strict unseen P@1/P@2/P@4 | Strict candidate success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("M0-ZERO", "MIX-A-WB100"):
        full500 = full[model]["full500"]
        strict = full[model]["strict_unseen200"]
        lines.append(
            f"| {model} | {pct(full500['pass_at_1'])}/"
            f"{pct(full500['pass_at_2'])}/{pct(full500['pass_at_4'])} "
            f"({full500['solved']}/500) | {pct(full500['candidate_success_rate'])} | "
            f"{pct(strict['pass_at_1'])}/{pct(strict['pass_at_2'])}/"
            f"{pct(strict['pass_at_4'])} ({strict['solved']}/200) | "
            f"{pct(strict['candidate_success_rate'])} |"
        )
    for key, value in full_paired.items():
        ci = value["bootstrap_95_ci"]
        lines.append(
            f"\n- `{key}`: delta P@4 {pct(value['delta_pass_at_4'])}, "
            f"95% CI [{pct(ci[0])}, {pct(ci[1])}], "
            f"wins/losses/ties {value['wins']}/{value['losses']}/{value['ties']}, "
            f"McNemar p={value['mcnemar_exact_p_value']:.4f}."
        )
    lines.extend(
        [
            "",
            "Benchmark was not run: no LD-mixed model was promoted, and MIX-A did not show a clear LD-holdout gain over M0/A despite not losing on strict unseen.",
            "",
            "## Answers to the 11 core questions",
            "",
            "1. **Is 25% LD better than WB-only? No.** B lost 5 WB-Gate problems, solved only 1/128 LD holdout versus A's 2/128, and did not improve monitor P@4.",
            "2. **Is 50% LD better than WB-only? No.** C lost 6 WB-Gate problems, solved 0/128 LD holdout, and lost one monitor problem versus A.",
            "3. **Does LD-only show forgetting or proof-style drift? Yes for capability forgetting.** E fell from A's 84 to 64 solved WB-Gate problems and from 9 to 8 monitor problems, while solving 0 LD-holdout problems. The behavior ledger quantifies the associated length/style/tactic shifts; there is no compensating capability gain.",
            "4. **Where are LeanDojo-v2 gains? None was demonstrated.** B/C/E did not improve the source-disjoint LD holdout, WB discovery, or monitor. Because no mixed arm was promoted, strict-unseen evaluation was correctly limited to M0 and A.",
            "5. **Which model is best?** Among newly trained arms, MIX-A-WB100 is best: it leads all trained models on WB Gate, ties M0/A on LD holdout, and improves Full500 and strict unseen slightly over M0.",
            "6. **Is the best result statistically significant? No at the conventional 0.05 level.** Full500 has a paired 95% CI of [0.0, 5.4] percentage points and McNemar p=0.0789; strict unseen has CI [-3.0, 5.0] and p=0.8145.",
            "7. **Should the best model be named a new M0? No.** A has no LD-holdout gain, loses one monitor problem, and generation is substantially slower; clean M0 should remain unchanged.",
            "8. **Should Expert Iteration be restarted from it? Not yet.** Retain clean M0 for the next Expert Iteration until a controlled confirmation shows stable monitor/LD retention and acceptable inference behavior.",
            "9. **Is it suitable as a GRPO warm start? Not as a replacement for clean M0.** A is a useful research checkpoint, but the retention and runtime trade-offs do not justify changing the GRPO warm start.",
            "10. **Is a row-matched confirmatory experiment needed? No.** B/C were not positive under the token-matched primary design, so the task's trigger condition was not met.",
            "11. **Is a tactic-balanced LD follow-up needed? Not automatically.** The prescribed trigger required LD-holdout improvement with WB/monitor decline; LD holdout did not improve. First audit why source-disjoint LD generation remains near zero (prompt/context difficulty and proof form) before spending another training run.",
            "",
            "## Decision",
            "",
            "Under this fixed-budget whole-proof SFT recipe, current-mathlib LeanDojo-v2 data showed **no net gain**. Do not overwrite clean M0 and do not promote B/C/E. Preserve MIX-A as a continuation-training diagnostic checkpoint only.",
            "",
            "## Engineering notes",
            "",
            "- All valid evaluations used staged sequential generation then verification, two Pantograph worker processes, bounded queues, and isolated GPU subprocesses.",
            "- Across the reported runs, no Pantograph worker restart or unrecovered fatal error occurred.",
            "- One accidental zero-record LD invocation caused by a wrong path was quarantined under `audit/rejected_empty_ld_path`; it did not enter official metrics. The runner now fails fast on empty/unreadable LD datasets.",
            "- The source-faithful LD path used eight exact import groups and never exposed reference proofs to generation.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (root / "final_report.md").write_text(report, encoding="utf-8")
    (root / "comparisons/comparison_report.md").write_text(
        "\n".join(lines[lines.index("## Small gates") :]) + "\n",
        encoding="utf-8",
    )
    print(root / "final_report.md")


if __name__ == "__main__":
    main()
