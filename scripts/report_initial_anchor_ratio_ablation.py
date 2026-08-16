#!/usr/bin/env python3
"""Write the final human-readable initial-anchor ratio ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = (
    "BASE-ZERO",
    "CLEAN-M0",
    "ANCHOR-A0",
    "ANCHOR-A5",
    "ANCHOR-A10",
    "ANCHOR-A20",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f} pp"


def compact_hash(value: str) -> str:
    return f"`{value}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/initial_anchor_ratio_ablation"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    metrics = read_json(root / "comparisons/core_metrics.json")
    behaviors = read_json(root / "comparisons/core_behavior.json")
    paired = read_json(root / "comparisons/paired_statistics.json")
    decisions = read_json(root / "comparisons/promotion_decisions.json")
    full = read_json(root / "comparisons/full_metrics.json")
    runtime = read_json(root / "runtime/runtime_summary.json")
    manifest_audit = read_json(root / "audit/manifest_audit.json")
    checkpoint_hashes = read_json(root / "audit/checkpoint_hashes.json")
    expansion = read_json(root / "audit/ld_easy_expansion/expansion_summary.json")

    lines = [
        "# Initial Anchor WB / LD-easy Ratio Ablation",
        "",
        "## Executive conclusion",
        "",
        "The experiment completed all four independent one-epoch SFT arms and the "
        "entire mandatory core evaluation. Introducing manually reviewed LD-easy "
        "improved the in-domain LD-easy holdout monotonically, but every mixed arm "
        "lost too many WB Gate problems relative to A0. No mixed arm passed the "
        "pre-registered promotion gate, so Full500 and Strict unseen200 were "
        "correctly not run.",
        "",
        "The evidence does **not** support replacing CLEAN-M0 or using any mixed "
        "arm as the next production anchor. It supports a narrower conclusion: "
        "LD-easy exposure is learnable, but the tested row ratios and/or first-round "
        "training recipe trade away too much WB retention and induce substantial "
        "long-output/repetition drift.",
        "",
        "## Frozen data and training contract",
        "",
        f"- Newly accepted manual LD-easy rows: {expansion['new_trainable_easy']}",
        f"- Total trainable manual LD-easy rows: "
        f"{expansion['total_trainable_nonprotected_easy']}",
        "- Every arm: 3000 rows, one epoch, no replacement, one theorem group at most once.",
        "- All arms start independently from the same Qwen2.5-1.5B-Instruct Base.",
        "- A0/A5/A10/A20 LD row counts: 0 / 250 / 500 / 1000. Their actual "
        "row shares are 0% / 8.33% / 16.67% / 33.33%; A5/A10/A20 are retained "
        "only as legacy arm labels.",
        f"- Hard evaluation leaks: {manifest_audit['hard_evaluation_leaks']}",
        f"- Automatic difficulty labels admitted: "
        f"{manifest_audit['automatic_difficulty_labels']}.",
        f"- Base model weight SHA-256: "
        f"{compact_hash(manifest_audit['base_hashes']['model.safetensors'])}.",
        f"- Environment SHA-256: "
        f"{compact_hash(manifest_audit['environment_hash'])}.",
        "",
        "A20 could not simultaneously match A0's token budget under the frozen "
        "1000-LD-row, 3000-row, no-duplication and no-truncation constraints; "
        "the observed token difference is retained rather than hidden.",
        "",
        "## Manifest and token-budget audit",
        "",
        "| Arm | WB | LD | Actual LD share | Label tokens | Label Δ vs A0 | "
        "Total tokens | Total Δ vs A0 | Duplicate rows | Group duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    arm_names = (
        ("A0_WB3000_LD0", "A0"),
        ("A5_WB2750_LD250", "A5"),
        ("A10_WB2500_LD500", "A10"),
        ("A20_WB2000_LD1000", "A20"),
    )
    base_arm = manifest_audit["arms"]["A0_WB3000_LD0"]
    for arm, label in arm_names:
        audit = manifest_audit["arms"][arm]
        wb_rows = audit["sources"].get("WB", 0)
        ld_rows = audit["sources"].get("LD", 0)
        lines.append(
            f"| {label} | {wb_rows} | {ld_rows} | {pct(ld_rows / audit['rows'])} | "
            f"{audit['label_tokens']} | "
            f"{signed_pct(audit['label_tokens'] / base_arm['label_tokens'] - 1)} | "
            f"{audit['total_tokens']} | "
            f"{signed_pct(audit['total_tokens'] / base_arm['total_tokens'] - 1)} | "
            f"{audit['duplicate_rows']} | {audit['theorem_group_duplicates']} |"
        )
    lines.extend(
        [
            "",
            "All arms contain 3000 unique draws, max repeat 1, zero truncated rows, "
            "zero zero-label rows, and identical optimizer-step counts. A20's "
            "label-token difference is -5.21% (just outside the 5% target) and its "
            "total-token difference is -18.71% (outside the 8% target); this is a "
            "material confound when interpreting A20.",
            "",
            "## Teacher-forcing WB eval160",
            "",
            "| Model | Eval loss | Token accuracy |",
            "|---|---:|---:|",
        ]
    )
    for model in MODELS:
        loss = metrics[model].get("teacher_forcing_eval160", {})
        lines.append(
            f"| {model} | {loss.get('eval_loss', float('nan')):.6f} | "
            f"{pct(loss.get('eval_token_accuracy'))} |"
        )
    lines.extend(
        [
            "",
        "## Core evaluation",
        "",
        "| Model | Dataset | Solved | P@1 | P@2 | P@4 | Candidate success |",
        "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    dataset_labels = (
        ("wb_gate150", "WB Gate150"),
        ("ld_easy_holdout64", "LD-easy64"),
        ("monitor64", "Monitor64"),
        ("ld_hard128", "Hard-LD128"),
    )
    for model in MODELS:
        for dataset, label in dataset_labels:
            row = metrics[model][dataset]
            lines.append(
                f"| {model} | {label} | {row['solved']}/{row['statements']} | "
                f"{pct(row['pass_at_1'])} | {pct(row['pass_at_2'])} | "
                f"{pct(row['pass_at_4'])} | "
                f"{pct(row['candidate_success_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Generation behavior",
            "",
            "| Model | Dataset | Mean tokens | Length finish | Repetition | "
            "Duplicate candidates | Extraction | Format valid |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for dataset, label in dataset_labels:
            row = behaviors[model][dataset]
            lines.append(
                f"| {model} | {label} | {row['output_tokens']['mean']:.1f} | "
                f"{pct(row['length_finish_ratio'])} | "
                f"{pct(row['repetition_ratio'])} | "
                f"{pct(row['duplicate_candidate_ratio'])} | "
                f"{pct(row['proof_extraction_success_rate'])} | "
                f"{pct(row['format_validity_rate'])} |"
            )
    lines.extend(
        [
            "",
            "Relative to CLEAN-M0, every newly trained arm produces much longer "
            "outputs and reaches the 256-token limit far more often. On WB Gate, "
            f"CLEAN-M0 averages "
            f"{behaviors['CLEAN-M0']['wb_gate150']['output_tokens']['mean']:.1f} "
            f"tokens with {pct(behaviors['CLEAN-M0']['wb_gate150']['length_finish_ratio'])} "
            f"length finishes, versus "
            f"{behaviors['ANCHOR-A0']['wb_gate150']['output_tokens']['mean']:.1f}/"
            f"{pct(behaviors['ANCHOR-A0']['wb_gate150']['length_finish_ratio'])} "
            f"for A0 and "
            f"{behaviors['ANCHOR-A20']['wb_gate150']['output_tokens']['mean']:.1f}/"
            f"{pct(behaviors['ANCHOR-A20']['wb_gate150']['length_finish_ratio'])} "
            "for A20. This is a material generation-behavior drift, not merely an "
            "eval-loss difference.",
            "",
            "## Paired statistics",
            "",
            "| Dataset | Comparison | ΔP@4 | Bootstrap 95% CI | Wins | Losses | "
            "Ties | McNemar p |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for key, row in paired.items():
        dataset, comparison = key.split(":", 1)
        if dataset not in {item[0] for item in dataset_labels}:
            continue
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {dict(dataset_labels)[dataset]} | "
            f"{comparison.replace('_vs_', ' vs ')} | "
            f"{signed_pct(row['delta_pass_at_4'])} | "
            f"[{signed_pct(low)}, {signed_pct(high)}] | "
            f"{row['wins']} | {row['losses']} | {row['ties']} | "
            f"{row['mcnemar_exact_p_value']:.4f} |"
        )
    lines.extend(["", "## Promotion decisions", ""])
    for model, decision in decisions.items():
        lines.append(
            f"- {model}: LD-easy solved gain {decision['ld_easy_solved_gain']:+d}; "
            f"WB drop {decision['wb_solved_drop']:+d}; "
            f"Monitor drop {decision['monitor_solved_drop']:+d}; "
            f"full evaluation selected = "
            f"{decision['selected_for_full_evaluation']}."
        )
    if not any(
        decision["selected_for_full_evaluation"]
        for decision in decisions.values()
    ):
        lines.extend(
            [
                "",
                "No arm qualified. Per the pre-registered protocol, Full500 and "
                "Strict unseen200 were therefore not run; this is a gate outcome, "
                "not missing experimental work.",
            ]
        )
    if full:
        lines.extend(["", "## Conditional full evaluation", ""])
        for model, datasets in full.items():
            for dataset, row in datasets.items():
                lines.append(
                    f"- {model} / {dataset}: {row['solved']}/{row['statements']} "
                    f"solved, P@4={pct(row['pass_at_4'])}."
                )
    lines.extend(
        [
            "",
            "## Reproducibility artifacts",
            "",
            "| Arm | Manifest SHA-256 | Best-checkpoint SHA-256 |",
            "|---|---|---|",
        ]
    )
    for arm, label in arm_names:
        model = f"ANCHOR-{label}"
        lines.append(
            f"| {label} | {compact_hash(manifest_audit['manifest_hashes'][arm])} | "
            f"{compact_hash(checkpoint_hashes[model]['sha256'])} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Training time: {runtime['training_seconds_total'] / 3600:.2f} h",
            f"- Generation time: {runtime['generation_seconds_total'] / 3600:.2f} h",
            f"- Verification time: {runtime['verification_seconds_total'] / 3600:.2f} h",
            f"- Peak GPU reserved memory: "
            f"{runtime['gpu_peak_reserved_bytes'] / (1024 ** 3):.2f} GiB",
            f"- Peak coordinator/training RSS: "
            f"{runtime['process_max_rss_kib'] / (1024 ** 2):.2f} GiB",
            f"- Pantograph worker restarts: "
            f"{runtime['pantograph_worker_restarts_total']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Required questions and answers",
            "",
            "1. **Did A0 reproduce CLEAN-M0's main capability trend? No.** A0 "
            "solved 15/150 WB, 0/64 LD-easy and 2/64 Monitor problems, compared "
            "with CLEAN-M0's 41/150, 7/64 and 13/64. A0 also shows severe "
            "long-output drift. Because CLEAN-M0 is historical rather than a "
            "strict one-variable control, the gap points to a reproducibility "
            "difference in data identity, formatter/code version, optimizer "
            "contract or checkpoint construction that must be audited.",
            "",
            "2. **Was A5 better than pure WB? No overall.** The actual LD share "
            "was 8.33%, not 5%. It gained 1/64 LD-easy problem and 1/64 Monitor "
            "problem, but lost 7/150 WB problems; the WB paired ΔP@4 was "
            "-4.67 pp (95% CI -9.33 to 0.00 pp).",
            "",
            "3. **Was A10 better than pure WB? No overall.** The actual LD share "
            "was 16.67%. It gained 3/64 LD-easy problems, tied Monitor, and lost "
            "7/150 WB problems; its WB paired ΔP@4 was -4.67 pp.",
            "",
            "4. **Was A20 better than pure WB? No overall.** The actual LD share "
            "was 33.33%. It gained 4/64 LD-easy, 1/64 Monitor and 1/128 Hard-LD "
            "problems, but lost 10/150 WB problems. Its WB ΔP@4 was -6.67 pp "
            "(95% CI -11.33 to -2.67 pp, McNemar p=0.0063). The larger token-budget "
            "mismatch further weakens any causal claim about this arm.",
            "",
            "5. **Best tested ratio:** none satisfies the joint retention and "
            "transfer criteria. A0 is best for WB retention; A20 is best only for "
            "LD transfer. No single mixed recipe is a production winner.",
            "",
            "6. **Low-ratio positive, high-ratio degradation?** There is a "
            "monotonic LD-easy gain as LD rows increase, but WB retention degrades "
            "already at the lowest tested nonzero share. This is a transfer-versus-"
            "retention trade-off, not a clean low-ratio optimum.",
            "",
            "7. **Did LD-easy gain accompany WB/Monitor forgetting?** It "
            "consistently accompanied WB forgetting. Monitor did not show a "
            "monotonic loss (A5/A20 +1 solved, A10 tied), but all new arms remained "
            "far below CLEAN-M0.",
            "",
            "8. **Did generation behavior drift? Yes.** Outputs became much longer "
            "and more repetitive, with far more length-limit finishes. Duplicate "
            "candidate rates stayed low, so the main issue is within-output "
            "repetition/over-generation rather than four candidates collapsing "
            "to identical proofs.",
            "",
            "9. **Does this support introducing LD in the first round? Not with "
            "the tested recipe.** The experiment proves the model can learn some "
            "LD-easy transfer, but the retention cost violates the gate.",
            "",
            "10. **Fixed dataset baseline for a future LoRA-rank ablation:** do "
            "not use A5/A10/A20. First reproduce CLEAN-M0 using the frozen A0 "
            "manifest and current code. Only after the A0/CLEAN-M0 gap is resolved "
            "should that verified pure-WB manifest become the fixed rank-ablation "
            "dataset.",
            "",
            "11. **Replace CLEAN-M0? No.** CLEAN-M0 remains decisively stronger on "
            "all practical core gates.",
            "",
            "12. **Is a replication seed needed? Yes, but only after the A0 "
            "reproduction gap is diagnosed.** Repeating the current mixed arms "
            "before that audit would conflate seed variance with a larger pipeline "
            "or data-identity mismatch.",
            "",
            "## Recommended next action",
            "",
            "Freeze all current artifacts. Audit A0 against the historical "
            "CLEAN-M0 training manifest, formatter, packing behavior, optimizer "
            "schedule and merged-checkpoint construction. If exact reproduction "
            "becomes possible, replicate the pure-WB control with another seed; "
            "then test a genuinely low LD share below 8.33% with token-budget "
            "matching. Do not begin LoRA-rank ablation, Expert Iteration or GRPO "
            "from any model produced here.",
        ]
    )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
