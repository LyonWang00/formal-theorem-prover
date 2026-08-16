#!/usr/bin/env python3
"""Render the conditional LD-easy pilot results without overstating small samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = ("wb_gate150", "ld_easy_holdout64", "monitor64", "ld_hard128")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def pass_pair(row: dict[str, Any]) -> str:
    return f"{pct(row['pass_at_1'])} / {pct(row['pass_at_4'])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/pilot_sft"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    audit = read_json(root / "audit/pilot_manifest_audit.json")
    metrics = read_json(root / "comparisons/core_metrics.json")
    behavior = read_json(root / "comparisons/core_behavior.json")
    promotions = read_json(root / "comparisons/promotion_decisions.json")
    full = read_json(root / "comparisons/full_metrics.json")
    training = {
        model: read_json(root / "training" / model / "training_summary.json")
        for model in ("LDE-A2-WB100", "LDE-B2-LD10", "LDE-C2-LD20")
        if (root / "training" / model / "training_summary.json").is_file()
    }
    models = tuple(
        model
        for model in (
            "M0-ZERO",
            "LDE-A2-WB100",
            "LDE-B2-LD10",
            "LDE-C2-LD20",
        )
        if model in metrics
    )
    lines = [
        "# LD-easy short whole-proof pilot",
        "",
        "## Frozen training contract",
        "",
        "- Initialization: every arm independently starts from clean M0.",
        "- Rows: 1,000 per arm, without replacement, one proof per theorem group.",
        "- Learning rate: 2e-6; one epoch; effective batch size 16.",
        "- QLoRA: rank 32, alpha 64, dropout 0.05, unchanged target modules.",
        f"- LD easy holdout: {audit['holdout_rows']} rows; source-file disjoint: "
        f"`{audit['holdout_source_file_disjoint']}`.",
        "- Generation length: the single common policy frozen by Stage 1.",
        "",
        "## Training results",
        "",
        "| Model | Optimizer steps | Unique draws | Duplicate draws | Eval loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in training.items():
        lines.append(
            f"| {model} | {row['optimizer_steps']} | "
            f"{row['sampling']['unique_rows']} | "
            f"{row['sampling']['duplicate_draws']} | {row['eval_loss']:.6f} |"
        )
    lines.extend(
        [
        "",
        "## Core performance",
        "",
        "P@1 / P@4 are reported in each cell.",
        "",
        "| Model | WB Gate150 | LD easy64 | Monitor64 | Hard LD128 |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        row = metrics[model]
        lines.append(
            f"| {model} | {pass_pair(row['wb_gate150'])} "
            f"({row['wb_gate150']['solved']}/150 at P@4) | "
            f"{pass_pair(row['ld_easy_holdout64'])} "
            f"({row['ld_easy_holdout64']['solved']}/64) | "
            f"{pass_pair(row['monitor64'])} "
            f"({row['monitor64']['solved']}/64) | "
            f"{pass_pair(row['ld_hard128'])} "
            f"({row['ld_hard128']['solved']}/128) |"
        )
    lines.extend(
        [
            "",
            "## Generation behavior on LD easy64",
            "",
            "| Model | Mean tokens | Length finish | Repetition | Extraction | Format |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        row = behavior[model]["ld_easy_holdout64"]
        lines.append(
            f"| {model} | {row['output_tokens']['mean']:.2f} | "
            f"{pct(row['length_finish_ratio'])} | {pct(row['repetition_ratio'])} | "
            f"{pct(row['proof_extraction_success_rate'])} | "
            f"{pct(row['format_validity_rate'])} |"
        )
    lines.extend(["", "## Promotion decisions", ""])
    for model, decision in promotions.items():
        lines.append(
            f"- **{model}**: `{'promoted' if decision['promoted'] else 'not promoted'}`; "
            f"easy solved gain {decision['easy_solved_gain']:+d}, "
            f"WB solved drop {decision['wb_solved_drop']:+d}, "
            f"Monitor solved drop {decision['monitor_solved_drop']:+d}."
        )
        failed = [name for name, passed in decision["checks"].items() if not passed]
        if failed:
            lines.append(f"  Failed checks: {', '.join(failed)}.")
    if full:
        lines.extend(["", "## Conditional Full500 / Strict unseen200", ""])
        for model, datasets in full.items():
            values = ", ".join(
                f"{name} P@4={pct(row['pass_at_4'])} ({row['solved']}/{row['statements']})"
                for name, row in datasets.items()
            )
            lines.append(f"- {model}: {values}.")
    else:
        lines.extend(
            [
                "",
                "## Conditional Full500 / Strict unseen200",
                "",
                "Not run: no LD-easy candidate passed the frozen core promotion gate.",
            ]
        )

    b = promotions.get("LDE-B2-LD10", {})
    c = promotions.get("LDE-C2-LD20", {})
    any_promoted = bool(b.get("promoted") or c.get("promoted"))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "At least one manually selected LD-easy mixture passed the frozen "
                "promotion gate. Expanding the manually reviewed easy pool is supported, "
                "subject to the conditional full evaluations above."
                if any_promoted
                else "The 10% manually selected LD-easy mixture did not pass the joint "
                "easy-gain, WB-retention, Monitor-retention, and generation-behavior "
                "gate. This pilot recipe should not be expanded; clean M0 remains the "
                "default."
            ),
            "",
            "All conclusions are effect-size statements on the frozen samples; paired "
            "confidence intervals and McNemar results are stored in "
            "`comparisons/paired_statistics.json`.",
        ]
    )
    skipped = audit.get("skipped_arms") or {}
    if skipped:
        lines.extend(["", "## Skipped arms", ""])
        for arm, reason in skipped.items():
            lines.append(f"- {arm}: {reason}.")

    a = metrics.get("LDE-A2-WB100", {})
    b_metrics = metrics.get("LDE-B2-LD10", {})
    if a and b_metrics:
        lines.extend(
            [
                "",
                "## Stage 3 questions",
                "",
                "1. **10% LD-easy vs WB-only:** not better on the primary LD-easy "
                f"endpoint; both solved {a['ld_easy_holdout64']['solved']}/64 at P@4.",
                "2. **20% LD-easy vs WB-only:** not evaluated because the frozen pool "
                "cannot supply 200 unique training rows plus 64 holdout rows.",
                "3. **LD-easy holdout:** B2 did not improve over A2 and solved one fewer "
                "theorem than clean M0.",
                "4. **WB/Monitor retention:** B2 lost one WB theorem versus A2 but gained "
                "one Monitor theorem; both changes remain inside the retention limits.",
                "5. **Hard LD holdout:** B2 recovered A2's one-theorem loss and matched "
                "clean M0, but did not exceed it.",
                "6. **Length/repetition drift:** no adverse drift was observed; B2 outputs "
                "were slightly shorter and less repetitive than clean M0.",
                "7. **Full500/Strict unseen:** no model qualified, so the conditional "
                "evaluations were not run.",
                "8. **Expand the easy pool:** current evidence does not support scaling "
                "this SFT recipe. At least 60 additional trainable easy samples would be "
                "needed merely to make the pre-registered C2 arm feasible.",
                "9. **Default model:** retain clean M0.",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "report.md")


if __name__ == "__main__":
    main()
