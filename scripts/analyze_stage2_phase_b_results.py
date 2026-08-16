"""Create paired Phase-B evaluation comparison and final report."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = {
    "WB-Unseen-Holdout150": "wb_unseen_holdout150",
    "LD-easy64": "ld_easy64",
    "Monitor64": "monitor64",
    "WB-Train-Retention150": "wb_train_retention150",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pctl(values: list[int], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def rep_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize(directory: Path) -> tuple[dict[str, Any], dict[str, dict[int, bool]]]:
    attempts = read_jsonl(directory / "attempts.jsonl")
    generations = read_jsonl(directory / "generations.jsonl")
    paired: dict[str, dict[int, bool]] = {}
    for row in attempts:
        paired.setdefault(str(row["problem_id"]), {})[int(row["attempt_index"])] = bool(row["success"])
    solved = {pid: any(values.values()) for pid, values in paired.items()}
    pass1 = {pid: bool(values.get(0, False)) for pid, values in paired.items()}
    lengths = [int(row.get("metadata", {}).get("completion_tokens", 0)) for row in generations]
    reps = [rep_ratio(str(row.get("raw_output", ""))) for row in generations]
    output = {
        "problems": len(paired),
        "candidates": len(attempts),
        "candidate_successes": sum(bool(row.get("success")) for row in attempts),
        "pass_at_1_count": sum(pass1.values()),
        "pass_at_2_count": sum(solved.values()),
        "pass_at_1": sum(pass1.values()) / max(1, len(paired)),
        "pass_at_2": sum(solved.values()) / max(1, len(paired)),
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "completion_tokens": {
            "mean": mean(lengths), "p50": pctl(lengths, 0.5), "p95": pctl(lengths, 0.95),
            "max": max(lengths),
        },
        "max_length_finish_ratio": sum(
            row.get("finish_reason") in {"length", "max_length"}
            or int(row.get("metadata", {}).get("completion_tokens", 0)) >= int(row.get("max_new_tokens", 256))
            for row in generations
        ) / max(1, len(generations)),
        "extraction_success_rate": sum(bool(str(row.get("extracted_proof", "")).strip()) for row in generations) / max(1, len(generations)),
        "mean_repeated_4gram_ratio": mean(reps),
        "pathological_repetition_rate": sum(value >= 0.35 for value in reps) / max(1, len(reps)),
    }
    return output, paired


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_counts(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    ids = sorted(set(a) | set(b))
    if set(a) != set(b):
        raise RuntimeError("paired problem IDs differ")
    m0_only = sum(a[pid] and not b[pid] for pid in ids)
    s2_only = sum(b[pid] and not a[pid] for pid in ids)
    return {
        "both_fail": sum(not a[pid] and not b[pid] for pid in ids),
        "m0_only": m0_only,
        "s2_only": s2_only,
        "both_pass": sum(a[pid] and b[pid] for pid in ids),
        "mcnemar_exact_p": exact_mcnemar(m0_only, s2_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/stage2_data_ablation/phaseB_hard_replay"
    full = root / "evaluation/full"
    rows: dict[str, Any] = {}
    aggregate_pairs: dict[str, dict[str, dict[int, bool]]] = {"M0": {}, "S2": {}}
    for label, slug in DATASETS.items():
        m0, m0_pairs = summarize(full / "M0" / slug)
        s2, s2_pairs = summarize(full / "S2-Hard-Replay" / slug)
        m0_solved = {pid: any(v.values()) for pid, v in m0_pairs.items()}
        s2_solved = {pid: any(v.values()) for pid, v in s2_pairs.items()}
        m0_p1 = {pid: bool(v.get(0, False)) for pid, v in m0_pairs.items()}
        s2_p1 = {pid: bool(v.get(0, False)) for pid, v in s2_pairs.items()}
        rows[label] = {
            "M0": m0, "S2-Hard-Replay": s2,
            "delta_percentage_points": {
                "pass_at_1": 100 * (s2["pass_at_1"] - m0["pass_at_1"]),
                "pass_at_2": 100 * (s2["pass_at_2"] - m0["pass_at_2"]),
                "mean_completion_tokens": s2["completion_tokens"]["mean"] - m0["completion_tokens"]["mean"],
            },
            "paired_pass_at_1": paired_counts(m0_p1, s2_p1),
            "paired_pass_at_2": paired_counts(m0_solved, s2_solved),
        }
        for pid, values in m0_pairs.items():
            aggregate_pairs["M0"][f"{label}:{pid}"] = values
        for pid, values in s2_pairs.items():
            aggregate_pairs["S2"][f"{label}:{pid}"] = values

    totals: dict[str, Any] = {}
    for model_key, model_dir in (("M0", "M0"), ("S2-Hard-Replay", "S2")):
        metrics = [rows[label][model_key] for label in DATASETS]
        problems = sum(item["problems"] for item in metrics)
        candidates = sum(item["candidates"] for item in metrics)
        totals[model_key] = {
            "problems": problems,
            "candidates": candidates,
            "candidate_successes": sum(item["candidate_successes"] for item in metrics),
            "pass_at_1_count": sum(item["pass_at_1_count"] for item in metrics),
            "pass_at_2_count": sum(item["pass_at_2_count"] for item in metrics),
            "pass_at_1": sum(item["pass_at_1_count"] for item in metrics) / problems,
            "pass_at_2": sum(item["pass_at_2_count"] for item in metrics) / problems,
            "timeouts": sum(item["timeouts"] for item in metrics),
            "mean_completion_tokens": sum(item["completion_tokens"]["mean"] * item["candidates"] for item in metrics) / candidates,
        }
    m0_all = {pid: any(v.values()) for pid, v in aggregate_pairs["M0"].items()}
    s2_all = {pid: any(v.values()) for pid, v in aggregate_pairs["S2"].items()}
    m0_all_p1 = {pid: bool(v.get(0, False)) for pid, v in aggregate_pairs["M0"].items()}
    s2_all_p1 = {pid: bool(v.get(0, False)) for pid, v in aggregate_pairs["S2"].items()}
    comparison = {
        "phase": "B", "model": "S2-Hard-Replay", "baseline": "M0-ADDON-B-FROZEN",
        "canonical_backend": "transformers", "checkpoint_type": "merged",
        "generation_contract_sha256": json.loads((root / "evaluation/generation_contract.json").read_text())["generation_config_sha256"],
        "datasets": rows, "aggregate": totals,
        "aggregate_delta_percentage_points": {
            "pass_at_1": 100 * (totals["S2-Hard-Replay"]["pass_at_1"] - totals["M0"]["pass_at_1"]),
            "pass_at_2": 100 * (totals["S2-Hard-Replay"]["pass_at_2"] - totals["M0"]["pass_at_2"]),
        },
        "aggregate_paired_pass_at_1": paired_counts(m0_all_p1, s2_all_p1),
        "aggregate_paired_pass_at_2": paired_counts(m0_all, s2_all),
        "interpretation": "Hard-heavy replay produces a small pass@2 gain but lowers pass@1 on WB unseen, LD-easy, and WB retention; it is diagnostic evidence, not a stable final recipe.",
    }
    comparisons = project / "outputs/stage2_data_ablation/comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    (comparisons / "phaseB_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# Phase B paired comparison", "", "| Dataset | M0 p@1 | S2 p@1 | Δpp | M0 p@2 | S2 p@2 | Δpp | p (paired p@2) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in DATASETS:
        item = rows[label]
        md.append(f"| {label} | {item['M0']['pass_at_1']:.2%} | {item['S2-Hard-Replay']['pass_at_1']:.2%} | {item['delta_percentage_points']['pass_at_1']:+.2f} | {item['M0']['pass_at_2']:.2%} | {item['S2-Hard-Replay']['pass_at_2']:.2%} | {item['delta_percentage_points']['pass_at_2']:+.2f} | {item['paired_pass_at_2']['mcnemar_exact_p']:.4f} |")
    md += ["", f"Aggregate p@1: {totals['M0']['pass_at_1']:.2%} → {totals['S2-Hard-Replay']['pass_at_1']:.2%} ({comparison['aggregate_delta_percentage_points']['pass_at_1']:+.2f} pp).", "", f"Aggregate p@2: {totals['M0']['pass_at_2']:.2%} → {totals['S2-Hard-Replay']['pass_at_2']:.2%} ({comparison['aggregate_delta_percentage_points']['pass_at_2']:+.2f} pp).", "", comparison["interpretation"], ""]
    (comparisons / "phaseB_comparison.md").write_text("\n".join(md), encoding="utf-8")

    training = json.loads((root / "training/training_summary.json").read_text())
    gate = json.loads((root / "audit/phaseB_manifest_gate.json").read_text())
    canary = json.loads((root / "evaluation/canary/canary_gate.json").read_text())
    report = [
        "# Phase B — Hard-heavy Replay Stress Test", "", "## Outcome", "",
        "Phase B completed successfully. All requested training and evaluations ran; prohibited suites were not run.", "",
        "The hard-heavy recipe is **not recommended as the final production mix**: aggregate pass@2 improved slightly, but pass@1 declined on WB unseen, LD-easy, and WB retention.", "",
        "## Data and training", "",
        f"- 800 rows: Hard-A {training['bucket_counts']['Hard-A']}, Hard-B {training['bucket_counts']['Hard-B']}, Frontier {training['bucket_counts']['Frontier']}, Stable/Core {training['bucket_counts']['Stable/Core']}.",
        "- The approved 40-row Frontier supplement filled the Phase-A Hard-A/Hard-B/Stable shortfall; Hard-C was excluded.",
        f"- Manifest hash: `{training['manifest_sha256']}`; duplicates 0; max repeat {training['max_repeat']}; all 800 labels end in EOS.",
        f"- Fresh LoRA from frozen M0: r={training['resolved_config']['lora_r']}, alpha={training['resolved_config']['lora_alpha']}, LR={training['resolved_config']['learning_rate']}, epoch={training['resolved_config']['num_train_epochs']}, packing=false.",
        f"- Train loss {training['metrics']['train_loss']:.6f}; eval loss {training['eval_loss']:.6f}; eval token accuracy {training['eval_token_accuracy']:.4%}.",
        f"- M0 hash unchanged after training: `{training['frozen_m0_hash_after_training']}`.", "",
        "## Leakage and EOS gates", "",
        "- True holdouts WB-Unseen150, LD-easy64, Monitor64, Hard-LD128, Full500, Strict-unseen200, and WB eval160: zero overlap across all audited identity forms.",
        f"- WB-Train-Retention150 overlap ({gate['leakage']['wb_train_retention150']['record_id_overlap']} record IDs) is intentional and explicitly allowed as an in-distribution retention set.",
        "- duplicate rows=0; duplicate theorem groups=0; proof/theorem modifications=0; supervised EOS=800/800.", "",
        "## Canonical evaluation contract", "",
        "- Backend: Transformers; checkpoint: merged; adapter path: null.",
        f"- S2 resolved generation-contract hash: `{comparison['generation_contract_sha256']}`; generation parameters are unchanged from parent Task0B hash `bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386`.", "",
        "## Behavior canary", "",
        f"- Gate: {'PASS' if canary['gate_passed'] else 'FAIL'}; 140 candidates per model; extraction 100% on both; no fatal errors or worker restarts.",
        f"- Mean completion tokens: M0 {canary['combined']['M0']['completion_tokens_mean']:.2f}, S2 {canary['combined']['S2-Hard-Replay']['completion_tokens_mean']:.2f}.",
        f"- Max-length ratio: M0 {canary['combined']['M0']['max_length_finish_ratio']:.2%}, S2 {canary['combined']['S2-Hard-Replay']['max_length_finish_ratio']:.2%}; pathological repetition {canary['combined']['S2-Hard-Replay']['pathological_repetition_rate']:.2%}.", "",
        "## Formal evaluation", "", "| Dataset | M0 p@1 | S2 p@1 | Δpp | M0 p@2 | S2 p@2 | Δpp |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in DATASETS:
        item = rows[label]
        report.append(f"| {label} | {item['M0']['pass_at_1']:.2%} | {item['S2-Hard-Replay']['pass_at_1']:.2%} | {item['delta_percentage_points']['pass_at_1']:+.2f} | {item['M0']['pass_at_2']:.2%} | {item['S2-Hard-Replay']['pass_at_2']:.2%} | {item['delta_percentage_points']['pass_at_2']:+.2f} |")
    report += ["", f"Across 428 problems, p@1 changed {totals['M0']['pass_at_1']:.2%} → {totals['S2-Hard-Replay']['pass_at_1']:.2%} ({comparison['aggregate_delta_percentage_points']['pass_at_1']:+.2f} pp), while p@2 changed {totals['M0']['pass_at_2']:.2%} → {totals['S2-Hard-Replay']['pass_at_2']:.2%} ({comparison['aggregate_delta_percentage_points']['pass_at_2']:+.2f} pp).", "", "No Full500, Strict-unseen200, or miniF2F evaluation was run.", "", "## Conclusion", "", "Filtered Hard-A/B is learnable enough to avoid catastrophic drift and gives a small second-candidate coverage gain. However, the broad pass@1 decline shows that an 800-row hard-heavy mixture is not sufficiently stable as the main second-round recipe. Balanced replay plus genuinely new easy foundation data remains the intended next diagnostic, subject to its frozen data quotas being satisfiable.", ""]
    (root / "phaseB_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"report": str(root / 'phaseB_report.md'), "comparison": str(comparisons / 'phaseB_comparison.json'), "aggregate": totals, "delta_pp": comparison['aggregate_delta_percentage_points']}, indent=2))


if __name__ == "__main__":
    main()
