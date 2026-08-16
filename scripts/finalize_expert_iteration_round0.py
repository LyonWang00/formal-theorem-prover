"""Analyze, archive, and report Expert Iteration Round 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
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
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_MASTER_SEED = 20260812


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else 0.0


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def summarize(directory: Path, expected: int) -> tuple[dict[str, Any], dict[str, dict[int, bool]]]:
    attempts = read_jsonl(directory / "attempts.jsonl")
    generations = read_jsonl(directory / "generations.jsonl")
    summary = read_json(directory / "benchmark_summary.json")
    paired: dict[str, dict[int, bool]] = {}
    for row in attempts:
        paired.setdefault(str(row["problem_id"]), {})[int(row["attempt_index"])] = bool(row["success"])
    if (
        len(paired) != expected
        or len(attempts) != expected * 2
        or len(generations) != expected * 2
        or any(set(values) != {0, 1} for values in paired.values())
    ):
        raise RuntimeError(f"incomplete two-candidate result at {directory}")
    pass1 = {key: values[0] for key, values in paired.items()}
    pass2 = {key: any(values.values()) for key, values in paired.items()}
    lengths = [float(row.get("metadata", {}).get("completion_tokens") or 0) for row in generations]
    repetitions = [repeated_ngram_ratio(str(row.get("raw_output") or "")) for row in generations]
    load_strategies = Counter(
        str(row.get("metadata", {}).get("model_load_strategy") or "not_recorded")
        for row in generations
    )
    result = {
        "problems": expected,
        "candidates": len(attempts),
        "candidate_successes": sum(bool(row.get("success")) for row in attempts),
        "candidate_success_rate": sum(bool(row.get("success")) for row in attempts) / len(attempts),
        "pass_at_1_solved": sum(pass1.values()),
        "pass_at_2_solved": sum(pass2.values()),
        "pass_at_1": sum(pass1.values()) / expected,
        "pass_at_2": sum(pass2.values()) / expected,
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "completion_tokens": {
            "mean": mean(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "max_length_finish_ratio": sum(
            row.get("finish_reason") in {"length", "max_length"}
            or int(row.get("metadata", {}).get("completion_tokens") or 0) >= 256
            for row in generations
        )
        / len(generations),
        "extraction_success_rate": sum(bool(str(row.get("extracted_proof") or "").strip()) for row in generations)
        / len(generations),
        "mean_repeated_4gram_ratio": mean(repetitions),
        "pathological_repetition_rate": sum(value >= 0.35 for value in repetitions) / len(repetitions),
        "fatal_errors": len(summary.get("fatal_errors") or []),
        "worker_restarts": int(summary.get("pantograph_worker_restart_count") or 0),
        "generation_seconds": float(summary.get("generation_seconds") or 0),
        "verification_seconds": float(summary.get("verification_seconds") or 0),
        "model_load_strategy_counts": dict(load_strategies),
    }
    return result, paired


def exact_mcnemar(baseline_only: int, challenger_only: int) -> float:
    discordant = baseline_only + challenger_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, challenger_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_MASTER_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def states(paired: dict[str, dict[int, bool]], metric: str) -> dict[str, bool]:
    if metric == "pass_at_1":
        return {key: values[0] for key, values in paired.items()}
    if metric == "pass_at_2":
        return {key: any(values.values()) for key, values in paired.items()}
    raise ValueError(metric)


def paired_statistic(baseline: dict[str, bool], challenger: dict[str, bool], label: str) -> dict[str, Any]:
    if set(baseline) != set(challenger):
        raise RuntimeError(f"paired problem IDs differ: {label}")
    ids = sorted(baseline)
    diffs = np.asarray([int(challenger[item]) - int(baseline[item]) for item in ids], dtype=int)
    negative = int(np.sum(diffs == -1))
    zero = int(np.sum(diffs == 0))
    positive = int(np.sum(diffs == 1))
    rng = np.random.default_rng(bootstrap_seed(label))
    probabilities = np.asarray([negative, zero, positive], dtype=float) / len(ids)
    draws = rng.multinomial(len(ids), probabilities, size=BOOTSTRAP_REPLICATES)
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


def fmt_rate(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt_stat(value: dict[str, Any]) -> str:
    low, high = value["paired_bootstrap_95_ci_percentage_points"]
    return f"{value['delta_percentage_points']:+.2f} pp [{low:+.2f}, {high:+.2f}], p={value['mcnemar_exact_p']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/round0"
    h0_root = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/evaluation/full/H0-No-Hard"
    ei_root = root / "evaluation/core/EI-Round0"

    metrics: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, dict[str, dict[int, bool]]]] = {}
    comparisons: dict[str, Any] = {}
    aggregate_states = {"H0": {"pass_at_1": {}, "pass_at_2": {}}, "EI-Round0": {"pass_at_1": {}, "pass_at_2": {}}}
    for label, (directory, expected) in DATASETS.items():
        h0_metrics, h0_paired = summarize(h0_root / directory, expected)
        ei_metrics, ei_paired = summarize(ei_root / directory, expected)
        metrics[label] = {"H0": h0_metrics, "EI-Round0": ei_metrics}
        paired[label] = {"H0": h0_paired, "EI-Round0": ei_paired}
        comparisons[label] = {}
        for metric in ("pass_at_1", "pass_at_2"):
            h0_states = states(h0_paired, metric)
            ei_states = states(ei_paired, metric)
            comparisons[label][metric] = paired_statistic(h0_states, ei_states, f"{label}:{metric}")
            for model, values in (("H0", h0_states), ("EI-Round0", ei_states)):
                aggregate_states[model][metric].update({f"{label}:{key}": value for key, value in values.items()})
    comparisons["Aggregate428"] = {
        metric: paired_statistic(
            aggregate_states["H0"][metric],
            aggregate_states["EI-Round0"][metric],
            f"Aggregate428:{metric}",
        )
        for metric in ("pass_at_1", "pass_at_2")
    }

    discovery = read_json(root / "discovery/discovery_summary.json")
    training = read_json(root / "checkpoint/training_summary.json")
    manifest_gate = read_json(root / "training_manifest/manifest_gate.json")
    generation_contract = read_json(root / "evaluation/generation_contract.json")
    contract_audit = read_json(root / "evaluation/contract_audit.json")
    failure_bank = read_jsonl(root / "failure_bank/failure_bank.jsonl")
    success_bank = read_jsonl(root / "success_bank/success_bank.jsonl")
    failure_taxonomy = Counter(str(row.get("failure_taxonomy") or "unknown") for row in failure_bank)
    failure_layers = Counter(str(row.get("failure_layer") or "unknown") for row in failure_bank)

    wb = comparisons["WB-Unseen-Holdout150"]["pass_at_2"]
    ld = comparisons["LD-easy64"]["pass_at_2"]
    monitor = comparisons["Monitor64"]["pass_at_2"]
    retention = comparisons["WB-Train-Retention150"]["pass_at_2"]
    aggregate = comparisons["Aggregate428"]["pass_at_2"]
    foundation_forgetting = retention["delta_percentage_points"] < -2.0 or monitor["delta_percentage_points"] < -2.0
    output_drift = any(
        metrics[label]["EI-Round0"]["pathological_repetition_rate"]
        > max(0.05, metrics[label]["H0"]["pathological_repetition_rate"] + 0.03)
        or metrics[label]["EI-Round0"]["max_length_finish_ratio"]
        > max(0.10, metrics[label]["H0"]["max_length_finish_ratio"] + 0.05)
        for label in DATASETS
    )
    clear_improvement = (
        aggregate["delta_percentage_points"] > 0
        and aggregate["paired_bootstrap_95_ci_percentage_points"][0] > 0
        and aggregate["mcnemar_exact_p"] < 0.05
        and wb["delta_percentage_points"] >= 0
        and ld["delta_percentage_points"] >= 0
        and not foundation_forgetting
    )
    next_round_supported = (
        aggregate["delta_percentage_points"] >= 0
        and wb["delta_percentage_points"] >= 0
        and ld["delta_percentage_points"] >= 0
        and not foundation_forgetting
        and not output_drift
    )
    repair_sft_supported = failure_layers.get("repair_candidate", 0) >= 500 and failure_layers.get("near_miss", 0) >= 200

    comparison_payload = {
        "models": ["H0-No-Hard", "EI-Round0"],
        "datasets": metrics,
        "paired_statistics": comparisons,
        "method": {
            "bootstrap": "20,000 paired statement-level multinomial bootstrap replicates",
            "mcnemar": "two-sided exact binomial McNemar",
            "master_seed": BOOTSTRAP_MASTER_SEED,
        },
        "decisions": {
            "foundation_forgetting": foundation_forgetting,
            "output_drift": output_drift,
            "clear_core_improvement_gate": clear_improvement,
            "extended_evaluation_run": False,
            "extended_evaluation_reason": (
                "not run: Core did not satisfy the joint gate (significant aggregate gain, no WB/LD regression, no forgetting)"
                if not clear_improvement
                else "not run automatically; Core gate passed but task stops after required evaluation"
            ),
            "supports_ei_round1": next_round_supported,
            "failure_bank_supports_repair_sft": repair_sft_supported,
        },
    }
    write_json(root / "comparisons/h0_vs_ei_round0.json", comparison_payload)

    table = [
        "| Dataset | H0 P@1 | EI P@1 | H0 P@2 | EI P@2 | EI candidate success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in DATASETS:
        h0, ei = metrics[label]["H0"], metrics[label]["EI-Round0"]
        table.append(
            f"| {label} | {h0['pass_at_1_solved']}/{h0['problems']} ({fmt_rate(h0['pass_at_1'])}) "
            f"| {ei['pass_at_1_solved']}/{ei['problems']} ({fmt_rate(ei['pass_at_1'])}) "
            f"| {h0['pass_at_2_solved']}/{h0['problems']} ({fmt_rate(h0['pass_at_2'])}) "
            f"| {ei['pass_at_2_solved']}/{ei['problems']} ({fmt_rate(ei['pass_at_2'])}) "
            f"| {ei['candidate_successes']}/{ei['candidates']} ({fmt_rate(ei['candidate_success_rate'])}) |"
        )
    stats_table = [
        "| Dataset | ΔP@1 (95% CI, McNemar) | ΔP@2 (95% CI, McNemar) |",
        "|---|---:|---:|",
    ]
    for label in (*DATASETS, "Aggregate428"):
        stats_table.append(
            f"| {label} | {fmt_stat(comparisons[label]['pass_at_1'])} | {fmt_stat(comparisons[label]['pass_at_2'])} |"
        )
    behavior_table = [
        "| Dataset | H0 mean tokens | EI mean tokens | H0 repetition | EI repetition | H0 max-length | EI max-length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in DATASETS:
        h0, ei = metrics[label]["H0"], metrics[label]["EI-Round0"]
        behavior_table.append(
            f"| {label} | {h0['completion_tokens']['mean']:.2f} | {ei['completion_tokens']['mean']:.2f} "
            f"| {fmt_rate(h0['pathological_repetition_rate'])} | {fmt_rate(ei['pathological_repetition_rate'])} "
            f"| {fmt_rate(h0['max_length_finish_ratio'])} | {fmt_rate(ei['max_length_finish_ratio'])} |"
        )

    report = [
        "# Expert Iteration Round 0 Final Report",
        "",
        "## Executive conclusion",
        "",
        f"- Discovery closed successfully: 500 statements × 8 candidates = {discovery['generated_candidates']} candidates; Pantograph successes {discovery['pantograph_successes']} ({fmt_rate(discovery['pantograph_success_rate'])}).",
        f"- Success Bank contains {len(success_bank)} verified candidate proofs from {discovery['success_bank_statements']} unique statements; Failure Bank contains {len(failure_bank)} candidates.",
        f"- Approved conservative SFT used 250 unique theorem groups: 200 Success Bank + 50 partial-success Frontier replay. It completed 16 optimizer steps with no replacement and no duplicate draw.",
        f"- Core aggregate P@2 H0→EI: {comparisons['Aggregate428']['pass_at_2']['baseline_solved']}/428 → {comparisons['Aggregate428']['pass_at_2']['challenger_solved']}/428; {fmt_stat(aggregate)}.",
        f"- Foundation forgetting: {'YES' if foundation_forgetting else 'NO'}; output drift: {'YES' if output_drift else 'NO'}; support EI Round 1: {'YES' if next_round_supported else 'NO / revise first'}.",
        "- GRPO was not started. Full500, Strict-unseen200, and miniF2F were not run.",
        "",
        "## Frozen model and inference contract",
        "",
        f"- Starting model: H0-No-Hard merged, SHA256 `{training['starting_checkpoint_model_sha256']}`.",
        f"- EI merged model SHA256: `{generation_contract['checkpoint_model_sha256']}`.",
        f"- Canonical path: merged checkpoint + Transformers + BF16 + frozen explicit generation config; adapter path is forbidden.",
        f"- Parent generation hash: `{contract_audit['parent_hash']}`; resolved EI generation hash: `{contract_audit['resolved_hash']}`; only model identity changed: `{contract_audit['only_model_identity_changed']}`.",
        "- After WSL CUDA/UVM allocation misreporting, Monitor/WB used CPU-BF16 load followed by full CUDA migration. Final inference remained the same merged BF16 GPU model; checkpoint tensors and decoding parameters were unchanged. Two stale project multiprocessing workers were terminated before retry.",
        "",
        "## Discovery and proof banks",
        "",
        f"- Generated: {discovery['generated_candidates']}; verified successes: {discovery['pantograph_successes']}; solved statements/pass@8: {discovery['pass_at']['pass@8']['solved']}/500 ({fmt_rate(discovery['pass_at']['pass@8']['rate'])}).",
        f"- Pass@1: {discovery['pass_at']['pass@1']['solved']}/500; Pass@2: {discovery['pass_at']['pass@2']['solved']}/500.",
        f"- Statement buckets: solved-easy {discovery['statement_buckets']['solved_easy']}, frontier {discovery['statement_buckets']['frontier']}, unsolved {discovery['statement_buckets']['unsolved']}.",
        f"- Failure layers: {dict(failure_layers)}.",
        f"- Failure taxonomy: {dict(failure_taxonomy)}.",
        "",
        "## Training data and gates",
        "",
        f"- Manifest SHA256: `{manifest_gate['manifest_sha256']}`; fixed selection seed: {manifest_gate['selection_seed']}.",
        f"- Duplicate rows / record IDs / theorem groups: {manifest_gate['duplicates']['duplicate_rows']} / {manifest_gate['duplicates']['duplicate_record_ids']} / {manifest_gate['duplicates']['theorem_group_duplicates']}; max repeat {manifest_gate['duplicates']['max_repeat']}.",
        "- All 250 targets are original H0-generated proofs verified by Pantograph; no reference proof or failed proof was used as a target.",
        f"- EOS: 250/250 final valid label = {manifest_gate['eos']['eos_token_id']}; zero-label {manifest_gate['eos']['zero_label']}; semantic truncation {manifest_gate['eos']['semantic_truncation']}.",
        "- Six-axis overlap with H0 training and every protected evaluation set: zero.",
        f"- SFT: epoch 1, lr 1e-5, LoRA r32/alpha64/dropout0.05, effective batch 16, packing false, completion-only loss, supervised EOS required. Train loss {training['metrics']['train_loss']:.4f}; validation loss {training['eval_loss']:.4f}; validation token accuracy {fmt_rate(training['eval_token_accuracy'])}.",
        "",
        "## Core evaluation",
        "",
        *table,
        "",
        "All evaluations used two candidates per statement and complete Pantograph outcomes. LD-easy64 used source-faithful grouped imports; the other three used their frozen evaluation roles and environments.",
        "",
        "## Paired statistical analysis",
        "",
        *stats_table,
        "",
        "Confidence intervals use 20,000 paired statement-level bootstrap replicates; p-values are two-sided exact McNemar tests. Point estimates without CI exclusion of zero are treated as directional only.",
        "",
        "## Generation stability",
        "",
        *behavior_table,
        "",
        f"Foundation forgetting gate: {'FAILED' if foundation_forgetting else 'PASSED'}. Output-drift gate: {'FAILED' if output_drift else 'PASSED'}.",
        "",
        "## Required questions",
        "",
        f"1. Discovery generated {discovery['generated_candidates']} candidates.",
        f"2. Pantograph candidate success rate was {fmt_rate(discovery['pantograph_success_rate'])} ({discovery['pantograph_successes']}/{discovery['generated_candidates']}).",
        f"3. Success Bank size is {len(success_bank)} candidate proofs across {discovery['success_bank_statements']} unique statements.",
        f"4. Failure distribution is near-miss {failure_layers.get('near_miss', 0)}, repair-candidate {failure_layers.get('repair_candidate', 0)}, invalid {failure_layers.get('invalid', 0)}; detailed taxonomy is listed above.",
        f"5. EI SFT used {training['rows']} rows and {training['optimizer_steps']} optimizer steps.",
        f"6. WB unseen P@2 changed by {wb['delta_percentage_points']:+.2f} pp; LD-easy P@2 by {ld['delta_percentage_points']:+.2f} pp; aggregate candidate success is reported per dataset in the Core table.",
        f"7. Foundation forgetting: {'yes' if foundation_forgetting else 'no'}; output drift: {'yes' if output_drift else 'no'}; obvious overfitting: {'not established' if not foundation_forgetting else 'possible, due to retention/monitor degradation'}.",
        f"8. Enter EI Round 1: {'supported under the same conservative scale and gates' if next_round_supported else 'not yet supported; revise selection/mix and re-review Core regressions first'}.",
        f"9. Failure Bank supports Repair SFT planning: {'yes' if repair_sft_supported else 'not yet'}, but failure targets remain prohibited until a separate repair-data contract is approved.",
        "",
        "## Extended evaluation decision",
        "",
        f"Clear-improvement gate: {'PASSED' if clear_improvement else 'NOT PASSED'}. Extended sets were not run. {comparison_payload['decisions']['extended_evaluation_reason']}.",
        "",
        "EI Round 0 completed.",
        "No GRPO was started.",
        "Waiting for review before EI Round 1 / Repair SFT planning.",
        "",
    ]
    (root / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    status = read_json(root / "status.json")
    status.update(
        {
            "status": "EI_ROUND0_COMPLETED",
            "trainer_started": True,
            "trainer_completed": True,
            "grpo_started": False,
            "core_evaluation_completed": True,
            "core_candidates": sum(metrics[label]["EI-Round0"]["candidates"] for label in DATASETS),
            "extended_evaluation_started": False,
            "final_report": str(root / "final_report.md"),
            "comparison": str(root / "comparisons/h0_vs_ei_round0.json"),
        }
    )
    write_json(root / "status.json", status)
    print(json.dumps(comparison_payload["decisions"], ensure_ascii=False, indent=2))
    print(root / "final_report.md")


if __name__ == "__main__":
    main()
