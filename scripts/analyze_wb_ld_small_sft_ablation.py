#!/usr/bin/env python3
"""Aggregate paired gate metrics, behavior diagnostics, and promotion decisions."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODEL_DIRS = {
    "M0-ZERO": "zero_step",
    "MIX-A-WB100": "MIX-A-WB100",
    "MIX-B-LD25": "MIX-B-LD25",
    "MIX-C-LD50": "MIX-C-LD50",
    "MIX-E-LD100": "MIX-E-LD100",
}
DATASETS = {
    "wb_gate150": "wb_gate150",
    "ld_holdout": "ld_holdout",
    "monitor64": "monitor64",
}
FULL_DATASETS = ("full500", "strict_unseen200")
COMPARISONS = (
    ("MIX-A-WB100", "M0-ZERO"),
    ("MIX-B-LD25", "MIX-A-WB100"),
    ("MIX-C-LD50", "MIX-A-WB100"),
    ("MIX-E-LD100", "MIX-A-WB100"),
    ("MIX-B-LD25", "M0-ZERO"),
    ("MIX-C-LD50", "M0-ZERO"),
    ("MIX-E-LD100", "M0-ZERO"),
)
TACTICS = (
    "simp",
    "norm_num",
    "linarith",
    "nlinarith",
    "ring",
    "ring_nf",
    "aesop",
    "exact",
    "apply",
    "rw",
    "simpa",
    "constructor",
    "cases",
    "induction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def evaluation_path(root: Path, model: str, dataset: str) -> Path:
    model_dir = MODEL_DIRS[model]
    if model == "M0-ZERO":
        return root / "evaluation" / model_dir / DATASETS[dataset]
    return root / "evaluation" / DATASETS[dataset] / model_dir


def full_evaluation_path(root: Path, model: str, dataset: str) -> Path:
    return root / "evaluation" / dataset / model


def proof_style(proof: str) -> str:
    value = proof.strip()
    if re.match(r"^by\b", value):
        return "tactic"
    if re.search(r"\bby\b", value):
        return "mixed"
    return "term"


def repetitive(proof: str) -> bool:
    tokens = re.findall(r"\S+", proof)
    if len(tokens) < 12:
        return False
    for width in (2, 3, 4):
        grams = Counter(tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1))
        if max(grams.values(), default=0) >= 4:
            return True
    return False


def behavior(generations: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    attempt_by_id = {str(row["generation_id"]): row for row in attempts}
    lengths = sorted(
        int((row.get("metadata") or {}).get("completion_tokens") or 0)
        for row in generations
    )
    extracted = [str(row.get("extracted_proof") or "").strip() for row in generations]
    by_statement: dict[str, list[str]] = defaultdict(list)
    for row, proof in zip(generations, extracted):
        by_statement[str(row["statement_id"])].append(proof)
    total = max(1, len(generations))
    unique_count = sum(len(set(values)) for values in by_statement.values())
    duplicate_count = sum(len(values) - len(set(values)) for values in by_statement.values())
    proof_frequencies = Counter(value for value in extracted if value)
    multiple_proof_statement_count = sum(
        len(set(value for value in values if value)) > 1
        for values in by_statement.values()
    )
    tactics = Counter()
    for proof in extracted:
        matched = False
        for tactic in TACTICS:
            count = len(re.findall(rf"\b{re.escape(tactic)}\b", proof))
            if count:
                tactics[tactic] += count
                matched = True
        if not matched and proof:
            tactics["other"] += 1
    status = Counter(
        str(attempt_by_id.get(str(row["generation_id"]), {}).get("status") or "missing")
        for row in generations
    )

    def percentile(ratio: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, round((len(lengths) - 1) * ratio))]

    return {
        "candidates": len(generations),
        "output_tokens": {
            "mean": sum(lengths) / max(1, len(lengths)),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(lengths, default=0),
        },
        "proof_extraction_success_rate": sum(bool(value) for value in extracted) / total,
        "format_validity_rate": sum(
            bool(value) and ":=" not in value.splitlines()[0] for value in extracted
        )
        / total,
        "proof_style": dict(Counter(proof_style(value) for value in extracted if value)),
        "duplicate_candidate_ratio": duplicate_count / total,
        "multiple_proof_ratio": multiple_proof_statement_count
        / max(1, len(by_statement)),
        "unique_proof_ratio": unique_count / total,
        "mean_unique_candidates_per_statement": unique_count / max(1, len(by_statement)),
        "top10_proof_coverage": sum(
            count for _, count in proof_frequencies.most_common(10)
        )
        / total,
        "top50_proof_coverage": sum(
            count for _, count in proof_frequencies.most_common(50)
        )
        / total,
        "repetition_ratio": sum(repetitive(value) for value in extracted) / total,
        "finish_reason": dict(Counter(str(row.get("finish_reason") or "unknown") for row in generations)),
        "length_finish_ratio": sum(
            str(row.get("finish_reason") or "").lower() == "length"
            for row in generations
        )
        / total,
        "generation_timeout_count": sum(
            str(row.get("finish_reason")) in {"abort", "error"} for row in generations
        ),
        "pantograph_failure_taxonomy": dict(status),
        "tactic_usage": dict(tactics),
    }


def solve_vectors(attempts: list[dict[str, Any]], k: int = 4) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for row in attempts:
        if int(row.get("attempt_index") or 0) < k and row.get("success"):
            values[str(row["problem_id"])] = 1
        else:
            values.setdefault(str(row["problem_id"]), 0)
    return dict(values)


def paired_bootstrap(
    candidate: dict[str, int],
    baseline: dict[str, int],
    *,
    seed: int,
    draws: int = 20000,
) -> dict[str, Any]:
    ids = sorted(set(candidate) & set(baseline))
    deltas = [candidate[key] - baseline[key] for key in ids]
    observed = sum(deltas) / max(1, len(deltas))
    rng = random.Random(seed)
    samples = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(draws)
    )
    return {
        "paired_theorems": len(ids),
        "delta_pass_at_4": observed,
        "bootstrap_draws": draws,
        "bootstrap_95_ci": [
            samples[round((draws - 1) * 0.025)],
            samples[round((draws - 1) * 0.975)],
        ],
    }


def mcnemar(candidate: dict[str, int], baseline: dict[str, int]) -> dict[str, Any]:
    ids = sorted(set(candidate) & set(baseline))
    candidate_only = sum(candidate[key] and not baseline[key] for key in ids)
    baseline_only = sum(baseline[key] and not candidate[key] for key in ids)
    both = sum(candidate[key] and baseline[key] for key in ids)
    neither = len(ids) - candidate_only - baseline_only - both
    discordant = candidate_only + baseline_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(candidate_only, baseline_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "both_solved": both,
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "neither": neither,
        "wins": candidate_only,
        "losses": baseline_only,
        "ties": both + neither,
        "mcnemar_exact_p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/wb_ld_small_sft_ablation")
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    all_metrics: dict[str, dict[str, Any]] = {}
    all_behavior: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    missing: list[str] = []
    for model in MODEL_DIRS:
        all_metrics[model] = {}
        all_behavior[model] = {}
        for dataset in DATASETS:
            path = evaluation_path(root, model, dataset)
            generations = read_jsonl(path / "generations.jsonl")
            attempts = read_jsonl(path / "attempts.jsonl")
            if not generations or not attempts:
                missing.append(f"{model}/{dataset}")
                continue
            summary_path = path / "benchmark_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            all_metrics[model][dataset] = {
                "pass_at_1": summary["pass_at"]["pass@1"],
                "pass_at_2": summary["pass_at"]["pass@2"],
                "pass_at_4": summary["pass_at"]["pass@4"],
                "solved": summary["successes"],
                "candidate_success_rate": sum(bool(row["success"]) for row in attempts)
                / len(attempts),
            }
            all_behavior[model][dataset] = behavior(generations, attempts)
            vectors[dataset][model] = solve_vectors(attempts)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"incomplete evaluations: {missing}")

    paired: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    mcnemar_rows: dict[str, Any] = {}
    for dataset in DATASETS:
        for candidate, baseline in COMPARISONS:
            if candidate not in vectors[dataset] or baseline not in vectors[dataset]:
                continue
            key = f"{dataset}:{candidate}_vs_{baseline}"
            boot = paired_bootstrap(
                vectors[dataset][candidate],
                vectors[dataset][baseline],
                seed=20260801 + len(paired),
            )
            mc = mcnemar(vectors[dataset][candidate], vectors[dataset][baseline])
            paired[key] = {**boot, **mc}
            bootstrap[key] = boot
            mcnemar_rows[key] = mc
    write_json(root / "comparisons/paired_metrics.json", paired)
    write_json(root / "comparisons/bootstrap_results.json", bootstrap)
    write_json(root / "comparisons/mcnemar_results.json", mcnemar_rows)
    write_json(root / "comparisons/behavior_analysis.json", all_behavior)
    write_json(root / "comparisons/small_gate_metrics.json", all_metrics)

    full_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    full_behavior: dict[str, dict[str, Any]] = defaultdict(dict)
    full_paired: dict[str, Any] = {}
    for dataset in FULL_DATASETS:
        full_vectors: dict[str, dict[str, int]] = {}
        for model in ("M0-ZERO", "MIX-A-WB100"):
            path = full_evaluation_path(root, model, dataset)
            generations = read_jsonl(path / "generations.jsonl")
            attempts = read_jsonl(path / "attempts.jsonl")
            if not generations or not attempts:
                missing.append(f"{model}/{dataset}")
                continue
            summary = json.loads(
                (path / "benchmark_summary.json").read_text(encoding="utf-8")
            )
            full_metrics[model][dataset] = {
                "pass_at_1": summary["pass_at"]["pass@1"],
                "pass_at_2": summary["pass_at"]["pass@2"],
                "pass_at_4": summary["pass_at"]["pass@4"],
                "solved": summary["successes"],
                "candidate_success_rate": sum(
                    bool(row["success"]) for row in attempts
                )
                / len(attempts),
            }
            full_behavior[model][dataset] = behavior(generations, attempts)
            full_vectors[model] = solve_vectors(attempts)
        if set(full_vectors) == {"M0-ZERO", "MIX-A-WB100"}:
            key = f"{dataset}:MIX-A-WB100_vs_M0-ZERO"
            full_paired[key] = {
                **paired_bootstrap(
                    full_vectors["MIX-A-WB100"],
                    full_vectors["M0-ZERO"],
                    seed=20260801 + len(full_paired),
                ),
                **mcnemar(
                    full_vectors["MIX-A-WB100"],
                    full_vectors["M0-ZERO"],
                ),
            }
    write_json(
        root / "comparisons/full_evaluation_metrics.json", dict(full_metrics)
    )
    write_json(
        root / "comparisons/full_behavior_analysis.json", dict(full_behavior)
    )
    write_json(root / "comparisons/full_paired_metrics.json", full_paired)

    promotions: list[str] = []
    if all(model in all_metrics and len(all_metrics[model]) == 3 for model in MODEL_DIRS):
        a = all_metrics["MIX-A-WB100"]
        for model in ("MIX-B-LD25", "MIX-C-LD50", "MIX-E-LD100"):
            value = all_metrics[model]
            ld_gain = (
                value["ld_holdout"]["solved"] - a["ld_holdout"]["solved"]
            )
            wb_drop = a["wb_gate150"]["solved"] - value["wb_gate150"]["solved"]
            monitor_drop = a["monitor64"]["solved"] - value["monitor64"]["solved"]
            extraction_drop = (
                all_behavior["MIX-A-WB100"]["wb_gate150"][
                    "proof_extraction_success_rate"
                ]
                - all_behavior[model]["wb_gate150"]["proof_extraction_success_rate"]
            )
            if (
                ld_gain > 0
                and wb_drop <= 2
                and monitor_drop <= 1
                and extraction_drop <= 0.02
            ):
                promotions.append(model)
        promotions.sort(
            key=lambda model: (
                all_metrics[model]["ld_holdout"]["solved"]
                - all_metrics["MIX-A-WB100"]["ld_holdout"]["solved"],
                all_metrics[model]["wb_gate150"]["solved"],
                all_metrics[model]["monitor64"]["solved"],
            ),
            reverse=True,
        )
        promotions = promotions[:2]
    promotion = {
        "rules": {
            "ld_holdout_solved_gain_over_A": ">0",
            "wb_gate_solved_drop_vs_A": "<=2",
            "monitor_solved_drop_vs_A": "<=1",
            "format_extraction_regression": "none_material",
            "maximum_promoted_models": 2,
        },
        "promoted_models": promotions,
        "forced_promotion": False,
        "complete": not missing,
        "missing": missing,
    }
    write_json(root / "comparisons/promotion_decision.json", promotion)
    print(json.dumps({"metrics": all_metrics, "promotion": promotion}, indent=2))


if __name__ == "__main__":
    main()
