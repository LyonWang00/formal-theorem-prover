#!/usr/bin/env python3
"""Analyze paired length-ablation canary results and apply expansion gates."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from lean_prover.lean_training.expert_iteration.datasets import load_statements
from lean_prover.lean_training.expert_iteration.schemas import DataRole


MODELS = (
    "M0-ZERO",
    "MIX-A-WB100",
    "MIX-B-LD25",
    "MIX-C-LD50",
    "MIX-E-LD100",
)
LENGTHS = (256, 512, 1024)
DATASETS = ("wb", "ld")
COMPARISONS = ((512, 256), (1024, 256), (1024, 512))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def solve_vector(rows: list[dict[str, Any]], k: int = 4) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        problem_id = str(row["problem_id"])
        result.setdefault(problem_id, 0)
        if int(row["attempt_index"]) < k and row.get("success"):
            result[problem_id] = 1
    return result


def candidate_vector(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["problem_id"]), int(row["attempt_index"])): row for row in rows
    }


def paired_statistics(
    candidate: dict[str, int],
    baseline: dict[str, int],
    *,
    seed: int,
    draws: int = 20000,
) -> dict[str, Any]:
    ids = sorted(set(candidate) & set(baseline))
    differences = [candidate[key] - baseline[key] for key in ids]
    observed = sum(differences) / max(1, len(differences))
    rng = random.Random(seed)
    bootstraps = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(draws)
    )
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    discordant = wins + losses
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(wins, losses) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "paired_theorems": len(ids),
        "delta_pass_at_4": observed,
        "bootstrap_draws": draws,
        "bootstrap_95_ci": [
            bootstraps[round((draws - 1) * 0.025)],
            bootstraps[round((draws - 1) * 0.975)],
        ],
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - discordant,
        "mcnemar_exact_p": p_value,
    }


def validate_pairing(
    generations_by_length: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    keys: dict[int, set[tuple[str, int]]] = {}
    prompts: dict[int, dict[tuple[str, int], str]] = {}
    seeds: dict[int, dict[tuple[str, int], int]] = {}
    for length, rows in generations_by_length.items():
        keys[length] = {
            (str(row["statement_id"]), int(row["sample_index"])) for row in rows
        }
        prompts[length] = {
            (str(row["statement_id"]), int(row["sample_index"])): str(
                row.get("prompt") or ""
            )
            for row in rows
        }
        seeds[length] = {
            (str(row["statement_id"]), int(row["sample_index"])): int(
                row.get("generation_seed") or 0
            )
            for row in rows
        }
        if any(int(row["max_new_tokens"]) != length for row in rows):
            raise ValueError(f"L{length} contains a mismatched max_new_tokens")
    canonical = keys[256]
    if any(value != canonical for value in keys.values()):
        raise ValueError("length runs do not contain identical theorem/candidate keys")
    prompt_match = all(
        prompts[length][key] == prompts[256][key]
        for length in LENGTHS
        for key in canonical
    )
    seed_match = all(
        seeds[length][key] == seeds[256][key]
        for length in LENGTHS
        for key in canonical
    )
    if not prompt_match or not seed_match:
        prompt_mismatches = [
            {
                "key": key,
                "L256": prompts[256][key],
                "L512": prompts[512][key],
                "L1024": prompts[1024][key],
            }
            for key in sorted(canonical)
            if len({prompts[length][key] for length in LENGTHS}) != 1
        ]
        seed_mismatches = [
            {
                "key": key,
                "L256": seeds[256][key],
                "L512": seeds[512][key],
                "L1024": seeds[1024][key],
            }
            for key in sorted(canonical)
            if len({seeds[length][key] for length in LENGTHS}) != 1
        ]
        raise ValueError(
            "paired prompt or candidate seed mismatch: "
            f"prompt_count={len(prompt_mismatches)}, "
            f"seed_count={len(seed_mismatches)}, "
            f"prompt_examples={prompt_mismatches[:2]}, "
            f"seed_examples={seed_mismatches[:5]}"
        )
    return {
        "paired_candidate_keys": len(canonical),
        "prompt_match": prompt_match,
        "candidate_seed_match": seed_match,
        "only_max_new_tokens_changes_within_frozen_configs": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    args = parser.parse_args()
    root = args.root
    metrics: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    new_success_rows: list[dict[str, Any]] = []
    gate_decisions: dict[str, Any] = {}
    combined_statements = load_statements(
        root / "manifests/combined_length_canary_114.jsonl",
        DataRole.BENCHMARK,
    )
    statement_by_id = {
        statement.statement_id: statement for statement in combined_statements
    }
    for model_index, model in enumerate(MODELS):
        generations_by_length: dict[int, list[dict[str, Any]]] = {}
        attempts: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
        for length in LENGTHS:
            unit = root / "generations" / model / f"L{length}"
            if not (unit / "run_summary.json").is_file():
                raise FileNotFoundError(f"incomplete length unit: {unit}")
            generations_by_length[length] = read_jsonl(unit / "generations.jsonl")
            for dataset in DATASETS:
                key = f"{model}:{dataset}:L{length}"
                metrics[key] = read_json(unit / dataset / "metrics.json")
                attempts[dataset][length] = read_jsonl(
                    unit / dataset / "attempts.jsonl"
                )
        metrics[f"{model}:pairing_audit"] = validate_pairing(
            generations_by_length
        )
        for dataset_index, dataset in enumerate(DATASETS):
            vectors = {
                length: solve_vector(attempts[dataset][length])
                for length in LENGTHS
            }
            candidate_vectors = {
                length: candidate_vector(attempts[dataset][length])
                for length in LENGTHS
            }
            theorem_ids = sorted(set.intersection(*(set(value) for value in vectors.values())))
            three_way_patterns: dict[str, int] = defaultdict(int)
            for theorem_id in theorem_ids:
                pattern = "".join(
                    str(vectors[length][theorem_id]) for length in LENGTHS
                )
                three_way_patterns[pattern] += 1
            transitions[f"{model}:{dataset}:all_lengths"] = {
                "paired_theorems": len(theorem_ids),
                "pattern_order": ["L256", "L512", "L1024"],
                "solve_pattern_counts": dict(sorted(three_way_patterns.items())),
                "all_lengths_fail": three_way_patterns["000"],
                "all_lengths_success": three_way_patterns["111"],
            }
            for comparison_index, (candidate_length, baseline_length) in enumerate(
                COMPARISONS
            ):
                comparison_key = (
                    f"{model}:{dataset}:L{candidate_length}_vs_L{baseline_length}"
                )
                paired[comparison_key] = paired_statistics(
                    vectors[candidate_length],
                    vectors[baseline_length],
                    seed=(
                        20260801
                        + model_index * 100
                        + dataset_index * 10
                        + comparison_index
                    ),
                )
                candidate = candidate_vectors[candidate_length]
                baseline = candidate_vectors[baseline_length]
                common = sorted(set(candidate) & set(baseline))
                common_theorems = sorted(
                    set(vectors[candidate_length]) & set(vectors[baseline_length])
                )
                l256_length_fail_to_success = sum(
                    not baseline[key]["success"]
                    and str(baseline[key].get("finish_reason")) == "length"
                    and candidate[key]["success"]
                    for key in common
                )
                transitions[comparison_key] = {
                    "theorem_fail_to_success": sum(
                        not vectors[baseline_length][key]
                        and vectors[candidate_length][key]
                        for key in common_theorems
                    ),
                    "theorem_success_to_fail": sum(
                        vectors[baseline_length][key]
                        and not vectors[candidate_length][key]
                        for key in common_theorems
                    ),
                    "theorem_both_fail": sum(
                        not vectors[baseline_length][key]
                        and not vectors[candidate_length][key]
                        for key in common_theorems
                    ),
                    "theorem_both_success": sum(
                        vectors[baseline_length][key]
                        and vectors[candidate_length][key]
                        for key in common_theorems
                    ),
                    "candidate_fail_to_success": sum(
                        not baseline[key]["success"] and candidate[key]["success"]
                        for key in common
                    ),
                    "candidate_success_to_fail": sum(
                        baseline[key]["success"] and not candidate[key]["success"]
                        for key in common
                    ),
                    "candidate_both_fail": sum(
                        not baseline[key]["success"]
                        and not candidate[key]["success"]
                        for key in common
                    ),
                    "candidate_both_success": sum(
                        baseline[key]["success"] and candidate[key]["success"]
                        for key in common
                    ),
                    "baseline_length_finish_fail_to_success": (
                        l256_length_fail_to_success
                    ),
                }
                for key in common:
                    if baseline[key]["success"] or not candidate[key]["success"]:
                        continue
                    statement = statement_by_id[key[0]]
                    base_generation = next(
                        row
                        for row in generations_by_length[baseline_length]
                        if str(row["statement_id"]) == key[0]
                        and int(row["sample_index"]) == key[1]
                    )
                    long_generation = next(
                        row
                        for row in generations_by_length[candidate_length]
                        if str(row["statement_id"]) == key[0]
                        and int(row["sample_index"]) == key[1]
                    )
                    nested = statement.metadata.get("metadata") or {}
                    length_meta = (
                        nested.get("length_ablation")
                        or statement.metadata.get("length_ablation")
                        or {}
                    )
                    new_success_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "comparison": (
                                f"L{candidate_length}_vs_L{baseline_length}"
                            ),
                            "statement_id": key[0],
                            "sample_index": key[1],
                            "reference_proof_tokens": length_meta.get(
                                "reference_proof_tokens"
                            ),
                            "baseline_generation": base_generation.get(
                                "raw_output"
                            ),
                            "longer_generation": long_generation.get("raw_output"),
                            "baseline_finish_reason": base_generation.get(
                                "finish_reason"
                            ),
                            "longer_finish_reason": long_generation.get(
                                "finish_reason"
                            ),
                            "pantograph_result": candidate[key],
                        }
                    )
        base_ld = metrics[f"{model}:ld:L256"]
        model_gates: dict[str, Any] = {}
        for length in (512, 1024):
            long_ld = metrics[f"{model}:ld:L{length}"]
            comparison_key = f"{model}:ld:L{length}_vs_L256"
            conditions = {
                "at_least_two_more_ld_solved": (
                    int(long_ld["solved_count"]) - int(base_ld["solved_count"])
                    >= 2
                ),
                "candidate_success_gain_at_least_one_point": (
                    float(long_ld["candidate_success_rate"])
                    - float(base_ld["candidate_success_rate"])
                    >= 0.01
                ),
                "two_l256_length_failures_become_success": (
                    transitions[comparison_key][
                        "baseline_length_finish_fail_to_success"
                    ]
                    >= 2
                ),
            }
            model_gates[f"L{length}"] = {
                "conditions": conditions,
                "positive_signal": any(conditions.values()),
            }
        gate_decisions[model] = model_gates
    positive_pairs = [
        {"model": model, "length": length}
        for model, decisions in gate_decisions.items()
        for length, decision in decisions.items()
        if decision["positive_signal"]
    ]
    expansion = {
        "positive_model_length_pairs": positive_pairs,
        "run_full_ld_holdout": bool(positive_pairs),
        "run_full_wb_gate_pairs": positive_pairs,
        "stop_full_expansion_if_empty": not positive_pairs,
    }
    write_json(root / "comparisons/canary_metrics.json", metrics)
    write_json(root / "comparisons/paired_statistics.json", paired)
    write_json(root / "comparisons/transitions.json", transitions)
    write_json(root / "comparisons/expansion_decision.json", expansion)
    write_json(root / "comparisons/gate_decisions.json", gate_decisions)
    with (root / "comparisons/new_verified_successes.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in new_success_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(expansion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
