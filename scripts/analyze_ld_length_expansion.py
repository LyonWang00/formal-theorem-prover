#!/usr/bin/env python3
"""Analyze the gated full LD/WB length expansion with paired candidate identity."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.analyze_ld_length_ablation import paired_statistics
from scripts.run_ld_length_canary import percentile, repetitive


LENGTHS = (256, 512, 1024)


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


def baseline_dir(project: Path, dataset: str, model: str) -> Path:
    root = project / "outputs/wb_ld_small_sft_ablation/evaluation"
    name = "ld_holdout" if dataset == "ld" else "wb_gate150"
    if model == "M0-ZERO":
        return root / "zero_step" / name
    return root / name / model


def load_unit(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_jsonl(path / "generations.jsonl"), read_jsonl(
        path / "attempts.jsonl"
    )


def candidate_key(row: dict[str, Any], *, generation: bool) -> tuple[str, int]:
    return (
        str(row["statement_id"] if generation else row["problem_id"]),
        int(row["sample_index"] if generation else row["attempt_index"]),
    )


def summarize(
    generations: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt_by_key = {
        candidate_key(row, generation=False): row for row in attempts
    }
    successes: dict[str, set[int]] = defaultdict(set)
    proofs_by_statement: dict[str, list[str]] = defaultdict(list)
    output_tokens: list[int] = []
    finish = Counter()
    statuses = Counter()
    extraction = 0
    valid_format = 0
    repetition = 0
    for generation in generations:
        key = candidate_key(generation, generation=True)
        attempt = attempt_by_key[key]
        if attempt.get("success"):
            successes[key[0]].add(key[1])
        else:
            successes.setdefault(key[0], set())
        proof = str(
            generation.get("extracted_proof")
            or attempt.get("generated_proof")
            or ""
        ).strip()
        proofs_by_statement[key[0]].append(proof)
        output_tokens.append(
            int((generation.get("metadata") or {}).get("completion_tokens") or 0)
        )
        finish[str(generation.get("finish_reason") or "unknown")] += 1
        statuses[str(attempt.get("status") or "missing")] += 1
        extraction += bool(proof)
        valid_format += bool(proof) and ":=" not in proof.splitlines()[0]
        repetition += repetitive(proof)
    candidates = len(generations)
    statements = len(successes)
    duplicate_count = sum(
        len(values) - len(set(values)) for values in proofs_by_statement.values()
    )
    success_candidates = sum(bool(row.get("success")) for row in attempts)
    return {
        "statements": statements,
        "candidates": candidates,
        "solved_count": sum(bool(value) for value in successes.values()),
        "pass_at": {
            f"pass@{k}": sum(
                bool(value & set(range(k))) for value in successes.values()
            )
            / max(1, statements)
            for k in (1, 2, 4)
        },
        "candidate_success_rate": success_candidates / max(1, candidates),
        "output_tokens": {
            "mean": statistics.fmean(output_tokens) if output_tokens else 0.0,
            "p50": percentile(output_tokens, 0.50),
            "p90": percentile(output_tokens, 0.90),
            "p95": percentile(output_tokens, 0.95),
            "max": max(output_tokens, default=0),
        },
        "finish_reason_distribution": dict(finish),
        "length_finish": {
            "count": finish["length"],
            "rate": finish["length"] / max(1, candidates),
        },
        "proof_extraction_success": {
            "count": extraction,
            "rate": extraction / max(1, candidates),
        },
        "format_validity": {
            "count": valid_format,
            "rate": valid_format / max(1, candidates),
        },
        "duplicate_candidate_ratio": duplicate_count / max(1, candidates),
        "repetition_ratio": repetition / max(1, candidates),
        "unique_proof_ratio": sum(
            len(set(values)) for values in proofs_by_statement.values()
        )
        / max(1, candidates),
        "pantograph_compile_success": {
            "count": success_candidates,
            "rate": success_candidates / max(1, candidates),
        },
        "pantograph_failure_taxonomy": dict(statuses),
    }


def solve_vector(attempts: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in attempts:
        problem_id = str(row["problem_id"])
        result.setdefault(problem_id, 0)
        if int(row["attempt_index"]) < 4 and row.get("success"):
            result[problem_id] = 1
    return result


def validate_pairing(
    units: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]
) -> dict[str, Any]:
    generation_maps = {
        length: {
            candidate_key(row, generation=True): row for row in generations
        }
        for length, (generations, _) in units.items()
    }
    canonical = set(generation_maps[256])
    if any(set(rows) != canonical for rows in generation_maps.values()):
        raise ValueError("full expansion candidate keys are not paired")
    if any(
        generation_maps[length][key].get("prompt")
        != generation_maps[256][key].get("prompt")
        for length in generation_maps
        for key in canonical
    ):
        raise ValueError("full expansion prompt drift")
    if any(
        int(generation_maps[length][key].get("generation_seed") or 0)
        != int(generation_maps[256][key].get("generation_seed") or 0)
        for length in generation_maps
        for key in canonical
    ):
        raise ValueError("full expansion candidate seed drift")
    return {
        "candidate_keys": len(canonical),
        "prompt_match": True,
        "candidate_seed_match": True,
        "lengths": sorted(generation_maps),
    }


def compare(
    candidate_attempts: list[dict[str, Any]],
    baseline_attempts: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_solve = solve_vector(candidate_attempts)
    baseline_solve = solve_vector(baseline_attempts)
    paired = paired_statistics(candidate_solve, baseline_solve, seed=seed)
    candidate_by_key = {
        candidate_key(row, generation=False): row for row in candidate_attempts
    }
    baseline_by_key = {
        candidate_key(row, generation=False): row for row in baseline_attempts
    }
    common = sorted(set(candidate_by_key) & set(baseline_by_key))
    theorem_ids = sorted(set(candidate_solve) & set(baseline_solve))
    transitions = {
        "theorem_fail_to_success": sum(
            not baseline_solve[key] and candidate_solve[key]
            for key in theorem_ids
        ),
        "theorem_success_to_fail": sum(
            baseline_solve[key] and not candidate_solve[key]
            for key in theorem_ids
        ),
        "theorem_both_fail": sum(
            not baseline_solve[key] and not candidate_solve[key]
            for key in theorem_ids
        ),
        "theorem_both_success": sum(
            baseline_solve[key] and candidate_solve[key]
            for key in theorem_ids
        ),
        "candidate_fail_to_success": sum(
            not baseline_by_key[key].get("success")
            and candidate_by_key[key].get("success")
            for key in common
        ),
        "candidate_success_to_fail": sum(
            baseline_by_key[key].get("success")
            and not candidate_by_key[key].get("success")
            for key in common
        ),
        "baseline_length_finish_to_success": sum(
            not baseline_by_key[key].get("success")
            and baseline_by_key[key].get("finish_reason") == "length"
            and candidate_by_key[key].get("success")
            for key in common
        ),
    }
    return paired, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    args = parser.parse_args()
    project = args.root.resolve()
    output = (project / args.output).resolve()
    plan = read_json(output / "manifests/expansion_plan.json")
    if not plan.get("authorized"):
        print(json.dumps({"authorized": False}, indent=2))
        return
    metrics: dict[str, Any] = {}
    pairing: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    new_successes: list[dict[str, Any]] = []

    for model_index, model in enumerate(plan["positive_models"]):
        units = {
            256: load_unit(baseline_dir(project, "ld", model)),
            512: load_unit(output / "expansion/ld" / model / "L512"),
            1024: load_unit(output / "expansion/ld" / model / "L1024"),
        }
        pairing[f"{model}:ld"] = validate_pairing(units)
        for length, (generations, attempts) in units.items():
            metrics[f"{model}:ld:L{length}"] = summarize(generations, attempts)
        for index, (candidate_length, baseline_length) in enumerate(
            ((512, 256), (1024, 256), (1024, 512))
        ):
            key = f"{model}:ld:L{candidate_length}_vs_L{baseline_length}"
            paired[key], transitions[key] = compare(
                units[candidate_length][1],
                units[baseline_length][1],
                seed=20261000 + model_index * 10 + index,
            )
        baseline_generations = {
            candidate_key(row, generation=True): row for row in units[256][0]
        }
        baseline_attempts = {
            candidate_key(row, generation=False): row for row in units[256][1]
        }
        for length in (512, 1024):
            candidate_generations = {
                candidate_key(row, generation=True): row
                for row in units[length][0]
            }
            candidate_attempts = {
                candidate_key(row, generation=False): row
                for row in units[length][1]
            }
            for key in sorted(candidate_attempts):
                if baseline_attempts[key].get("success") or not candidate_attempts[
                    key
                ].get("success"):
                    continue
                new_successes.append(
                    {
                        "model": model,
                        "dataset": "ld",
                        "length": length,
                        "statement_id": key[0],
                        "sample_index": key[1],
                        "l256_finish_reason": baseline_generations[key].get(
                            "finish_reason"
                        ),
                        "l256_generation": baseline_generations[key].get(
                            "raw_output"
                        ),
                        "longer_generation": candidate_generations[key].get(
                            "raw_output"
                        ),
                        "pantograph_result": candidate_attempts[key],
                    }
                )

    for pair_index, pair in enumerate(plan["positive_pairs"]):
        model = pair["model"]
        length = int(str(pair["length"]).removeprefix("L"))
        units = {
            256: load_unit(baseline_dir(project, "wb", model)),
            length: load_unit(output / "expansion/wb" / model / f"L{length}"),
        }
        pairing[f"{model}:wb:L{length}"] = validate_pairing(units)
        for unit_length, (generations, attempts) in units.items():
            metrics[f"{model}:wb:L{unit_length}"] = summarize(
                generations, attempts
            )
        key = f"{model}:wb:L{length}_vs_L256"
        paired[key], transitions[key] = compare(
            units[length][1],
            units[256][1],
            seed=20262000 + pair_index,
        )

    write_json(output / "comparisons/full_expansion_metrics.json", metrics)
    write_json(output / "comparisons/full_expansion_pairing.json", pairing)
    write_json(
        output / "comparisons/full_expansion_paired_statistics.json", paired
    )
    write_json(output / "comparisons/full_expansion_transitions.json", transitions)
    with (
        output / "comparisons/full_expansion_new_successes.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as handle:
        for row in new_successes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "authorized": True,
        "positive_models": plan["positive_models"],
        "positive_pairs": plan["positive_pairs"],
        "new_success_records": len(new_successes),
    }
    write_json(output / "comparisons/full_expansion_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
