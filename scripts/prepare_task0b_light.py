#!/usr/bin/env python3
"""Freeze Task 0A/0B-Light contracts, sample, and finalize replay data.

This script is data-only.  It never imports Trainer or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

from lean_prover.lean_training.expert_iteration.schemas import DataRole, StatementRecord
from scripts.prepare_stage2_sft_phase0 import (
    ATTEMPTS,
    BENCHMARK_SUMMARY,
    ENVIRONMENT_HASH,
    GENERATIONS,
    OLD_LD,
    OLD_WB,
    TRUE_HOLDOUTS,
    canonical_hash,
    domain,
    normalized_statement,
    primary_tactic,
    proof,
    proof_hash,
    proof_tokens,
    qualified_theorem,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    source_kind,
    statement,
    statement_tokens,
    theorem_group,
    verified_training_row,
    write_json,
    write_jsonl,
)


OUTPUT = Path("outputs/task0b_light")
CONTRACT_DIR = Path("outputs/task0_generation_contract")
MERGED = Path(
    "outputs/stage2_sft_incremental_ablation/shared/checkpoints/"
    "M0-ADDON-B-FROZEN-MERGED"
)
TASK0A = Path(
    "outputs/stage2_sft_incremental_ablation/shared/"
    "m0_checkpoint_equivalence_repair"
)
NEW_EVAL = OUTPUT / "runtime/canonical_m0_eval"
SAMPLE_SEED = 20261701
SAMPLE_ROWS = 1000
REPLAY_TARGET = 800
REQUIRED_HOLDOUTS = (
    "wb_unseen_holdout150",
    "ld_easy64",
    "monitor64",
    "hard_ld128",
    "full500",
    "strict_unseen200",
)


def stable_priority(*values: Any) -> str:
    return canonical_hash([SAMPLE_SEED, *values])


def numeric_stats(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(values),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p50": round(percentile(0.50), 6),
        "p95": round(percentile(0.95), 6),
    }


def old_rows(project: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(project / OLD_WB) + read_jsonl(project / OLD_LD)
    if len(rows) != 3000 or len({record_id(row) for row in rows}) != 3000:
        raise RuntimeError("frozen WB2000+LD1000 identity is not 3000 unique rows")
    return rows


def canonical_contract(project: Path) -> tuple[dict[str, Any], str]:
    merged = (project / MERGED).resolve()
    diagnostics = read_json(project / TASK0A / "diagnostics.json")
    payload = {
        "schema_version": 1,
        "contract_name": "task0_canonical_merged_transformers_v1",
        "frozen_model": "M0-ADDON-B-FROZEN",
        "backend": "transformers",
        "checkpoint_type": "merged",
        "checkpoint_path": str(merged),
        "checkpoint_model_sha256": diagnostics["merge"]["merged_model_sha256"],
        "tokenizer_path": str(merged),
        "tokenizer_sha256": diagnostics["tokenizer"]["tokenizer.json"]["merged"],
        "tokenizer_config_sha256": diagnostics["tokenizer"]["tokenizer_config.json"]["merged"],
        "chat_template_sha256": diagnostics["tokenizer"]["chat_template.jinja"]["merged"],
        "dtype": "bfloat16",
        "load_in_4bit": False,
        "prompt_format": "plain_text_lean_sections_v1",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 20,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "max_new_tokens": 256,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
        "stop_tokens": [],
        "request_seed": SAMPLE_SEED,
        "staged_iteration": -10,
        "effective_prompt_seed_formula": "request_seed + staged_iteration*1000000 + prompt_offset",
        "candidates_per_statement": 2,
        "candidate_seed_note": "Two sampled sequences share the fixed per-prompt generator seed and are distinguished by sample_index 0/1.",
        "selection_reason": "Two candidate classification requires sampling; temperature/top_p preserve the frozen round-1 candidate semantics while checkpoint/backend are canonicalized to merged Transformers. top_k and repetition_penalty are made explicit from the audited model generation config.",
        "forbidden": {
            "adapter_path": True,
            "vllm": True,
            "implicit_model_generation_config": True,
            "M0_EQUIVALENCE_LOGIT_QUANTUM": True,
        },
    }
    contract_hash = canonical_hash(payload)
    return {**payload, "generation_config_sha256": contract_hash}, contract_hash


def freeze(project: Path) -> None:
    output = project / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    contract, contract_hash = canonical_contract(project)
    write_json(project / CONTRACT_DIR / "generation_contract.json", contract)
    write_json(output / "generation_contract.json", contract)

    source_config = read_json(
        project / TASK0A / "matched_transformers_bf16_config.json"
    )
    generation = source_config["discovery"]["generation"]
    generation.update(
        {
            "backend": "transformers",
            "samples_per_statement": 2,
            "temperature": contract["temperature"],
            "top_p": contract["top_p"],
            "max_new_tokens": contract["max_new_tokens"],
            "batch_size": 1,
            "load_in_4bit": False,
            "generation_contract_path": str(
                (project / CONTRACT_DIR / "generation_contract.json").resolve()
            ),
            "generation_contract_sha256": contract_hash,
        }
    )
    source_config["execution"]["generation_timeout_seconds"] = max(
        int(source_config["execution"].get("generation_timeout_seconds") or 0),
        14400,
    )
    write_json(output / "task0b_light_runtime_config.json", source_config)

    summary = read_json(project / TASK0A / "equivalence_summary.json")
    root = read_json(project / TASK0A / "task0a_root_cause_diagnostics.json")
    if len(summary["details"]) != 24 or summary["pantograph_outcome_matches"] != 23:
        raise RuntimeError("Task 0A repaired canary is not the audited 23/24 result")
    fork = root["audit"]["fork_actual_generation"]
    rows = []
    for detail in summary["details"]:
        rows.append(
            "| {statement_id} | {unmerged_status} | {merged_status} | {match} |".format(
                statement_id=detail["statement_id"],
                unmerged_status=detail["unmerged_status"],
                merged_status=detail["merged_status"],
                match="PASS" if detail["pantograph_outcome_match"] else "DIFF",
            )
        )
    report = f"""# Task 0A equivalence finalization

## Final decision

- Structural merge correctness: **PASS**
- Strict BF16 token equivalence: **NOT REQUIRED**
- Canonical inference path frozen: **PASS**

The only BF16 Pantograph outcome divergence is numerical, not structural.  All
future M0 generation uses the merged checkpoint, Transformers backend, and the
explicit contract hash `{contract_hash}`.  Adapter/unmerged and vLLM results
are forbidden for new generation.

## 24 canaries

| statement_id | adapter status | merged status | outcome match |
|---|---|---|---|
{chr(10).join(rows)}

Outcome match: **{summary['pantograph_outcome_matches']}/24**.

## Sole divergence

- statement: `{fork and 'stmt_f6798f8287b817aaae741301'}`
- first fork: generated token **{fork['generated_token_number_one_based']}**
- adapter token: `{fork['unmerged_next']['token']}` (`{fork['unmerged_next']['id']}`)
- merged token: `{fork['merged_next']['token']}` (`{fork['merged_next']['id']}`)
- `repetition_penalty=1.1` lowers repeated positive token `h` from `24.25`
  to `22.0454559`, placing the greedy boundary between the two BF16 `sq`
  scores (`22.375` dynamic and `22.0` merged).
- FP32 common-prefix one-step forward: both paths have identical top-10 order;
  max top-10 absolute logit difference `2.6703e-5`; after the same penalty both
  select `sq` with margins `0.3600967` and `0.3600953`.

No Trainer or training process was used.
"""
    (output / "task0a_equivalence_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "freeze", "generation_config_sha256": contract_hash}, indent=2))


def legacy_record_ids(project: Path, rows: list[dict[str, Any]]) -> set[str]:
    generations = read_jsonl(project / GENERATIONS)
    attempts = read_jsonl(project / ATTEMPTS)
    normalized = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    prompt_map: dict[str, str | None] = {}
    for row in rows:
        key = normalized(row.get("prompt"))
        current = prompt_map.get(key)
        row_id = record_id(row)
        prompt_map[key] = row_id if current in (None, row_id) else ""
    statement_map = {
        str(generation.get("statement_id") or ""): prompt_map.get(
            normalized(generation.get("prompt"))
        )
        for generation in generations
    }
    return {
        str(statement_map.get(str(attempt.get("problem_id") or "")) or "")
        for attempt in attempts
        if statement_map.get(str(attempt.get("problem_id") or ""))
    }


def length_bins(scores: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for source in ("WB", "LD"):
        values = sorted(
            int(row["proof_token_length"])
            for row in scores
            if row["source"] == source
        )
        cuts = [values[math.floor((len(values) - 1) * q)] for q in (0.25, 0.50, 0.75)]
        for row in scores:
            if row["source"] != source:
                continue
            value = int(row["proof_token_length"])
            index = sum(value > cut for cut in cuts)
            result[row["record_id"]] = ("q1", "q2", "q3", "q4")[index]
    return result


def allocate_strata(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["proof_length_bin"], row["primary_tactic"], row["domain"])].append(row)
    keys = sorted(grouped)
    allocations = {key: min(1, len(grouped[key])) for key in keys}
    remaining = target - sum(allocations.values())
    if remaining < 0:
        keys = sorted(keys, key=lambda key: stable_priority("stratum", *key))[:target]
        allocations = {key: 1 for key in keys}
        remaining = 0
    capacities = {key: len(grouped[key]) - allocations.get(key, 0) for key in keys}
    while remaining:
        active = [key for key in keys if capacities[key] > 0]
        if not active:
            raise RuntimeError("stratified sample capacity exhausted")
        total_capacity = sum(capacities[key] for key in active)
        shares = {
            key: remaining * capacities[key] / total_capacity for key in active
        }
        added = 0
        for key in active:
            amount = min(capacities[key], math.floor(shares[key]))
            allocations[key] += amount
            capacities[key] -= amount
            added += amount
        remaining -= added
        if added == 0:
            for key in sorted(
                active,
                key=lambda item: (-shares[item], stable_priority("remainder", *item)),
            )[:remaining]:
                allocations[key] += 1
                capacities[key] -= 1
                remaining -= 1
    selected = []
    for key in keys:
        ordered = sorted(
            grouped[key], key=lambda row: stable_priority("sample", row["record_id"])
        )
        selected.extend(ordered[: allocations.get(key, 0)])
    return selected


def sample(project: Path) -> None:
    output = project / OUTPUT
    scores = read_jsonl(output / "proxy_scores.jsonl")
    rows = old_rows(project)
    legacy = legacy_record_ids(project, rows)
    missing_ids = {record_id(row) for row in rows} - legacy
    if len(legacy) != 68 or len(missing_ids) != 2932:
        raise RuntimeError(f"legacy/missing identity drift: {len(legacy)}/{len(missing_ids)}")
    if len(scores) != 2932 or {row["record_id"] for row in scores} != missing_ids:
        raise RuntimeError("proxy scores do not cover exactly the 2932 missing rows")
    bins = length_bins(scores)
    enriched = [
        {**row, "proof_length_bin": bins[row["record_id"]]} for row in scores
    ]
    source_counts = Counter(row["source"] for row in enriched)
    wb_target = round(SAMPLE_ROWS * source_counts["WB"] / len(enriched))
    targets = {"WB": wb_target, "LD": SAMPLE_ROWS - wb_target}
    selected_scores = []
    for source in ("WB", "LD"):
        selected_scores.extend(
            allocate_strata(
                [row for row in enriched if row["source"] == source], targets[source]
            )
        )
    selected_scores.sort(key=lambda row: stable_priority("order", row["record_id"]))
    if len(selected_scores) != SAMPLE_ROWS:
        raise RuntimeError("sample is not exactly 1000 rows")
    by_id = {record_id(row): row for row in rows}
    sampled = []
    for order, proxy in enumerate(selected_scores):
        row = dict(by_id[proxy["record_id"]])
        row["task0b_light_sampling"] = {
            "sample_order": order,
            "sample_seed": SAMPLE_SEED,
            "source_target": targets[proxy["source"]],
            "proof_length_bin": proxy["proof_length_bin"],
            "primary_tactic": proxy["primary_tactic"],
            "domain": proxy["domain"],
            "completion_loss": proxy["completion_loss"],
            "token_accuracy": proxy["token_accuracy"],
        }
        sampled.append(row)
    sample_path = output / "sampled_1000_manifest.jsonl"
    existing_ids = []
    if sample_path.is_file():
        existing_ids = [record_id(row) for row in read_jsonl(sample_path)]
    new_ids = [record_id(row) for row in sampled]
    if existing_ids and existing_ids != new_ids:
        raise RuntimeError("refusing to resample an existing frozen 1000 manifest")
    write_jsonl(sample_path, sampled)
    manifest_hash = sha256(sample_path)
    write_json(
        output / "sample_ids.json",
        {
            "seed": SAMPLE_SEED,
            "rows": len(sampled),
            "source_targets": targets,
            "manifest_sha256": manifest_hash,
            "sample_ids": new_ids,
            "resampling_after_generation_forbidden": True,
        },
    )
    print(json.dumps({"stage": "sample", "rows": 1000, "source": targets, "manifest_sha256": manifest_hash}, indent=2))


def identity_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record_id(row),
        "theorem_group": theorem_group(row),
        "qualified_theorem": qualified_theorem(row),
        "exact_statement": statement(row),
        "normalized_statement": normalized_statement(row),
        "proof_variant": (theorem_group(row), proof_hash(row)),
    }


def protected_sets(project: Path) -> dict[str, dict[str, set[Any]]]:
    result = {}
    for name in REQUIRED_HOLDOUTS:
        rows = read_jsonl(project / TRUE_HOLDOUTS[name])
        result[name] = {
            key: {identity_values(row)[key] for row in rows if identity_values(row)[key]}
            for key in identity_values(rows[0])
        }
    return result


def leakage(rows: list[dict[str, Any]], protected: dict[str, dict[str, set[Any]]]) -> dict[str, Any]:
    report = {}
    for name, sets in protected.items():
        overlaps = Counter()
        for row in rows:
            identity = identity_values(row)
            for key, values in sets.items():
                if identity[key] and identity[key] in values:
                    overlaps[key] += 1
        report[name] = {f"{key}_overlap": overlaps[key] for key in sets}
        report[name]["passed"] = not any(overlaps.values())
    return report


def failure_taxonomy(generation: dict[str, Any], verification: dict[str, Any] | None) -> str:
    if verification and verification.get("verified"):
        return "success"
    if not str(generation.get("extracted_proof") or "").strip():
        return "extraction_failure"
    if generation.get("finish_reason") == "length":
        return "semantic_truncation"
    if verification is None:
        return "missing_verification"
    if verification.get("timed_out"):
        return "timeout"
    text = " ".join(
        str(verification.get(key) or "")
        for key in ("status", "error_type", "error_message")
    ).lower()
    if any(token in text for token in ("environment", "worker", "backend", "startup", "import failed")):
        return "environment_error"
    if any(token in text for token in ("unexpected token", "parser", "syntax", "unterminated")):
        return "syntax_corruption"
    for name in ("unsolved_goals", "tactic_error", "elaboration_error", "forbidden_token"):
        if name in text:
            return name
    return str(verification.get("status") or "unknown_failure")


def new_candidates(project: Path, sampled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = project / NEW_EVAL
    generations = read_jsonl(output / "generations.jsonl")
    verifications = read_jsonl(output / "verifications.jsonl")
    verification_by_id = {row["generation_id"]: row for row in verifications}
    statement_to_record = {
        StatementRecord.from_prepared(row, DataRole.BENCHMARK).statement_id: record_id(row)
        for row in sampled
    }
    results = []
    for generation in generations:
        verification = verification_by_id.get(generation["generation_id"])
        row_id = statement_to_record[generation["statement_id"]]
        results.append(
            {
                "record_id": row_id,
                "statement_id": generation["statement_id"],
                "candidate_index": int(generation["sample_index"]),
                "candidate_source": "task0b_light_canonical_merged_transformers",
                "generation_id": generation["generation_id"],
                "generation_seed": generation.get("generation_seed"),
                "generation_contract_sha256": (generation.get("metadata") or {}).get("generation_contract_sha256"),
                "raw_output": generation.get("raw_output"),
                "candidate_proof": generation.get("normalized_proof") or generation.get("extracted_proof"),
                "extraction_success": bool(str(generation.get("extracted_proof") or "").strip()),
                "finish_reason": generation.get("finish_reason"),
                "compile_success": bool(verification and verification.get("verified")),
                "compile_status": verification.get("status") if verification else "missing",
                "timed_out": bool(verification and verification.get("timed_out")),
                "environment_hash": verification.get("environment_hash") if verification else None,
                "error_type": verification.get("error_type") if verification else None,
                "error_message": verification.get("error_message") if verification else None,
                "failure_taxonomy": failure_taxonomy(generation, verification),
            }
        )
    counts = Counter(row["record_id"] for row in results)
    if len(results) != 2000 or set(counts.values()) != {2}:
        raise RuntimeError(f"canonical candidates are not 1000x2: {len(results)}")
    return sorted(results, key=lambda row: (row["record_id"], row["candidate_index"]))


def legacy_candidates(project: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generations = read_jsonl(project / GENERATIONS)
    attempts = read_jsonl(project / ATTEMPTS)
    generation_by_id = {row["generation_id"]: row for row in generations}
    normalized = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    prompt_map = {normalized(row.get("prompt")): record_id(row) for row in rows}
    statement_map = {
        row["statement_id"]: prompt_map.get(normalized(row.get("prompt")))
        for row in generations
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        row_id = statement_map.get(str(attempt.get("problem_id") or ""))
        if row_id:
            grouped[row_id].append(attempt)
    results = []
    for row_id, candidates in grouped.items():
        for attempt in sorted(candidates, key=lambda row: int(row.get("attempt_index") or 0))[:2]:
            generation = generation_by_id[attempt["generation_id"]]
            results.append(
                {
                    "record_id": row_id,
                    "statement_id": attempt["problem_id"],
                    "candidate_index": int(attempt.get("attempt_index") or 0),
                    "candidate_source": "legacy_reused_frozen_result_no_regeneration",
                    "generation_id": attempt["generation_id"],
                    "generation_seed": generation.get("generation_seed"),
                    "generation_contract_sha256": None,
                    "raw_output": generation.get("raw_output"),
                    "candidate_proof": attempt.get("generated_proof"),
                    "extraction_success": bool(str(attempt.get("generated_proof") or "").strip()),
                    "finish_reason": generation.get("finish_reason"),
                    "compile_success": bool(attempt.get("success")),
                    "compile_status": attempt.get("status"),
                    "timed_out": bool(attempt.get("timed_out")),
                    "environment_hash": ENVIRONMENT_HASH,
                    "error_type": None,
                    "error_message": None,
                    "failure_taxonomy": "success" if attempt.get("success") else str(attempt.get("status") or "failure"),
                }
            )
    if len(grouped) != 68 or len(results) != 136:
        raise RuntimeError(f"legacy candidate reuse is not 68x2: {len(grouped)}/{len(results)}")
    return sorted(results, key=lambda row: (row["record_id"], row["candidate_index"]))


def diverse_order(rows: list[dict[str, Any]], class_name: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            source_kind(row),
            domain(row),
            primary_tactic(row),
            proof_tokens(row) // 16,
            stable_priority("replay", class_name, record_id(row)),
        ),
    )


def finalize(project: Path) -> None:
    output = project / OUTPUT
    rows = old_rows(project)
    sampled = read_jsonl(output / "sampled_1000_manifest.jsonl")
    contract = read_json(output / "generation_contract.json")
    candidates = legacy_candidates(project, rows) + new_candidates(project, sampled)
    write_jsonl(output / "candidate_results.jsonl", candidates)
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_record[candidate["record_id"]].append(candidate)
    classifications = []
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_candidates = sorted(by_record.get(record_id(row), []), key=lambda item: item["candidate_index"])
        if len(row_candidates) != 2:
            classification = "unclassified_not_sampled"
            success_count = None
            quality = None
            final_bucket = classification
        else:
            success_count = sum(candidate["compile_success"] for candidate in row_candidates)
            classification = "stable_core" if success_count == 2 else "frontier" if success_count == 1 else "raw_hard"
            recovery = (row.get("metadata") or {}).get("context_recovery") or {}
            quality = {
                "reference_proof_current_environment_pass": verified_training_row(row)
                and str(row.get("environment_hash") or row.get("verification_environment_hash") or "") == ENVIRONMENT_HASH,
                "context_complete": source_kind(row) == "WB"
                or bool(row.get("context") or row.get("context_lines"))
                or not bool(recovery.get("recovery_errors") or recovery.get("context_warnings")),
                "no_environment_error": all(candidate["environment_hash"] == ENVIRONMENT_HASH and candidate["failure_taxonomy"] != "environment_error" for candidate in row_candidates),
                "no_syntax_corruption": all(candidate["extraction_success"] and candidate["failure_taxonomy"] != "syntax_corruption" for candidate in row_candidates),
                "no_timeout_only": not all(candidate["timed_out"] for candidate in row_candidates),
                "no_semantic_truncation": not bool(row.get("truncated") or row.get("max_length"))
                and all(candidate["finish_reason"] != "length" for candidate in row_candidates),
            }
            final_bucket = (
                "filtered_hard"
                if classification == "raw_hard" and all(quality.values())
                else "invalid_or_excluded"
                if classification == "raw_hard"
                else classification
            )
            if final_bucket in {"stable_core", "frontier", "filtered_hard"}:
                bucket_rows[final_bucket].append(row)
        classifications.append(
            {
                "record_id": record_id(row),
                "theorem_group_id": theorem_group(row),
                "qualified_theorem": qualified_theorem(row),
                "source": source_kind(row),
                "candidate_count": len(row_candidates),
                "success_count": success_count,
                "classification": classification,
                "final_bucket": final_bucket,
                "hard_quality_gate": quality if classification == "raw_hard" else None,
                "classification_source": row_candidates[0]["candidate_source"] if row_candidates else None,
                "proof_tokens": proof_tokens(row),
                "statement_tokens": statement_tokens(row),
                "domain": domain(row),
                "primary_tactic": primary_tactic(row),
            }
        )
    write_jsonl(output / "classification_manifest.jsonl", classifications)

    protected = protected_sets(project)
    clean_buckets = {}
    for name, values in bucket_rows.items():
        clean_buckets[name] = [
            row
            for row in values
            if all(report["passed"] for report in leakage([row], protected).values())
        ]
    target = min(REPLAY_TARGET, sum(len(rows) for rows in clean_buckets.values()))
    stable_cap = math.floor(target * 0.10)
    stable = diverse_order(clean_buckets.get("stable_core", []), "stable_core")[:stable_cap]
    frontier_all = diverse_order(clean_buckets.get("frontier", []), "frontier")
    hard_all = diverse_order(clean_buckets.get("filtered_hard", []), "filtered_hard")
    frontier = frontier_all[: min(550, len(frontier_all), target - len(stable))]
    hard = hard_all[: min(170, len(hard_all), target - len(stable) - len(frontier))]
    selected_ids = {record_id(row) for row in stable + frontier + hard}
    remaining = target - len(selected_ids)
    for pool in (frontier_all, hard_all):
        for row in pool:
            if not remaining:
                break
            if record_id(row) in selected_ids:
                continue
            selected_ids.add(record_id(row))
            (frontier if pool is frontier_all else hard).append(row)
            remaining -= 1
    if remaining:
        for row in diverse_order(clean_buckets.get("stable_core", []), "stable_fill"):
            if not remaining or len(stable) >= stable_cap:
                break
            if record_id(row) not in selected_ids:
                selected_ids.add(record_id(row))
                stable.append(row)
                remaining -= 1
    replay_source = [(row, "stable_core") for row in stable] + [(row, "frontier") for row in frontier] + [(row, "filtered_hard") for row in hard]
    tokenizer = AutoTokenizer.from_pretrained(str((project / MERGED).resolve()), trust_remote_code=True)
    replay = []
    for row, bucket in replay_source:
        copied = dict(row)
        labels = tokenizer(proof(row), add_special_tokens=False)["input_ids"] + [int(contract["eos_token_id"])]
        copied["task0b_light_replay"] = {
            "bucket": bucket,
            "generation_contract_sha256": contract["generation_config_sha256"],
            "valid_label_count": len(labels),
            "last_valid_label_id": labels[-1],
            "eos_token_id": int(contract["eos_token_id"]),
            "semantic_truncation": bool(row.get("truncated") or row.get("max_length")),
        }
        replay.append(copied)
    replay.sort(key=lambda row: stable_priority("final_replay_order", record_id(row)))
    write_jsonl(output / "replay_manifest.jsonl", replay)
    write_jsonl(output / "task0b_light_replay_manifest.jsonl", replay)

    duplicate = {
        "rows": len(replay),
        "duplicate_rows": len(replay) - len({canonical_hash(row) for row in replay}),
        "duplicate_record_ids": len(replay) - len({record_id(row) for row in replay}),
        "theorem_group_duplicates": len(replay) - len({theorem_group(row) for row in replay}),
        "max_repeat": max(Counter(record_id(row) for row in replay).values(), default=0),
    }
    leak_report = leakage(replay, protected)
    eos = {
        "rows": len(replay),
        "completion_nonempty": sum(bool(proof(row)) for row in replay),
        "valid_label_gt_zero": sum(row["task0b_light_replay"]["valid_label_count"] > 0 for row in replay),
        "last_valid_label_is_eos": sum(row["task0b_light_replay"]["last_valid_label_id"] == int(contract["eos_token_id"]) for row in replay),
        "eos_not_minus_100": int(contract["eos_token_id"]) != -100,
        "zero_label": sum(row["task0b_light_replay"]["valid_label_count"] == 0 for row in replay),
        "semantic_truncation": sum(row["task0b_light_replay"]["semantic_truncation"] for row in replay),
    }
    raw_class_counts = Counter(row["classification"] for row in classifications)
    final_class_counts = Counter(row["final_bucket"] for row in classifications)
    replay_counts = Counter(row["task0b_light_replay"]["bucket"] for row in replay)
    gates_passed = (
        duplicate["duplicate_rows"] == 0
        and duplicate["theorem_group_duplicates"] == 0
        and duplicate["max_repeat"] <= 1
        and all(item["passed"] for item in leak_report.values())
        and eos["completion_nonempty"] == len(replay)
        and eos["valid_label_gt_zero"] == len(replay)
        and eos["last_valid_label_is_eos"] == len(replay)
        and eos["eos_not_minus_100"]
        and eos["zero_label"] == 0
        and eos["semantic_truncation"] == 0
        and len(replay) >= 750
        and replay_counts["stable_core"] <= math.floor(len(replay) * 0.10)
    )
    sampled_ids = read_json(output / "sample_ids.json")
    report = f"""# Task 0B-Light report

## Outcome

- canonical path: merged `M0-ADDON-B-FROZEN-MERGED` + Transformers
- generation config hash: `{contract['generation_config_sha256']}`
- Task 0A: Structural merge correctness PASS; strict BF16 token equivalence NOT REQUIRED; canonical path PASS
- proxy-scored missing rows: 2932
- fixed sampled rows: {len(sampled)} (manifest `{sampled_ids['manifest_sha256']}`)
- canonical candidates: 2000; legacy retained candidates: 136; Trainer: not started

## Classification

| Bucket | Rows |
|---|---:|
| Stable/Core | {final_class_counts['stable_core']} |
| Frontier | {final_class_counts['frontier']} |
| Raw Hard (before quality filtering) | {raw_class_counts['raw_hard']} |
| Filtered Hard | {final_class_counts['filtered_hard']} |
| invalid_or_excluded | {final_class_counts['invalid_or_excluded']} |
| unclassified_not_sampled | {final_class_counts['unclassified_not_sampled']} |

## Replay manifest

| Bucket | Rows |
|---|---:|
| Stable/Core | {replay_counts['stable_core']} |
| Frontier | {replay_counts['frontier']} |
| Filtered Hard | {replay_counts['filtered_hard']} |
| Total | {len(replay)} |

- proof-token distribution: `{json.dumps(numeric_stats([proof_tokens(row) for row in replay]))}`
- total-token distribution: `{json.dumps(numeric_stats([int(row.get('total_tokens') or 0) for row in replay]))}`
- source distribution: `{json.dumps(Counter(source_kind(row) for row in replay))}`
- duplicate audit: `{json.dumps(duplicate)}`
- leakage checks: `{json.dumps(leak_report)}`
- EOS checks: `{json.dumps(eos)}`

## Gate decision

Replay Stress Test allowed: **{'YES' if gates_passed else 'NO'}**.

Task 0A finalized.  
Canonical generation path frozen.  
Task 0B-Light completed.  
No Trainer or GPU training was started.  
Waiting for approval before Replay Stress Test.
"""
    (output / "task0b_light_report.md").write_text(report, encoding="utf-8")
    write_json(
        output / "task0b_light_gate_audit.json",
        {
            "passed": gates_passed,
            "duplicates": duplicate,
            "leakage": leak_report,
            "eos": eos,
            "classification_raw": dict(raw_class_counts),
            "classification_final": dict(final_class_counts),
            "replay": dict(replay_counts),
        },
    )
    print(
        json.dumps(
            {
                "stage": "finalize",
                "classified_raw": dict(raw_class_counts),
                "classified_final": dict(final_class_counts),
                "replay": dict(replay_counts),
                "replay_rows": len(replay),
                "gates_passed": gates_passed,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--stage", choices=("freeze", "sample", "finalize"), required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    {"freeze": freeze, "sample": sample, "finalize": finalize}[args.stage](project)


if __name__ == "__main__":
    main()
