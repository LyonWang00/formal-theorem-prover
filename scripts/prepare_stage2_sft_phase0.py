#!/usr/bin/env python3
"""Prepare the Phase-0 audit for the staged second-round SFT experiment.

This module is deliberately data-only: it never imports Trainer, torch, PEFT,
or a generation backend.  It freezes identities, reuses existing candidate
results, audits protected-set leakage, and materializes a Phase-1 manifest only
when the frozen classifications are sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENVIRONMENT_HASH = "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
FROZEN_M0 = "M0-ADDON-B-FROZEN"
SOURCE_MODEL = "ADDON-B-WB2000-LD1000"
OLD_WB = Path(
    "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Core-WB2000.jsonl"
)
OLD_LD = Path(
    "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Core-LD1000.jsonl"
)
NEW_WB = Path(
    "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Extra-WB1000.jsonl"
)
FULL_OLD = Path(
    "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/"
    "ADDON-B-WB2000-LD1000.jsonl"
)
EASY_POOL = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/"
    "trainable_easy_pool.jsonl"
)
EXPANSION_LABELS = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/"
    "expansion_labels.jsonl"
)
EXPANSION_POOL = Path(
    "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/"
    "frozen_expansion_pool.jsonl"
)
ATTEMPTS = Path(
    "outputs/wb_ld_budget_support_replay_ablation/evaluation/core/"
    "ADDON-B-WB2000-LD1000/wb_train_retention150/attempts.jsonl"
)
GENERATIONS = ATTEMPTS.with_name("generations.jsonl")
BENCHMARK_SUMMARY = ATTEMPTS.with_name("benchmark_summary.json")
ADDON_IDENTITY = Path(
    "outputs/r_random_replication_strict_eval/audit/addon_b_identity.json"
)
ENVIRONMENT = Path(
    "outputs/r_random_replication_strict_eval/audit/environment.json"
)
ROUND1_CONTRACT = Path(
    "outputs/r_random_replication_strict_eval/audit/frozen_training_contract.json"
)
M0_POINTER = Path(
    "outputs/r_random_replication_strict_eval/checkpoints/"
    "M0-ADDON-B-FROZEN/FROZEN_POINTER.json"
)
M0_MERGE_IDENTITY = Path(
    "outputs/stage2_sft_incremental_ablation/shared/"
    "m0_checkpoint_equivalence/equivalence_summary.json"
)
M0_MERGED = Path(
    "outputs/stage2_sft_incremental_ablation/shared/"
    "checkpoints/M0-ADDON-B-FROZEN-MERGED"
)

RETENTION = {
    "wb_train_retention150": Path(
        "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl"
    )
}
TRUE_HOLDOUTS = {
    "wb_unseen_holdout150": Path(
        "outputs/wb_ld_budget_support_replay_ablation/datasets/"
        "wb_unseen_holdout_150.jsonl"
    ),
    "ld_easy64": Path(
        "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/"
        "ld_easy_holdout_64.jsonl"
    ),
    "monitor64": Path(
        "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl"
    ),
    "hard_ld128": Path(
        "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl"
    ),
    "full500": Path("outputs/b2_expanded_validation/datasets/full500.jsonl"),
    "strict_unseen200": Path(
        "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl"
    ),
    "wb_eval160": Path("data/processed/lean_workbook_verified_v2/eval.jsonl"),
}

EXPECTED_ROWS = {
    OLD_WB: 2000,
    OLD_LD: 1000,
    NEW_WB: 1000,
    FULL_OLD: 3000,
}
GENERATION_MISSING_STOP_THRESHOLD = 1500
PHASE1_TARGET = {"stable": 400, "frontier": 1200, "filtered_hard": 400}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or row.get("sample_id") or "")


def statement(row: dict[str, Any]) -> str:
    return str(
        row.get("lean_statement")
        or row.get("training_statement")
        or row.get("statement")
        or ""
    ).strip()


def normalized_statement(row: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", statement(row)).strip()


def theorem_group(row: dict[str, Any]) -> str:
    return str(
        row.get("theorem_group_id")
        or row.get("statement_id")
        or canonical_hash(normalized_statement(row))
    )


def qualified_theorem(row: dict[str, Any]) -> str:
    annotation = row.get("difficulty_annotation") or {}
    value = (
        row.get("qualified_name")
        or annotation.get("qualified_name")
        or row.get("declaration_name")
    )
    if value:
        return str(value)
    match = re.search(
        r"\b(?:theorem|lemma|example|def)\s+([^\s:{(\[]+)", statement(row)
    )
    return match.group(1) if match else ""


def proof(row: dict[str, Any]) -> str:
    return str(
        row.get("proof")
        or row.get("completion")
        or row.get("training_proof")
        or ""
    ).strip()


def proof_hash(row: dict[str, Any]) -> str:
    value = re.sub(r"\s+", " ", proof(row)).strip()
    return hashlib.sha256(value.encode()).hexdigest() if value else ""


def identity_sets(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    return {
        "record_id": {record_id(row) for row in rows if record_id(row)},
        "theorem_group": {theorem_group(row) for row in rows if theorem_group(row)},
        "qualified_theorem": {
            qualified_theorem(row) for row in rows if qualified_theorem(row)
        },
        "exact_statement": {statement(row) for row in rows if statement(row)},
        "normalized_statement": {
            normalized_statement(row) for row in rows if normalized_statement(row)
        },
        "proof_variant": {
            (theorem_group(row), proof_hash(row))
            for row in rows
            if theorem_group(row) and proof_hash(row)
        },
    }


def overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    first = identity_sets(left)
    second = identity_sets(right)
    intersections = {key: first[key] & second[key] for key in first}
    return {
        f"{key}_overlap": len(value)
        for key, value in intersections.items()
    } | {
        "sample_record_ids": sorted(intersections["record_id"])[:10],
        "sample_theorem_groups": sorted(intersections["theorem_group"])[:10],
        "passed": not any(intersections.values()),
    }


def has_overlap(row: dict[str, Any], protected: list[dict[str, Any]]) -> bool:
    result = overlap([row], protected)
    return not result["passed"]


def source_kind(row: dict[str, Any]) -> str:
    value = str(row.get("sampling_source") or row.get("source") or "").upper()
    return "LD" if "LD" in value or "LEANDOJO" in value else "WB"


def percentile(values: list[int | float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def numeric_stats(values: list[int | float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4) if values else None,
        "p50": round(percentile(values, 0.50), 4) if values else None,
        "p95": round(percentile(values, 0.95), 4) if values else None,
    }


def annotation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    annotation = row.get("difficulty_annotation") or {}
    return annotation.get("metrics") or row.get("metrics") or {}


def proof_tokens(row: dict[str, Any]) -> int:
    metrics = annotation_metrics(row)
    return int(metrics.get("proof_tokens") or row.get("label_tokens") or 0)


def statement_tokens(row: dict[str, Any]) -> int:
    metrics = annotation_metrics(row)
    # input_tokens is a tokenizer-derived prompt proxy for WB, whose source
    # manifests predate separate statement-token accounting.
    return int(metrics.get("statement_tokens") or row.get("input_tokens") or 0)


def primary_tactic(row: dict[str, Any]) -> str:
    value = proof(row)
    if not value:
        return "missing"
    after_by = re.sub(r"^\s*(?:by|term)\b", "", value).strip()
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_!?']*)", after_by)
    return match.group(1) if match else "other"


def domain(row: dict[str, Any]) -> str:
    metrics = annotation_metrics(row)
    if metrics.get("mathlib_domain"):
        return str(metrics["mathlib_domain"])
    source_file = str(row.get("source_file") or "")
    if source_kind(row) == "WB":
        return "Lean-Workbook"
    parts = source_file.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "Mathlib" else "unknown"


def pool_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    premises = [
        int(annotation_metrics(row)["premise_count"])
        for row in rows
        if annotation_metrics(row).get("premise_count") is not None
    ]
    return {
        "rows": len(rows),
        "proof_tokens": numeric_stats([proof_tokens(row) for row in rows]),
        "statement_tokens": numeric_stats([statement_tokens(row) for row in rows]),
        "statement_token_note": (
            "LD uses semantic-review statement_tokens; WB uses tokenizer input_tokens "
            "as the available prompt-level proxy."
        ),
        "source_file_count": len(
            {str(row.get("source_file") or "") for row in rows if row.get("source_file")}
        ),
        "domain_distribution": dict(Counter(domain(row) for row in rows).most_common()),
        "primary_tactic_distribution": dict(
            Counter(primary_tactic(row) for row in rows).most_common(20)
        ),
        "premise_count": numeric_stats(premises),
        "premise_count_distribution": dict(Counter(premises).most_common()),
    }


def verified_training_row(row: dict[str, Any]) -> bool:
    verification = row.get("verification") or {}
    return bool(
        (row.get("pantograph_verified") is True)
        or (row.get("proof_verified") is True)
        or (verification.get("compile_success") is True)
        or (str(row.get("verification_class") or "") == "success")
    )


def eos_static_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "proof_completion_nonempty": 0,
            "empty_completion": 0,
            "zero_label": 0,
            "semantic_truncation": 0,
            "pantograph_or_current_environment_verified": 0,
            "unverified": 0,
            "static_gate_passed": None,
            "status": "NOT_RUN_EMPTY_BLOCKED_MANIFEST",
            "supervised_eos_note": (
                "No executable manifest exists, so an empty set is not reported "
                "as an EOS gate pass."
            ),
        }
    empty_completion = sum(not proof(row) for row in rows)
    zero_label = sum(
        bool(row.get("zero_label"))
        or int(row.get("label_tokens") or proof_tokens(row) or 0) <= 0
        for row in rows
    )
    semantic_truncation = sum(
        bool(row.get("truncated") or row.get("max_length")) for row in rows
    )
    unverified = sum(not verified_training_row(row) for row in rows)
    return {
        "rows": len(rows),
        "proof_completion_nonempty": len(rows) - empty_completion,
        "empty_completion": empty_completion,
        "zero_label": zero_label,
        "semantic_truncation": semantic_truncation,
        "pantograph_or_current_environment_verified": len(rows) - unverified,
        "unverified": unverified,
        "static_gate_passed": not (
            empty_completion or zero_label or semantic_truncation or unverified
        ),
        "supervised_eos_note": (
            "This is a source-row static precheck only. Frozen old rows inherit "
            "the verified first-round EOS materialization; future new rows must "
            "be materialized and checked for last-label EOS before their phase."
        ),
    }


def generation_contract(
    project: Path, addon_identity: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    generations = read_jsonl(project / GENERATIONS)
    summary = read_json(project / BENCHMARK_SUMMARY)
    first = generations[0]
    metadata = first.get("metadata") or {}
    payload = {
        "frozen_model": FROZEN_M0,
        "source_model": SOURCE_MODEL,
        "checkpoint_tree_sha256": addon_identity["checkpoint_tree_sha256"],
        "prompt_format": metadata.get("prompt_format"),
        "generation_backend": summary.get("generation_backend"),
        "temperature": first.get("temperature"),
        "top_p": first.get("top_p"),
        "max_new_tokens": first.get("max_new_tokens"),
        "pass_k": summary.get("pass_k"),
        "early_stop_on_success": summary.get("early_stop_on_success"),
        "verifier_backend": summary.get("verifier_backend"),
        "environment_hash": ENVIRONMENT_HASH,
    }
    return payload, canonical_hash(payload)


def classify(
    old_rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    contract_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    def normalized_prompt(row: dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", str(row.get("prompt") or "")).strip()

    prompt_to_record: dict[str, str | None] = {}
    for row in old_rows:
        key = normalized_prompt(row)
        if not key:
            continue
        current = prompt_to_record.get(key)
        row_id = record_id(row)
        prompt_to_record[key] = row_id if current in (None, row_id) else ""
    statement_to_record: dict[str, str] = {}
    for generation in generations:
        statement_id = str(generation.get("statement_id") or "")
        matched = prompt_to_record.get(normalized_prompt(generation))
        if statement_id and matched:
            statement_to_record[statement_id] = matched

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        matched_record = statement_to_record.get(str(attempt.get("problem_id") or ""))
        if matched_record:
            grouped[matched_record].append(attempt)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest = []
    for row in old_rows:
        group = theorem_group(row)
        candidates = sorted(
            grouped.get(record_id(row), []),
            key=lambda item: int(item.get("attempt_index") or 0),
        )
        count = len(candidates)
        successes = sum(bool(item.get("success")) for item in candidates)
        if count >= 4:
            used_count = 4
            candidates = candidates[:4]
            successes = sum(bool(item.get("success")) for item in candidates)
            classification = (
                "stable" if successes >= 3 else "frontier" if successes >= 1 else "hard"
            )
        elif count == 2:
            used_count = 2
            classification = (
                "stable" if successes == 2 else "frontier" if successes == 1 else "hard"
            )
        else:
            used_count = count
            classification = "missing"
        timed_out = sum(bool(item.get("timed_out")) for item in candidates)
        context = (row.get("metadata") or {}).get("context_recovery") or {}
        context_complete = (
            source_kind(row) == "WB"
            or context.get("context_recovered") is True
            or not context
        )
        hard_quality = {
            "reference_proof_verified": verified_training_row(row),
            "context_complete": context_complete,
            "no_source_local_context_missing": not bool(
                context.get("recovery_errors") or context.get("context_warnings")
            ),
            "no_semantic_truncation": not bool(
                row.get("truncated") or row.get("max_length")
            ),
            "reasonable_proof_length": 0 < proof_tokens(row) <= 256
            and int(row.get("total_tokens") or 0) <= 1024,
            "not_timeout_noise": timed_out == 0,
        }
        filtered_hard = classification == "hard" and all(hard_quality.values())
        output = {
            "record_id": record_id(row),
            "theorem_group_id": group,
            "qualified_theorem": qualified_theorem(row),
            "source": source_kind(row),
            "source_file": row.get("source_file"),
            "number_of_candidates": used_count,
            "success_count": successes if used_count else None,
            "success_rate": (successes / used_count) if used_count else None,
            "classification": classification,
            "filtered_hard": filtered_hard,
            "hard_quality_gate": hard_quality if classification == "hard" else None,
            "classification_source": (
                "reused_frozen_m0_candidate_and_pantograph_results"
                if used_count in (2, 4)
                else "missing_frozen_m0_generation"
            ),
            "generation_contract_hash": (
                contract_hash if used_count in (2, 4) else None
            ),
            "proof_tokens": proof_tokens(row),
            "statement_tokens": statement_tokens(row),
            "total_tokens": int(row.get("total_tokens") or 0),
        }
        manifest.append(output)
        if classification == "hard" and filtered_hard:
            buckets["filtered_hard"].append(row)
        buckets[classification].append(row)
    return manifest, buckets


def duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "record_id": Counter(record_id(row) for row in rows),
        "theorem_group": Counter(theorem_group(row) for row in rows),
        "exact_row": Counter(canonical_hash(row) for row in rows),
    }
    return {
        "rows": len(rows),
        "duplicate_rows": sum(value - 1 for value in counts["exact_row"].values() if value > 1),
        "duplicate_record_ids": sum(
            value - 1 for value in counts["record_id"].values() if value > 1
        ),
        "theorem_group_duplicates": sum(
            value - 1 for value in counts["theorem_group"].values() if value > 1
        ),
        "max_repeat": max(counts["record_id"].values(), default=0),
    }


def protected_index(project: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    index: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for role, mapping in (("retention", RETENTION), ("true_holdout", TRUE_HOLDOUTS)):
        for name, relative in mapping.items():
            path = project / relative
            exists = path.is_file()
            values = read_jsonl(path) if exists else []
            rows[name] = values
            index[name] = {
                "role": role,
                "path": str(path),
                "exists": exists,
                "rows": len(values),
                "sha256": sha256(path) if exists else None,
                "proof_free": all(not proof(row) for row in values) if values else None,
            }
    index["ld_medium_holdout"] = {
        "role": "future_true_holdout",
        "path": None,
        "exists": False,
        "rows": 0,
        "sha256": None,
        "status": "not_yet_constructed_phase3_only",
    }
    return index, rows


def audit_pool_against_protected(
    rows: list[dict[str, Any]], protected: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    return {name: overlap(rows, values) for name, values in protected.items()}


def leak_free_rows(
    rows: list[dict[str, Any]], protected: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    true_sets = [
        values for name, values in protected.items() if name not in RETENTION
    ]
    merged = [row for values in true_sets for row in values]
    return [row for row in rows if not has_overlap(row, merged)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage2_sft_incremental_ablation"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = (project / args.output).resolve()
    shared = output / "shared"
    phase0 = output / "phase0"

    required = [
        OLD_WB,
        OLD_LD,
        NEW_WB,
        FULL_OLD,
        EASY_POOL,
        EXPANSION_LABELS,
        EXPANSION_POOL,
        ATTEMPTS,
        GENERATIONS,
        BENCHMARK_SUMMARY,
        ADDON_IDENTITY,
        ENVIRONMENT,
        ROUND1_CONTRACT,
        M0_POINTER,
    ]
    missing_files = [str(path) for path in required if not (project / path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Phase-0 inputs missing: {missing_files}")

    source_rows = {path: read_jsonl(project / path) for path in EXPECTED_ROWS}
    for path, expected in EXPECTED_ROWS.items():
        if len(source_rows[path]) != expected:
            raise RuntimeError(f"{path}: {len(source_rows[path])} != {expected}")
    old_wb = source_rows[OLD_WB]
    old_ld = source_rows[OLD_LD]
    old_rows = old_wb + old_ld
    full_old = source_rows[FULL_OLD]
    if {record_id(row) for row in old_rows} != {
        record_id(row) for row in full_old
    }:
        raise RuntimeError("Core-WB2000 + Core-LD1000 do not equal the frozen M0 manifest")

    addon_identity = read_json(project / ADDON_IDENTITY)
    environment = read_json(project / ENVIRONMENT)
    round1_contract = read_json(project / ROUND1_CONTRACT)
    pointer = read_json(project / M0_POINTER)
    contract_payload, generation_contract_hash = generation_contract(
        project, addon_identity
    )
    attempts = read_jsonl(project / ATTEMPTS)
    generations = read_jsonl(project / GENERATIONS)
    classification_manifest, buckets = classify(
        old_rows, attempts, generations, generation_contract_hash
    )
    class_lookup = {
        row["record_id"]: row["classification"] for row in classification_manifest
    }
    filtered_lookup = {
        row["record_id"]: bool(row["filtered_hard"]) for row in classification_manifest
    }
    missing_classification = sum(
        row["classification"] == "missing" for row in classification_manifest
    )

    protected_metadata, protected_rows = protected_index(project)
    new_wb_all = source_rows[NEW_WB]
    old_ids = {record_id(row) for row in old_rows}
    easy_all = read_jsonl(project / EASY_POOL)
    new_easy_all = [row for row in easy_all if record_id(row) not in old_ids]
    labels = read_jsonl(project / EXPANSION_LABELS)
    expansion = {
        record_id(row): row for row in read_jsonl(project / EXPANSION_POOL)
    }
    medium_labels = [
        row for row in labels if str(row.get("difficulty") or "") == "medium"
    ]
    medium_all = []
    medium_missing_source = []
    for label in medium_labels:
        sample_id = str(label.get("sample_id") or "")
        source = expansion.get(sample_id)
        if source is None:
            medium_missing_source.append(sample_id)
            continue
        merged = dict(source)
        merged["difficulty_annotation"] = label
        medium_all.append(merged)

    new_wb = leak_free_rows(
        [row for row in new_wb_all if record_id(row) not in old_ids],
        protected_rows,
    )
    new_easy = leak_free_rows(
        [row for row in new_easy_all if verified_training_row(row)],
        protected_rows,
    )
    medium_protected_clean = leak_free_rows(
        [
            row
            for row in medium_all
            if verified_training_row(row)
            and bool(proof(row))
            and bool(statement(row))
        ],
        protected_rows,
    )
    medium = []
    seen_medium_groups: set[str] = set()
    for row in sorted(medium_protected_clean, key=record_id):
        group = theorem_group(row)
        if group in seen_medium_groups:
            continue
        seen_medium_groups.add(group)
        medium.append(row)

    classified_rows = [
        row for row in old_rows if class_lookup[record_id(row)] != "missing"
    ]
    stable_rows = [
        row for row in old_rows if class_lookup[record_id(row)] == "stable"
    ]
    frontier_rows = [
        row for row in old_rows if class_lookup[record_id(row)] == "frontier"
    ]
    hard_rows = [row for row in old_rows if class_lookup[record_id(row)] == "hard"]
    filtered_hard_rows = [
        row for row in hard_rows if filtered_lookup[record_id(row)]
    ]

    classification_sufficient = (
        missing_classification <= GENERATION_MISSING_STOP_THRESHOLD
    )
    phase1_bucket_sufficient = (
        len(stable_rows) >= PHASE1_TARGET["stable"]
        and len(frontier_rows) >= PHASE1_TARGET["frontier"]
        and len(filtered_hard_rows) >= PHASE1_TARGET["filtered_hard"]
    )
    phase1_ready = classification_sufficient and phase1_bucket_sufficient

    phase1_rows: list[dict[str, Any]] = []
    selection_notes = []
    if phase1_ready:
        # Deterministic selection.  The current Phase-0 data is expected to stop
        # before this branch because >1500 frozen results are missing.
        stable = sorted(stable_rows, key=lambda row: (source_kind(row), record_id(row)))
        frontier = sorted(
            frontier_rows, key=lambda row: (source_kind(row), record_id(row))
        )
        hard = sorted(
            filtered_hard_rows, key=lambda row: (source_kind(row), record_id(row))
        )
        phase1_rows = (
            stable[: PHASE1_TARGET["stable"]]
            + frontier[: PHASE1_TARGET["frontier"]]
            + hard[: PHASE1_TARGET["filtered_hard"]]
        )
        selection_notes.append(
            "Deterministic source/id order; no replacement; target bucket counts met."
        )
    else:
        selection_notes.append(
            "No executable Phase-1 training manifest was materialized: the Phase-0 "
            "stop rule applies and an empty file prevents accidental training."
        )

    phase1_duplicates = duplicate_audit(phase1_rows)
    phase1_leakage = audit_pool_against_protected(phase1_rows, protected_rows)
    phase1_eos = eos_static_audit(phase1_rows)
    source_manifest_identity = {
        str(path): {
            "path": str(project / path),
            "sha256": sha256(project / path),
            "rows": len(source_rows[path]),
        }
        for path in EXPECTED_ROWS
    }

    merge_summary_exists = (project / M0_MERGE_IDENTITY).is_file()
    merge_summary = (
        read_json(project / M0_MERGE_IDENTITY) if merge_summary_exists else None
    )
    merged_exists = (project / M0_MERGED / "config.json").is_file()
    merged_model_read_only = (
        merged_exists
        and not bool((project / M0_MERGED / "model.safetensors").stat().st_mode & 0o222)
    )
    if merged_exists and merge_summary and merge_summary.get("passed"):
        m0_identity_status = "FROZEN_M0_READY"
    elif merge_summary_exists:
        m0_identity_status = "FROZEN_M0_CHECKPOINT_EQUIVALENCE_FAILED"
    else:
        m0_identity_status = "FROZEN_M0_MERGE_OR_EQUIVALENCE_PENDING"
    frozen_m0_identity = {
        "frozen_model_name": FROZEN_M0,
        "selected_source_model": SOURCE_MODEL,
        "pointer": pointer,
        "source_identity": addon_identity,
        "source_model_form": "base_plus_lora_adapter",
        "merged_model_path": str(project / M0_MERGED),
        "merged_model_exists": merged_exists,
        "merged_model_read_only": merged_model_read_only,
        "checkpoint_equivalence_path": str(project / M0_MERGE_IDENTITY),
        "checkpoint_equivalence_exists": merge_summary_exists,
        "checkpoint_equivalence_passed": (
            bool(merge_summary and merge_summary.get("passed"))
        ),
        "checkpoint_equivalence_summary": merge_summary,
        "identity_status": m0_identity_status,
    }

    stage2_contract = {
        "phase_scope": "Phase 0 only; no Trainer; no GPU training",
        "starting_model": FROZEN_M0,
        "epochs": 1,
        "total_rows": 2000,
        "effective_batch_size": 16,
        "learning_rate": 1e-5,
        "max_seq_length": 1024,
        "completion_only_loss": True,
        "packing": False,
        "sampling": "fixed_manifest_without_replacement",
        "require_supervised_eos": True,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "inherited_round1_contract": round1_contract,
        "phase1_replay_target": PHASE1_TARGET | {"total": 2000},
        "generation_classification_contract": contract_payload,
        "generation_contract_hash": generation_contract_hash,
    }

    leakage = {
        "old_rows": audit_pool_against_protected(old_rows, protected_rows),
        "new_wb": audit_pool_against_protected(new_wb, protected_rows),
        "new_ld_easy": audit_pool_against_protected(new_easy, protected_rows),
        "ld_medium": audit_pool_against_protected(medium, protected_rows),
        "phase1_manifest": phase1_leakage,
        "retention_overlap_is_allowed_but_not_unseen": True,
    }
    eos = {
        "old_rows": eos_static_audit(old_rows),
        "new_wb": eos_static_audit(new_wb),
        "new_ld_easy": eos_static_audit(new_easy),
        "ld_medium": eos_static_audit(medium),
        "phase1_manifest": phase1_eos,
        "frozen_round1_eos_contract_path": str(project / ROUND1_CONTRACT),
        "frozen_round1_eos_token_id": (
            round1_contract.get("effective", {}).get("eos_token_id")
        ),
    }

    by_class_source = {}
    for bucket_name, rows in (
        ("stable", stable_rows),
        ("frontier", frontier_rows),
        ("raw_hard", hard_rows),
        ("filtered_hard", filtered_hard_rows),
        ("missing", buckets["missing"]),
    ):
        by_class_source[bucket_name] = {
            "total": len(rows),
            "WB": sum(source_kind(row) == "WB" for row in rows),
            "LD": sum(source_kind(row) == "LD" for row in rows),
        }

    data_pool_summary = {
        "status": (
            "BLOCKED_MISSING_FROZEN_M0_GENERATIONS"
            if not classification_sufficient
            else "AUDITED"
        ),
        "old_total": len(old_rows),
        "classified_total": len(classified_rows),
        "missing_generation_results": missing_classification,
        "missing_generation_stop_threshold": GENERATION_MISSING_STOP_THRESHOLD,
        "classification_distribution": by_class_source,
        "available_pools": {
            "new_wb": {
                "raw": len(new_wb_all),
                "verified_unused_protected_clean": len(new_wb),
            },
            "new_ld_easy": {
                "reviewed_easy_pool": len(easy_all),
                "unused_after_round1": len(new_easy_all),
                "verified_unused_protected_clean": len(new_easy),
            },
            "reviewed_ld_medium": {
                "semantic_labels": len(medium_labels),
                "joined_to_frozen_source": len(medium_all),
                "missing_source_rows": len(medium_missing_source),
                "verified_protected_clean_before_dedup": len(medium_protected_clean),
                "removed_theorem_group_duplicates": (
                    len(medium_protected_clean) - len(medium)
                ),
                "verified_protected_clean_unique": len(medium),
            },
        },
        "pool_statistics": {
            "old_classified": pool_stats(classified_rows),
            "old_stable": pool_stats(stable_rows),
            "old_frontier": pool_stats(frontier_rows),
            "old_raw_hard": pool_stats(hard_rows),
            "old_filtered_hard": pool_stats(filtered_hard_rows),
            "new_wb": pool_stats(new_wb),
            "new_ld_easy": pool_stats(new_easy),
            "ld_medium": pool_stats(medium),
        },
        "source_manifest_identity": source_manifest_identity,
        "protected_leakage_audit": leakage,
        "eos_preflight": eos,
        "duplicate_audit": {
            "old": duplicate_audit(old_rows),
            "new_wb": duplicate_audit(new_wb),
            "new_ld_easy": duplicate_audit(new_easy),
            "ld_medium": duplicate_audit(medium),
        },
    }

    phase1_token_audit = {
        "status": (
            "READY"
            if phase1_ready
            else "BLOCKED_INSUFFICIENT_FROZEN_CLASSIFICATION"
        ),
        "target_rows": 2000,
        "selected_rows": len(phase1_rows),
        "target_buckets": PHASE1_TARGET,
        "selected_buckets": dict(
            Counter(
                (
                    "filtered_hard"
                    if filtered_lookup.get(record_id(row))
                    else class_lookup.get(record_id(row), "unknown")
                )
                for row in phase1_rows
            )
        ),
        "selected_source_distribution": dict(
            Counter(source_kind(row) for row in phase1_rows)
        ),
        "target_source_ratio": "WB:LD approximately 2:1 without replacement",
        "supervised_label_tokens": {
            "sum": sum(proof_tokens(row) for row in phase1_rows),
            **numeric_stats([proof_tokens(row) for row in phase1_rows]),
        },
        "total_tokens": {
            "sum": sum(int(row.get("total_tokens") or 0) for row in phase1_rows),
            **numeric_stats(
                [int(row.get("total_tokens") or 0) for row in phase1_rows]
            ),
        },
        "proof_length": numeric_stats([proof_tokens(row) for row in phase1_rows]),
        "duplicates": phase1_duplicates,
        "leakage": phase1_leakage,
        "eos_preflight": phase1_eos,
        "selection_notes": selection_notes,
        "token_matching_status": (
            "NOT_APPLICABLE_UNTIL_EXECUTABLE_MANIFEST_EXISTS"
            if not phase1_ready
            else "CONTROL_IS_THE_REFERENCE_FOR_LATER_PHASES"
        ),
    }

    blockers = []
    if not classification_sufficient:
        baseline_summary = read_json(project / BENCHMARK_SUMMARY)
        baseline_attempts = int(baseline_summary.get("recorded_attempt_results") or 600)
        estimated_candidates = missing_classification * 2
        generation_seconds = float(baseline_summary.get("generation_seconds") or 0.0)
        verification_seconds = float(
            baseline_summary.get("verification_seconds") or 0.0
        )
        total_seconds = float(baseline_summary.get("total_seconds") or 0.0)
        estimated_generation_hours = (
            generation_seconds / baseline_attempts * estimated_candidates / 3600
        )
        estimated_verification_hours = (
            verification_seconds / baseline_attempts * estimated_candidates / 3600
        )
        estimated_total_hours = (
            total_seconds / baseline_attempts * estimated_candidates / 3600
        )
        blockers.append(
            {
                "code": "MISSING_FROZEN_M0_GENERATION_RESULTS_GT_1500",
                "detail": (
                    f"{missing_classification} of 3000 old rows lack reusable results; "
                    f"threshold is {GENERATION_MISSING_STOP_THRESHOLD}."
                ),
                "required_action": (
                    "User decision required before any large missing-only generation. "
                    "Existing 150 rows must not be regenerated; each missing row would "
                    "use two candidates under the frozen contract."
                ),
                "estimated_additional_candidates": estimated_candidates,
                "estimated_cost_from_retention150_baseline": {
                    "baseline_candidates": baseline_attempts,
                    "baseline_total_seconds": total_seconds,
                    "estimated_generation_hours": round(
                        estimated_generation_hours, 3
                    ),
                    "estimated_verification_hours": round(
                        estimated_verification_hours, 3
                    ),
                    "estimated_end_to_end_hours": round(estimated_total_hours, 3),
                    "caveat": (
                        "Linear extrapolation from the existing 600-candidate "
                        "retention150 run; startup, batching, and theorem mix may "
                        "change actual wall time."
                    ),
                },
            }
        )
    if not phase1_bucket_sufficient:
        blockers.append(
            {
                "code": "PHASE1_BUCKET_COUNTS_NOT_ESTABLISHED",
                "detail": (
                    "Known stable/frontier/filtered-hard counts cannot satisfy the "
                    "400/1200/400 replay contract before missing rows are classified."
                ),
            }
        )
    if len(new_easy) < 400:
        blockers.append(
            {
                "code": "PHASE2_INSUFFICIENT_NEW_LD_EASY",
                "detail": f"{len(new_easy)} verified unused rows available; 400 required.",
                "affects_phase1": False,
            }
        )
    if len(medium) < 264:
        blockers.append(
            {
                "code": "PHASE3_BLOCKED_INSUFFICIENT_MEDIUM_DATA",
                "detail": (
                    f"{len(medium)} verified protected-clean medium rows available; "
                    "at least 264 are required for 200 train + 64 holdout."
                ),
                "affects_phase1": False,
            }
        )
    if not frozen_m0_identity["checkpoint_equivalence_passed"]:
        blockers.append(
            {
                "code": (
                    "M0_CHECKPOINT_EQUIVALENCE_FAILED"
                    if merge_summary_exists
                    else "M0_MERGED_EQUIVALENCE_PENDING"
                ),
                "detail": (
                    "Merged/unmerged 24-prompt M0 canary failed the frozen gate."
                    if merge_summary_exists
                    else "Merged/unmerged 24-prompt M0 canary must pass before Phase 1."
                ),
                "evidence": merge_summary,
            }
        )

    next_phase_decision = {
        "phase0_completed": True,
        "phase1_authorized": phase1_ready
        and frozen_m0_identity["checkpoint_equivalence_passed"],
        "phase1_manifest_rows": len(phase1_rows),
        "phase2_data_sufficient": len(new_wb) >= 800 and len(new_easy) >= 400,
        "phase3_data_sufficient": len(medium) >= 264,
        "blockers": blockers,
        "required_user_instruction": (
            "Explicitly approve the missing-only frozen-M0 generation plan before "
            "Phase 1 can be prepared."
        ),
        "trainer_created": False,
        "gpu_training_started": False,
        "later_phase_started": False,
    }

    write_json(shared / "environment.json", environment)
    write_json(shared / "frozen_m0_identity.json", frozen_m0_identity)
    write_json(shared / "frozen_training_contract.json", stage2_contract)
    write_json(shared / "protected_eval_index.json", protected_metadata)
    write_json(
        shared / "experiment_contract.json",
        {
            "name": "stage2_sft_incremental_ablation",
            "current_phase": "phase0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase_order": [
                "phase0",
                "phase1_replay_control",
                "phase2_new_easy",
                "phase3_medium10",
                "phase4_strict",
            ],
            "automatic_phase_transition": False,
            "phase0_only": True,
            "frozen_model": FROZEN_M0,
            "stage2_contract_sha256": canonical_hash(stage2_contract),
            "protected_eval_index_sha256": canonical_hash(protected_metadata),
        },
    )
    write_json(phase0 / "data_pool_summary.json", data_pool_summary)
    write_jsonl(phase0 / "classification_manifest.jsonl", classification_manifest)
    write_jsonl(phase0 / "phase1_replay_control_manifest.jsonl", phase1_rows)
    write_json(phase0 / "phase1_token_audit.json", phase1_token_audit)
    write_json(phase0 / "next_phase_decision.json", next_phase_decision)

    report = f"""# Stage-2 SFT Phase 0 preflight

## Outcome

Phase 0 completed as a data-only audit. No Trainer was created, no GPU training
was started, and no later phase was entered.

Phase 1 is **not authorized**. Only {len(classified_rows)} of 3000 frozen
round-1 rows have reusable candidate-level M0 results; {missing_classification}
are missing, which exceeds the automatic-generation stop threshold of
{GENERATION_MISSING_STOP_THRESHOLD}. The Phase-1 manifest is intentionally
empty to prevent accidental training.

## Frozen identity

- Frozen name: `{FROZEN_M0}`
- Source: `{SOURCE_MODEL}`
- Source checkpoint tree SHA-256: `{addon_identity['checkpoint_tree_sha256']}`
- Round-1 data: WB 2000 + semantically reviewed LD-easy 1000
- Environment SHA-256: `{environment['environment_hash']}`
- Merged model present: `{merged_exists}`
- Merged/unmerged 24-prompt canary passed: `{frozen_m0_identity['checkpoint_equivalence_passed']}`

## Old-data classification

| Bucket | WB | LD | Total |
|---|---:|---:|---:|
| Stable/Core | {by_class_source['stable']['WB']} | {by_class_source['stable']['LD']} | {by_class_source['stable']['total']} |
| Frontier | {by_class_source['frontier']['WB']} | {by_class_source['frontier']['LD']} | {by_class_source['frontier']['total']} |
| Raw Hard | {by_class_source['raw_hard']['WB']} | {by_class_source['raw_hard']['LD']} | {by_class_source['raw_hard']['total']} |
| Filtered Hard | {by_class_source['filtered_hard']['WB']} | {by_class_source['filtered_hard']['LD']} | {by_class_source['filtered_hard']['total']} |
| Missing frozen results | {by_class_source['missing']['WB']} | {by_class_source['missing']['LD']} | {by_class_source['missing']['total']} |

The reused result source is the frozen `wb_train_retention150` evaluation:
four candidates per row, M0 adapter identity fixed, vLLM temperature 0.8,
top-p 0.95, max 256 new tokens, and Pantograph verification in the frozen
environment. Existing results were not regenerated.

## Future data availability

| Pool | Available | Requirement | Status |
|---|---:|---:|---|
| New WB | {len(new_wb)} | 800 for Phase 2 | {'sufficient' if len(new_wb) >= 800 else 'insufficient'} |
| New LD-easy | {len(new_easy)} | 400 for Phase 2 | {'sufficient' if len(new_easy) >= 400 else 'insufficient'} |
| Reviewed LD-medium | {len(medium)} | 200 train + 64 holdout | {'sufficient' if len(medium) >= 264 else 'blocked_insufficient_medium_data'} |

The LD labels are semantic Codex reviews, not token-threshold labels. Detailed
token, source-file, domain, tactic and premise distributions are in
`data_pool_summary.json`.

## Gates

- Frozen old manifest identity: 3000 rows, no replacement.
- Phase-1 executable manifest: 0 rows (blocked by the explicit >1500 stop rule).
- Missing-only generation estimate: {missing_classification * 2} candidates
  ({missing_classification} rows × 2), approximately
  {blockers[0]['estimated_cost_from_retention150_baseline']['estimated_end_to_end_hours']}
  hours end-to-end by linear extrapolation from the existing retention150 run;
  no generation was started.
- Protected-set checks cover record ID, theorem group, qualified theorem, exact
  statement, normalized statement and theorem-qualified proof variant.
- Retention150 overlap is reported but is not treated as unseen leakage.
- Static EOS checks cover non-empty proof, non-zero labels, no semantic
  truncation and current-environment verification. Actual supervised EOS
  materialization is inherited for the frozen old rows; a Phase-1 effective
  manifest cannot be checked until it legally exists.

## Blockers

1. `{blockers[0]['code']}`: {blockers[0]['detail']}
2. Phase 1 bucket counts cannot be frozen until missing old rows are classified.
3. Phase 2 currently lacks 400 unused verified LD-easy rows.
4. Phase 3 currently lacks the minimum 264 reviewed medium rows.
5. M0 merged/unmerged equivalence failed: first token and extraction outcome
   matched 24/24, while major common prefix and Pantograph outcome matched only
   21/24.

## Decision

Do not execute Phase 1. Await an explicit user decision on the missing-only M0
generation budget; existing results must remain untouched.
"""
    (phase0 / "preflight_report.md").write_text(report, encoding="utf-8")
    readme = """# Stage-2 SFT incremental ablation

Only Phase 0 has been executed. The directory contains frozen identities, the
protected-evaluation index, old-data classifications, future-pool audits, and
the Phase-1 readiness decision. Training-phase directories are intentionally
absent.

Phase 1 is not authorized while `phase0/next_phase_decision.json` reports
blockers. The empty Phase-1 manifest is a safety stop, not a valid training
manifest.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    # Re-read critical artifacts so a partial write cannot be mistaken for a
    # completed Phase 0.
    if len(read_jsonl(phase0 / "classification_manifest.jsonl")) != 3000:
        raise RuntimeError("classification manifest did not freeze 3000 rows")
    if phase1_ready and len(read_jsonl(phase0 / "phase1_replay_control_manifest.jsonl")) != 2000:
        raise RuntimeError("ready Phase-1 manifest is not 2000 rows")
    print(
        json.dumps(
            {
                "phase0": "completed",
                "classified": len(classified_rows),
                "missing": missing_classification,
                "phase1_authorized": next_phase_decision["phase1_authorized"],
                "phase1_rows": len(phase1_rows),
                "new_wb": len(new_wb),
                "new_ld_easy": len(new_easy),
                "ld_medium": len(medium),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
