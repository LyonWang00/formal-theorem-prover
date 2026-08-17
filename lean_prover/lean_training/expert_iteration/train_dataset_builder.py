"""Build cumulative weighted SFT datasets from anchor data and Proof Bank."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.preparation import (
    NormalizedExample,
    proof_hash as calculate_proof_hash,
    statement_hash as calculate_statement_hash,
)
from lean_prover.lean_training.data.training import (
    build_generation_prompt,
    build_sft_general_data,
    normalize_sft_completion,
)

from .config import CategorySamplingConfig, TrainMixConfig
from .schemas import ProofBankRecord
from .utils import read_jsonl, write_json_atomic, write_jsonl_atomic


def build_iteration_train_dataset(
    *,
    train_seed_path: str | Path,
    selected_proofs: list[ProofBankRecord],
    iteration: int,
    mix: TrainMixConfig,
    category_sampling: CategorySamplingConfig,
    output_path: str | Path,
    manifest_path: str | Path,
    stats_path: str | Path,
    protected_statement_hashes: set[str],
) -> dict[str, Any]:
    """Write a unique-row train set with token-aware sampling weights."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_anchor_rows: set[tuple[str, str]] = set()
    for row in read_jsonl(train_seed_path):
        statement = str(row.get("lean_statement") or row.get("statement") or "").strip()
        statement_hash = (
            calculate_statement_hash(statement)
            if statement
            else str(row.get("statement_hash") or "")
        )
        if statement_hash and statement_hash in protected_statement_hashes:
            raise ValueError(f"protected eval/monitor/benchmark row entered train seed: {row.get('id')}")
        prepared = dict(row)
        prepared["statement_hash"] = statement_hash
        prepared["expert_source"] = "anchor"
        prepared["category"] = str(row.get("category") or "unknown")
        anchor_key = (
            statement_hash or str(row.get("lean_statement") or row.get("statement") or ""),
            str(row.get("completion") or row.get("proof") or "").strip(),
        )
        if anchor_key in seen_anchor_rows:
            continue
        seen_anchor_rows.add(anchor_key)
        groups["anchor"].append(prepared)
    for proof in selected_proofs:
        source_group = "current_expert" if proof.iteration_found == iteration else "historical_expert"
        statement_data = dict(proof.metadata.get("statement") or {})
        statement_hash = str(statement_data.get("statement_hash") or "")
        if statement_hash and statement_hash in protected_statement_hashes:
            raise ValueError(f"protected statement entered Proof Bank train mix: {proof.statement_id}")
        example = NormalizedExample(
            id=proof.statement_id,
            source=str(statement_data.get("source") or "discovery"),
            source_name=str(statement_data.get("source") or "discovery"),
            informal_statement=str((statement_data.get("metadata") or {}).get("informal_statement") or ""),
            lean_statement=str(statement_data.get("statement") or ""),
            proof=proof.proof,
            imports=tuple(statement_data.get("imports") or ()),
            context_lines=tuple(statement_data.get("context_lines") or ()),
            pantograph_verified=True,
        )
        row = build_sft_general_data(example)
        row.update(
            {
                "expert_source": source_group,
                "origin_data_role": "discovery",
                "iteration_found": proof.iteration_found,
                "proof_bank_id": proof.proof_id,
                "category": str(statement_data.get("category") or "unknown"),
            }
        )
        attestation = dict(proof.metadata.get("verification_attestation") or {})
        required_attestation = {
            "record_id",
            "environment_hash",
            "assembler_version",
            "normalization_version",
            "assembled_source_hash",
            "attestation_id",
        }
        missing_attestation = sorted(
            key for key in required_attestation if not attestation.get(key)
        )
        if missing_attestation:
            raise ValueError(
                f"Proof Bank row {proof.proof_id} lacks SFT attestation fields "
                f"{missing_attestation}"
            )
        row.update(
            {
                **attestation,
                "data_role": "train",
                "data_state": "verified",
                "statement_verified": True,
                "proof_verified": True,
                "pantograph_verified": True,
                "verification_status": "verified",
                "reference_proof_verified": True,
            }
        )
        groups[source_group].append(row)
    active_weights = {
        "anchor": mix.anchor,
        "historical_expert": mix.historical_expert,
        "current_expert": mix.current_expert,
    }
    unavailable = [name for name, weight in active_weights.items() if weight and not groups[name]]
    if unavailable:
        available_total = sum(weight for name, weight in active_weights.items() if groups[name])
        if available_total <= 0:
            raise ValueError("no train records are available")
        active_weights = {
            name: (weight / available_total if groups[name] else 0.0)
            for name, weight in active_weights.items()
        }
    group_tokens = {
        name: sum(_completion_tokens(row) for row in rows)
        for name, rows in groups.items()
    }
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    category_examples: dict[str, int] = defaultdict(int)
    category_tokens: dict[str, int] = defaultdict(int)
    for name in ("anchor", "historical_expert", "current_expert"):
        rows = groups[name]
        if active_weights.get(name, 0.0) <= 0:
            continue
        token_total = max(1, group_tokens.get(name, 0))
        per_token_weight = active_weights.get(name, 0.0) / token_total
        base_weights = [per_token_weight * _completion_tokens(row) for row in rows]
        adjusted_weights = _category_adjusted_weights(
            rows,
            base_weights,
            target_total=active_weights.get(name, 0.0),
            config=category_sampling,
        )
        for row, adjusted_weight in zip(rows, adjusted_weights, strict=True):
            completion_tokens = _completion_tokens(row)
            category = str(row.get("category") or "unknown")
            sample_weight = max(adjusted_weight, 1e-12)
            manifest_row = _canonical_manifest_row(
                row,
                source_group=name,
                category=category,
                completion_tokens=completion_tokens,
                sample_weight=sample_weight,
            )
            output_row = _canonical_training_row(
                manifest_row,
                sample_weight=sample_weight,
            )
            all_rows.append(output_row)
            category_examples[category] += 1
            category_tokens[category] += completion_tokens
            manifest.append(manifest_row)
    stats = {
        "iteration": iteration,
        "examples_by_source": dict(
            sorted(Counter(item["source_group"] for item in manifest).items())
        ),
        "available_examples_by_source": {
            name: len(groups[name]) for name in groups
        },
        "completion_tokens_by_source": group_tokens,
        "requested_train_mix": mix.model_dump(),
        "actual_sampling_mix": active_weights,
        "unavailable_groups": unavailable,
        "total_unique_examples": len(all_rows),
        "uses_weighted_sampler": True,
        "category_sampling": category_sampling.model_dump(),
        "examples_by_category": dict(sorted(category_examples.items())),
        "completion_tokens_by_category": dict(sorted(category_tokens.items())),
        "completion_token_share_by_category": {
            category: tokens / max(1, sum(category_tokens.values()))
            for category, tokens in sorted(category_tokens.items())
        },
        "sampling_weight_by_category": {
            category: sum(
                float(row["sample_weight"])
                for row in manifest
                if str(row.get("category") or "unknown") == category
            )
            for category in sorted(category_examples)
        },
    }
    write_jsonl_atomic(output_path, all_rows)
    write_jsonl_atomic(manifest_path, manifest)
    write_json_atomic(stats_path, stats)
    return stats


def _completion_tokens(row: dict[str, Any]) -> int:
    value = row.get("reference_proof_length_tokens")
    if value is not None:
        return max(1, int(value))
    completion = str(row.get("completion") or row.get("proof") or "")
    return max(1, len(completion.split()))


def _canonical_training_row(
    row: dict[str, Any],
    *,
    sample_weight: float,
) -> dict[str, Any]:
    """Project one rich manifest row onto the trainer's minimal schema."""

    prompt = build_generation_prompt(row)
    completion = str(row.get("completion") or row.get("proof") or "")
    return {
        "prompt": prompt,
        "completion": completion,
        "sample_weight": float(sample_weight),
    }


def _canonical_manifest_row(
    row: dict[str, Any],
    *,
    source_group: str,
    category: str,
    completion_tokens: int,
    sample_weight: float,
) -> dict[str, Any]:
    """Normalize legacy/Proof-Bank metadata into an auditable SFT sidecar row."""

    statement = str(row.get("lean_statement") or row.get("statement") or "").strip()
    proof = normalize_sft_completion(
        str(row.get("completion") or row.get("proof") or "").strip()
    )
    if not statement or not proof:
        raise ValueError("expert-iteration SFT row lacks statement or proof")
    if row.get("pantograph_verified") is not True:
        raise ValueError("expert-iteration SFT row is not Pantograph verified")
    metadata = dict(row.get("metadata") or {})
    informal_statement = str(row.get("informal_statement") or "").strip()
    if informal_statement:
        metadata["informal_statement"] = informal_statement
    metadata.update(
        {
            "source_group": source_group,
            "category": category,
            "completion_tokens": completion_tokens,
            "sample_weight": sample_weight,
            "proof_bank_id": str(row.get("proof_bank_id") or ""),
            "iteration_found": int(row.get("iteration_found", -1)),
        }
    )
    return {
        "schema_version": "sft_manifest",
        "data_stage": "sft",
        "split": "train",
        "record_id": str(row.get("record_id") or row.get("id") or ""),
        "lean_statement": statement,
        "proof": proof,
        "imports": [str(value) for value in row.get("imports") or []],
        "context_lines": [str(value) for value in row.get("context_lines") or []],
        "unknown_preamble_lines": [
            str(value) for value in row.get("unknown_preamble_lines") or []
        ],
        "source": str(row.get("source") or row.get("source_name") or source_group),
        "statement_hash": calculate_statement_hash(statement),
        "proof_hash": calculate_proof_hash(proof),
        "pantograph_verified": bool(row.get("pantograph_verified")),
        "verification_scope": "full_proof",
        "source_group": source_group,
        "category": category,
        "completion_tokens": completion_tokens,
        "sample_weight": sample_weight,
        "metadata": metadata,
    }


def _category_adjusted_weights(
    rows: list[dict[str, Any]],
    base_weights: list[float],
    *,
    target_total: float,
    config: CategorySamplingConfig,
) -> list[float]:
    """Apply capped sqrt inverse-frequency balancing without changing group mass."""

    if not rows or not config.enabled or target_total <= 0:
        return base_weights
    category_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        category_counts[str(row.get("category") or "unknown")] += 1
    adjusted = []
    for row, base_weight in zip(rows, base_weights, strict=True):
        category = str(row.get("category") or "unknown")
        factor = (len(rows) / category_counts[category]) ** 0.5
        adjusted.append(base_weight * min(config.max_oversample_factor, factor))
    adjusted_total = sum(adjusted)
    if adjusted_total <= 0:
        return base_weights
    scale = target_total / adjusted_total
    return [weight * scale for weight in adjusted]
