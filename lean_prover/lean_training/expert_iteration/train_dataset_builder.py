"""Build cumulative weighted SFT datasets from anchor data and Proof Bank."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.preparation import (
    NormalizedExample,
    statement_hash as calculate_statement_hash,
)
from lean_prover.lean_training.data.training import build_sft_training_record

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
        )
        row = build_sft_training_record(example)
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
            output_row = _canonical_training_row(row, sample_weight=sample_weight)
            all_rows.append(output_row)
            category_examples[category] += 1
            category_tokens[category] += completion_tokens
            manifest.append(
                {
                    "id": row.get("id"),
                    "statement_hash": row.get("statement_hash"),
                    "source_group": name,
                    "completion_tokens": completion_tokens,
                    "sample_weight": sample_weight,
                    "proof_bank_id": row.get("proof_bank_id"),
                    "category": category,
                }
            )
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
                for row in all_rows
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
    """Project anchor and expert rows onto one stable Arrow-compatible schema."""

    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or row.get("proof") or "")
    return {
        "id": str(row.get("id") or row.get("record_id") or ""),
        "record_id": str(row.get("record_id") or row.get("id") or ""),
        "prompt": prompt,
        "completion": completion,
        "proof": completion,
        "text": str(row.get("text") or prompt + completion),
        "lean_statement": str(row.get("lean_statement") or row.get("statement") or ""),
        "statement_hash": str(row.get("statement_hash") or ""),
        "imports": [str(value) for value in row.get("imports") or []],
        "data_role": "train",
        "data_state": str(row.get("data_state") or "verified"),
        "statement_verified": bool(row.get("statement_verified")),
        "proof_verified": bool(row.get("proof_verified")),
        "pantograph_verified": bool(row.get("pantograph_verified")),
        "environment_hash": str(row.get("environment_hash") or ""),
        "assembler_version": str(row.get("assembler_version") or ""),
        "normalization_version": str(row.get("normalization_version") or ""),
        "assembled_source_hash": str(row.get("assembled_source_hash") or ""),
        "attestation_id": str(row.get("attestation_id") or ""),
        "verification_status": str(row.get("verification_status") or "verified"),
        "expert_source": str(row.get("expert_source") or "anchor"),
        "origin_data_role": str(row.get("origin_data_role") or "train"),
        "proof_bank_id": str(row.get("proof_bank_id") or ""),
        "iteration_found": int(row.get("iteration_found", -1)),
        "category": str(row.get("category") or "unknown"),
        "reference_proof_length_tokens": int(
            row.get("reference_proof_length_tokens") or _completion_tokens(row)
        ),
        "sample_weight": float(sample_weight),
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
