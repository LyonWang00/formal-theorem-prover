"""Workflow-specific record formatting for SFT, GRPO, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from datasets import Dataset, load_dataset

from lean_prover.lean_training.data.preparation import (
    LeanPreamble,
    NormalizedExample,
    contains_forbidden_proof_token,
    lean_code_tokens,
    percentile,
    proof_hash,
    statement_hash,
)


def load_prepared_dataset(path: str) -> Dataset:
    """Load a prepared JSON or JSONL file as a Hugging Face dataset."""

    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    if dataset_path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"expected a .json or .jsonl file, got {dataset_path}")
    return load_dataset("json", data_files=str(dataset_path), split="train")


def build_generation_prompt(
    example: NormalizedExample | Mapping[str, Any],
) -> str:
    """Compose the proof-free prompt shared by SFT, GRPO, and evaluation."""

    if isinstance(example, Mapping):
        existing = example.get("prompt")
        if isinstance(existing, str) and existing.strip():
            return existing
        informal_statement = str(example.get("informal_statement") or "").strip()
        lean_statement = str(example.get("lean_statement") or "").strip()
    else:
        informal_statement = example.informal_statement.strip()
        lean_statement = example.lean_statement.strip()
    return (
        "### Informal statement\n"
        f"{informal_statement}\n\n"
        "### Lean statement\n"
        f"{lean_statement}\n\n"
        "### Lean proof\n"
    )


def proof_length_metrics(proof: str) -> dict[str, int]:
    """Measure a proof without exposing its contents to a GRPO prompt record."""

    normalized = proof.strip()
    return {
        "tokens": sum(1 for _ in lean_code_tokens(normalized)),
        "characters": len(normalized),
        "lines": len(normalized.splitlines()) if normalized else 0,
    }


def build_sft_training_record(example: NormalizedExample) -> dict[str, Any]:
    """Build a supervised row containing prompt, theorem, and target proof."""

    proof = _require_reference_proof(example, workflow="SFT")
    prompt = build_generation_prompt(example)
    record = _base_prompt_record(example, prompt=prompt)
    record.update(
        {
            "proof": proof,
            "completion": proof,
            "text": prompt + proof,
            **_reference_proof_metadata(proof),
        }
    )
    return record


def build_sft_evaluation_record(example: NormalizedExample) -> dict[str, Any]:
    """Build a proof-free row for generative SFT model evaluation."""

    return _base_prompt_record(example, prompt=build_generation_prompt(example))


def build_grpo_training_record(example: NormalizedExample) -> dict[str, Any]:
    """Build a proof-free GRPO row with only non-reversible reference metadata."""

    proof = _require_reference_proof(example, workflow="GRPO")
    record = _base_prompt_record(example, prompt=build_generation_prompt(example))
    record.update(_reference_proof_metadata(proof))
    return record


def build_grpo_evaluation_record(example: NormalizedExample) -> dict[str, Any]:
    """Build a proof-free GRPO evaluation row usable by the reward function."""

    return build_grpo_training_record(example)


def load_statement_hashes(paths: Iterable[str | Path]) -> set[str]:
    """Load statement hashes from prepared SFT files for optional GRPO exclusion."""

    hashes: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SFT overlap file not found: {path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = str(row.get("statement_hash") or "").strip()
                if not value:
                    statement = str(row.get("lean_statement") or "").strip()
                    if not statement:
                        raise ValueError(
                            f"{path}:{line_number} has neither statement_hash "
                            "nor lean_statement"
                        )
                    value = statement_hash(statement)
                hashes.add(value)
    return hashes


def exclude_statement_overlaps(
    records: list[NormalizedExample],
    excluded_hashes: set[str],
) -> tuple[list[NormalizedExample], list[NormalizedExample]]:
    """Partition normalized rows by overlap with an existing statement set."""

    kept: list[NormalizedExample] = []
    excluded: list[NormalizedExample] = []
    for record in records:
        target = (
            excluded
            if statement_hash(record.lean_statement) in excluded_hashes
            else kept
        )
        target.append(record)
    return kept, excluded


def filter_sft_records_by_token_length(
    records: list[NormalizedExample],
    *,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    min_completion_tokens: int,
    enabled: bool,
) -> list[NormalizedExample]:
    """Optionally remove rows that would be silently truncated during SFT."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    kept: list[NormalizedExample] = []
    counters = {
        "total": len(records),
        "kept": 0,
        "prompt_too_long": 0,
        "sequence_too_long": 0,
        "completion_too_short": 0,
        "empty_completion": 0,
    }
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    for record in records:
        prompt = build_generation_prompt(record)
        completion = _require_reference_proof(record, workflow="SFT")
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        completion_tokens = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        total_tokens = len(
            tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        )
        prompt_lengths.append(prompt_tokens)
        completion_lengths.append(completion_tokens)
        total_lengths.append(total_tokens)
        reason = None
        if not completion:
            reason = "empty_completion"
        elif completion_tokens < min_completion_tokens:
            reason = "completion_too_short"
        elif prompt_tokens >= max_seq_length:
            reason = "prompt_too_long"
        elif total_tokens > max_seq_length:
            reason = "sequence_too_long"
        if reason is not None:
            counters[reason] += 1
            if enabled:
                continue
        kept.append(record)
    counters["kept"] = len(kept)
    summary = {
        **counters,
        "max_seq_length": max_seq_length,
        "min_completion_tokens": min_completion_tokens,
        "filter_enabled": enabled,
        "prompt_tokens": _length_distribution(prompt_lengths),
        "completion_tokens": _length_distribution(completion_lengths),
        "total_tokens": _length_distribution(total_lengths),
        "overlength_ratio": (
            counters["sequence_too_long"] / len(records) if records else 0.0
        ),
    }
    print("SFT_LENGTH_FILTER_STATS " + json.dumps(summary, ensure_ascii=False))
    return kept


def filter_grpo_records_by_prompt_length(
    records: list[NormalizedExample],
    *,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    enabled: bool,
) -> list[NormalizedExample]:
    """Optionally remove GRPO rows whose proof-free prompts exceed context."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    kept: list[NormalizedExample] = []
    prompt_lengths: list[int] = []
    prompt_too_long = 0
    for record in records:
        prompt_tokens = len(
            tokenizer(
                build_generation_prompt(record),
                add_special_tokens=True,
            )["input_ids"]
        )
        prompt_lengths.append(prompt_tokens)
        if prompt_tokens >= max_seq_length:
            prompt_too_long += 1
            if enabled:
                continue
        kept.append(record)
    print(
        "GRPO_PROMPT_LENGTH_FILTER_STATS "
        + json.dumps(
            {
                "total": len(records),
                "kept": len(kept),
                "prompt_too_long": prompt_too_long,
                "max_seq_length": max_seq_length,
                "filter_enabled": enabled,
                "prompt_tokens": _length_distribution(prompt_lengths),
                "overlength_ratio": (
                    prompt_too_long / len(records) if records else 0.0
                ),
            },
            ensure_ascii=False,
        )
    )
    return kept


def _base_prompt_record(
    example: NormalizedExample,
    *,
    prompt: str,
) -> dict[str, Any]:
    return {
        "id": example.id,
        "source": example.source,
        "data_kind": example.source,
        "source_name": example.source_name or example.source,
        "informal_statement": example.informal_statement,
        "lean_statement": example.lean_statement,
        "statement_hash": statement_hash(example.lean_statement),
        "prompt": prompt,
        "imports": list(example.imports),
        "context_lines": list(example.context_lines),
        "unknown_preamble_lines": list(example.unknown_preamble_lines),
        "preamble": LeanPreamble(
            example.imports,
            example.context_lines,
            example.unknown_preamble_lines,
        ).to_json(),
        "pantograph_verified": example.pantograph_verified,
    }


def _require_reference_proof(example: NormalizedExample, *, workflow: str) -> str:
    proof = example.proof.strip()
    if not proof:
        raise ValueError(f"{workflow} example {example.id} has no reference proof")
    if contains_forbidden_proof_token(proof):
        raise ValueError(
            f"{workflow} example {example.id} contains forbidden proof token"
        )
    return proof


def _reference_proof_metadata(proof: str) -> dict[str, Any]:
    lengths = proof_length_metrics(proof)
    return {
        "reference_proof_hash": proof_hash(proof),
        "reference_proof_length_tokens": lengths["tokens"],
        "reference_proof_length_characters": lengths["characters"],
        "reference_proof_length_lines": lengths["lines"],
        "has_reference_proof": True,
    }


def _length_distribution(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    sorted_lengths = sorted(lengths)
    return {
        "p50": percentile(sorted_lengths, 0.50),
        "p90": percentile(sorted_lengths, 0.90),
        "p95": percentile(sorted_lengths, 0.95),
        "p99": percentile(sorted_lengths, 0.99),
        "max": sorted_lengths[-1],
    }
