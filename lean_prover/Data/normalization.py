"""Stable entry points for the three existing dataset normalizers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lean_prover.lean_training.data.preparation import (
    NormalizedExample,
    normalize_generic_lean_example,
    normalize_lean_workbook_example,
    normalize_minif2f_example,
)
from lean_prover.lean_training.data.adapters.kimina import normalize_kimina_grpo_record
from lean_prover.lean_training.data.adapters.minif2f import normalize_minif2f_benchmark_record
from lean_prover.lean_training.data.adapters.numinamath import (
    normalize_numinamath_grpo_record,
    normalize_numinamath_sft_record,
)

from .schemas import NormalizationFamily


def normalize_project_record(
    record: Mapping[str, Any],
    *,
    family: NormalizationFamily,
    index: int,
    source_name: str,
) -> NormalizedExample:
    """Route without changing any existing cleaning/normalization behavior."""

    if family == NormalizationFamily.LEAN_WORKBOOK:
        return normalize_lean_workbook_example(
            record,
            index,
            source_name=source_name,
        )
    if family == NormalizationFamily.MINIF2F:
        return normalize_minif2f_benchmark_record(
            record,
            index,
            source_name=source_name,
        )
    if family == NormalizationFamily.NUMINAMATH_SFT:
        return normalize_numinamath_sft_record(
            record,
            index,
            source_name=source_name,
        )
    if family == NormalizationFamily.NUMINAMATH_GRPO:
        return normalize_numinamath_grpo_record(
            record,
            index,
            source_name=source_name,
        )
    if family == NormalizationFamily.KIMINA_GRPO:
        return normalize_kimina_grpo_record(
            record,
            index,
            source_name=source_name,
        )
    return normalize_generic_lean_example(
        record,
        source_name,
        index,
        source_name=source_name,
    )


__all__ = [
    "NormalizedExample",
    "normalize_generic_lean_example",
    "normalize_lean_workbook_example",
    "normalize_minif2f_example",
    "normalize_minif2f_benchmark_record",
    "normalize_numinamath_sft_record",
    "normalize_numinamath_grpo_record",
    "normalize_kimina_grpo_record",
    "normalize_project_record",
]
