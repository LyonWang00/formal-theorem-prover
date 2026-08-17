"""Registry and public exports for source- and role-specific data adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..preparation import NormalizedExample
from .kimina import normalize_kimina_grpo_record
from .lean_workbook import (
    normalize_lean_workbook_record,
    reconstruct_lean_workbook_records,
)
from .leandojo import (
    normalize_leandojo_sft_record,
    reconstruct_leandojo_record,
    reconstruct_leandojo_records,
)
from .minif2f import normalize_minif2f_benchmark_record
from .numinamath import (
    normalize_numinamath_grpo_record,
    normalize_numinamath_sft_record,
)

Adapter = Callable[..., NormalizedExample]

ADAPTERS: dict[str, Adapter] = {
    "lean-workbook": normalize_lean_workbook_record,
    "leandojo-sft": normalize_leandojo_sft_record,
    "numinamath-sft": normalize_numinamath_sft_record,
    "numinamath-grpo": normalize_numinamath_grpo_record,
    "kimina-grpo": normalize_kimina_grpo_record,
    "minif2f": normalize_minif2f_benchmark_record,
}


def get_adapter(dataset_kind: str) -> Adapter | None:
    """Return the registered ordinary-row adapter for a normalized kind."""

    return ADAPTERS.get(dataset_kind)


def normalize_with_adapter(
    dataset_kind: str,
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str,
) -> NormalizedExample | None:
    """Normalize with a registered adapter, or return ``None`` for generic data."""

    adapter = get_adapter(dataset_kind)
    if adapter is None:
        return None
    return adapter(record, index, source_name=source_name)


__all__ = [
    "ADAPTERS",
    "get_adapter",
    "normalize_kimina_grpo_record",
    "normalize_lean_workbook_record",
    "normalize_leandojo_sft_record",
    "normalize_minif2f_benchmark_record",
    "normalize_numinamath_grpo_record",
    "normalize_numinamath_sft_record",
    "normalize_with_adapter",
    "reconstruct_leandojo_record",
    "reconstruct_leandojo_records",
    "reconstruct_lean_workbook_records",
]
