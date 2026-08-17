"""Protected benchmark adapter for verified miniF2F valid/test records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import record_id, split_imports_and_body
from ..preparation import NormalizedExample


def normalize_minif2f_benchmark_record(
    record: Mapping[str, Any],
    index: int,
    *,
    source_name: str = "miniF2F",
) -> NormalizedExample:
    """Adapt a verified miniF2F row while keeping it proof-free."""

    if record.get("pantograph_verified") not in {"success", True}:
        raise ValueError("miniF2F benchmark record is not Pantograph verified")
    split = str(record.get("split") or "").lower()
    if split not in {"valid", "test"}:
        raise ValueError("miniF2F benchmark split must be valid or test")
    imports, header_body = split_imports_and_body(str(record.get("header") or ""))
    statement = str(
        record.get("formal_statement") or record.get("lean_statement") or ""
    ).strip()
    if not statement:
        raise ValueError("miniF2F benchmark record lacks formal_statement")
    return NormalizedExample(
        id=record_id(record, prefix=f"minif2f-{split}", index=index),
        source="minif2f",
        source_name=source_name,
        informal_statement=str(record.get("informal_stmt") or "").strip(),
        lean_statement=statement,
        imports=imports or ("Mathlib",),
        context_lines=tuple(
            line for line in header_body.splitlines() if line.strip()
        ),
        pantograph_verified=True,
    )


__all__ = ["normalize_minif2f_benchmark_record"]
