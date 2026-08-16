#!/usr/bin/env python3
"""Create compact human-review samples for the need_decompose partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=10)
    args = parser.parse_args()

    selected: dict[str, list[dict[str, Any]]] = {
        "top_level_helper_with_placeholder": [],
        "local_helper_with_placeholder": [],
    }
    for row in rows(args.input):
        metadata = row["need_decompose"]
        source = str(row.get(metadata["source_field"]) or "")
        source_lines = source.splitlines()
        for evidence in metadata["evidence"]:
            kind = str(evidence["kind"])
            if kind not in selected or len(selected[kind]) >= args.per_kind:
                continue
            declaration_line = int(evidence["declaration_line"])
            placeholder_line = int(evidence["placeholder_line"])
            start = max(0, min(declaration_line, placeholder_line) - 2)
            end = min(len(source_lines), max(declaration_line, placeholder_line) + 2)
            selected[kind].append({
                "record_id": row["record_id"],
                "helper_name": evidence["helper_name"],
                "declaration_line": declaration_line,
                "placeholder_line": placeholder_line,
                "source_excerpt": [
                    {"line": index + 1, "text": source_lines[index]}
                    for index in range(start, end)
                ],
            })
        if all(len(value) >= args.per_kind for value in selected.values()):
            break

    report = {
        "schema_version": "numinamath_need_decompose_sample_audit_v1",
        "sample_size": sum(len(value) for value in selected.values()),
        "samples_by_kind": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample_size": report["sample_size"], "counts": {key: len(value) for key, value in selected.items()}}, indent=2))


if __name__ == "__main__":
    main()
