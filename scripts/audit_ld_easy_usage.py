#!/usr/bin/env python3
"""Report LD-easy identity usage in frozen training and evaluation manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_statement(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def signatures(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "id",
        "record_id",
        "sample_id",
        "source_id",
        "statement_id",
        "theorem_id",
        "theorem_group_id",
        "aggregation_group_id",
        "qualified_name",
        "qualified_theorem",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            values.add(f"id:{value}")
    for key in ("statement", "lean_statement", "training_statement", "theorem"):
        value = normalized_statement(str(row.get(key) or ""))
        if value:
            values.add(f"statement:{value}")
    return values


def signature_index(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        result.update(signatures(row))
    return result


def classify_path(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if "/evaluation/" in text or "/eval" in path.name.lower() or "holdout" in text:
        return "evaluation"
    if "discovery" in text:
        return "discovery"
    return "training_or_data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    freeze = project / "outputs/ld_manual_classification_freeze/two_round_v1"
    manual = read_jsonl(freeze / "combined_manual_labels_1725.jsonl")
    easy_manual = [row for row in manual if str(row.get("difficulty") or "").lower() == "easy"]
    trainable = read_jsonl(freeze / "trainable_easy_identity_1004.jsonl")
    easy_index = signature_index(easy_manual)
    trainable_index = signature_index(trainable)

    patterns = (
        "outputs/**/*manifest*.jsonl",
        "outputs/**/ei_round0_train.jsonl",
        "outputs/**/training_data.jsonl",
        "outputs/**/datasets/*.jsonl",
    )
    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(project.glob(pattern))
    usage: list[dict[str, Any]] = []
    for path in sorted(candidates):
        try:
            rows = read_jsonl(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        easy_matches = sum(bool(signatures(row) & easy_index) for row in rows)
        trainable_matches = sum(bool(signatures(row) & trainable_index) for row in rows)
        if not easy_matches and not trainable_matches:
            continue
        usage.append(
            {
                "path": str(path.relative_to(project)),
                "kind": classify_path(path),
                "rows": len(rows),
                "manual_easy_matches": easy_matches,
                "trainable_easy_matches": trainable_matches,
            }
        )
    payload = {
        "source_pool_rows": 5311,
        "manual_reviewed_rows": len(manual),
        "manual_easy_rows": len(easy_manual),
        "protected_clean_trainable_easy_rows": len(trainable),
        "usage": usage,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
