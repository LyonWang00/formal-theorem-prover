#!/usr/bin/env python3
"""Materialize Codex-authored judgments with metric-locked blank templates.

This utility does not infer, rank, or change any judgment field.  It only
attaches frozen identity/metric fields and rejects missing or extra judgments.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JUDGMENT_FIELDS = {
    "difficulty",
    "confidence",
    "reason_summary",
    "evidence",
    "trainable_for_short_whole_proof",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty-dir",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/difficulty"),
    )
    parser.add_argument("--classification-version", default="ld-manual-v1")
    args = parser.parse_args()
    root = args.difficulty_dir.resolve()
    template_paths = sorted((root / "label_templates").glob("batch_*.jsonl"))
    if not template_paths:
        raise FileNotFoundError("no frozen blank templates")
    materialized: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for template_path in template_paths:
        judgment_path = root / "manual_judgments" / template_path.name
        if not judgment_path.is_file():
            continue
        templates = read_jsonl(template_path)
        judgments = read_jsonl(judgment_path)
        by_id = {str(row.get("sample_id") or ""): row for row in judgments}
        expected = {str(row["sample_id"]) for row in templates}
        if len(by_id) != len(judgments) or set(by_id) != expected:
            raise ValueError(
                f"{judgment_path} must contain exactly one judgment for every "
                f"template ID; expected={len(expected)}, actual={len(by_id)}"
            )
        for template in templates:
            sample_id = str(template["sample_id"])
            judgment = by_id[sample_id]
            missing = JUDGMENT_FIELDS - set(judgment)
            extras = set(judgment) - JUDGMENT_FIELDS - {"sample_id"}
            if missing or extras:
                raise ValueError(
                    f"{sample_id}: missing={sorted(missing)}, extras={sorted(extras)}"
                )
            row = dict(template)
            for field in JUDGMENT_FIELDS:
                row[field] = judgment[field]
            row["reviewed_by"] = "codex"
            row["classification_version"] = args.classification_version
            row["classified_at"] = timestamp
            materialized.append(row)
    if not materialized:
        raise FileNotFoundError("no Codex-authored manual judgment batches")
    write_jsonl(root / "difficulty_labels.jsonl", materialized)
    print(
        json.dumps(
            {
                "materialized_labels": len(materialized),
                "completed_batches": len(materialized) // 25,
                "classification_version": args.classification_version,
                "judgments_inferred": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
