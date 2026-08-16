#!/usr/bin/env python3
"""Rank unresolved NuminaMath diagnostics without rescanning the JSONL file."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


UNKNOWN = re.compile(r"Unknown (?:constant|identifier) `([^`]+)`")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--max-source-chars", type=int, default=30000)
    parser.add_argument("--sample-per-name", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    exclude_manifests = list(args.exclude_manifest)
    for root in args.exclude_root:
        exclude_manifests.extend(root.rglob("candidate_manifest.jsonl"))
    excluded = {
        str(row["record_id"])
        for path in exclude_manifests
        for row in read_jsonl(path)
        if row.get("record_id")
    }
    counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    class_counts: collections.Counter[str] = collections.Counter()
    easy_counts: collections.Counter[str] = collections.Counter()
    class_samples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    connection = sqlite3.connect(args.index_file)
    try:
        query = """
            select record_id, error_head, source_chars, final_placeholder,
                   classification, error_family, easy_family
            from records
            where source_chars <= ?
            order by line_number
        """
        for record_id, error_head, source_chars, final_placeholder, classification, error_family, easy_family in connection.execute(
            query, (args.max_source_chars,)
        ):
            if record_id in excluded:
                continue
            class_counts[str(classification)] += 1
            easy_counts[str(easy_family)] += 1
            if len(class_samples[str(classification)]) < args.sample_per_name:
                class_samples[str(classification)].append({
                    "record_id": record_id,
                    "source_chars": source_chars,
                    "error_family": error_family,
                    "easy_family": easy_family,
                    "error_head": str(error_head or "")[:600],
                })
            names = sorted(set(UNKNOWN.findall(str(error_head or ""))))
            for name in names:
                counts[name] += 1
                if len(samples[name]) < args.sample_per_name:
                    samples[name].append({
                        "record_id": record_id,
                        "source_chars": source_chars,
                        "final_placeholder": bool(final_placeholder),
                        "classification": classification,
                        "error_family": error_family,
                        "easy_family": easy_family,
                    })
    finally:
        connection.close()

    result = {
        "excluded_record_count": len(excluded),
        "max_source_chars": args.max_source_chars,
        "classification_counts": dict(class_counts.most_common()),
        "easy_family_counts": dict(easy_counts.most_common()),
        "classification_samples": dict(class_samples),
        "unknown_names": [
            {"name": name, "count": count, "samples": samples[name]}
            for name, count in counts.most_common(args.limit)
        ],
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
