#!/usr/bin/env python3
"""Select compact, non-frozen NuminaMath rows for manual proof review."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from lean_prover.Dataset.numinamath_repair_campaign import IndexedFailRows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--classification", default="missing_proof_clean")
    parser.add_argument("--error-family")
    parser.add_argument("--easy-family")
    parser.add_argument("--min-source-chars", type=int, default=0)
    parser.add_argument("--max-source-chars", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--list-classifications", action="store_true")
    parser.add_argument("--list-easy-families", action="store_true")
    parser.add_argument("--list-error-families", action="store_true")
    args = parser.parse_args()

    if args.list_classifications:
        connection = sqlite3.connect(args.index_file)
        try:
            for classification, count in connection.execute(
                "select classification, count(*) from records "
                "group by classification order by count(*) desc, classification asc"
            ):
                print(json.dumps({"classification": classification, "count": count}))
        finally:
            connection.close()
        return

    if args.list_easy_families or args.list_error_families:
        column = "easy_family" if args.list_easy_families else "error_family"
        connection = sqlite3.connect(args.index_file)
        try:
            for value, count in connection.execute(
                f"select {column}, count(*) from records "
                f"group by {column} order by count(*) desc, {column} asc"
            ):
                print(json.dumps({column: value, "count": count}))
        finally:
            connection.close()
        return

    manifests = list(args.exclude_manifest)
    for root in args.exclude_root:
        manifests.extend(root.rglob("candidate_manifest.jsonl"))
    excluded = {
        str(row["record_id"])
        for path in manifests
        for row in read_jsonl(path)
        if row.get("record_id")
    }

    where = ["classification=?", "source_chars between ? and ?"]
    parameters: list[Any] = [
        args.classification,
        args.min_source_chars,
        args.max_source_chars,
    ]
    if args.error_family:
        where.append("error_family=?")
        parameters.append(args.error_family)
    if args.easy_family:
        where.append("easy_family=?")
        parameters.append(args.easy_family)
    connection = sqlite3.connect(args.index_file)
    try:
        candidates = []
        for record_id, source_chars, error_family in connection.execute(
            f"select record_id,source_chars,error_family from records where {' and '.join(where)} "
            "order by source_chars asc, record_id asc",
            parameters,
        ):
            if record_id in excluded:
                continue
            candidates.append((record_id, source_chars, error_family))
            if len(candidates) >= args.limit:
                break
    finally:
        connection.close()

    indexed = IndexedFailRows(args.fail_file, args.index_file)
    try:
        for record_id, source_chars, error_family in candidates:
            row = indexed.get_row(record_id)
            compact = {
                "record_id": record_id,
                "source_chars": source_chars,
                "error_family": error_family,
                "problem": row.get("problem"),
                "formal_statement": row.get("formal_statement") or row.get("lean_statement"),
                "proof": row.get("proof") or row.get("formal_proof"),
                "error_message": (row.get("repair_error") or row.get("error_message") or "")[:1000],
            }
            print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    finally:
        indexed.close()


if __name__ == "__main__":
    main()
