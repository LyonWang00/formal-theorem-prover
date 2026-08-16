#!/usr/bin/env python3
"""Read selected canonical fail rows through the campaign byte-offset index."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from lean_prover.Dataset.numinamath_repair_campaign import IndexedFailRows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--max-proof-chars", type=int, default=8000)
    parser.add_argument("--max-error-chars", type=int, default=3000)
    args = parser.parse_args()

    ids = list(args.record_id)
    connection = sqlite3.connect(args.index_file)
    try:
        for prefix in args.prefix:
            ids.extend(
                row[0]
                for row in connection.execute(
                    "select record_id from records where record_id like ? order by line_number",
                    (prefix + "%",),
                )
            )
    finally:
        connection.close()

    indexed = IndexedFailRows(args.fail_file, args.index_file)
    try:
        for record_id in dict.fromkeys(ids):
            row = indexed.get_row(record_id)
            if args.compact:
                row = {
                    "record_id": row.get("record_id"),
                    "problem": row.get("problem"),
                    "formal_statement": row.get("formal_statement") or row.get("lean_statement"),
                    "proof": (row.get("proof") or row.get("formal_proof") or "")[: args.max_proof_chars],
                    "error_message": (row.get("repair_error") or row.get("error_message") or "")[: args.max_error_chars],
                    "record_hash": row.get("record_hash"),
                }
            print(json.dumps(row, ensure_ascii=True, sort_keys=True))
    finally:
        indexed.close()


if __name__ == "__main__":
    main()
