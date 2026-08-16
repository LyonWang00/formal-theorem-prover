#!/usr/bin/env python3
"""Stream-search canonical NuminaMath fail rows by literal source text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--contains", action="append", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    found = 0
    with args.fail_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not all(needle in line for needle in args.contains):
                continue
            row = json.loads(line)
            print(json.dumps({
                "line_number": line_number,
                "record_id": row.get("record_id"),
                "formal_statement": row.get("formal_statement") or row.get("lean_statement"),
                "problem": row.get("problem"),
            }, ensure_ascii=True, sort_keys=True))
            found += 1
            if found >= args.limit:
                break


if __name__ == "__main__":
    main()
