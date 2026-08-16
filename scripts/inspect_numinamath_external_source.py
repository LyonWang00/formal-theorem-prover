#!/usr/bin/env python3
"""Print compact schemas and samples for candidate NuminaMath recovery sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(value: object) -> object:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return value[:3]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f"\nSOURCE {path} bytes={path.stat().st_size}")
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            print(pq.read_schema(path))
            samples = pq.read_table(path).slice(0, 2).to_pylist()
        else:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            print(f"json_type={type(value).__name__} rows={len(value) if isinstance(value, list) else 'n/a'}")
            samples = value[:2] if isinstance(value, list) else [value]
        for sample in samples:
            print(json.dumps({key: compact(value) for key, value in sample.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
