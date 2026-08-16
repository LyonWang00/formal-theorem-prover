"""Check theorem-statement leakage across prepared Lean datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.lean_training.prepare_datasets import statement_hash


def read_records(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            records.append(row)
    return records


def record_hash(row: Mapping[str, Any]) -> str:
    value = row.get("statement_hash")
    if isinstance(value, str) and value:
        return value
    statement = row.get("lean_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError(f"record {row.get('id')} lacks lean_statement")
    return statement_hash(statement)


def index_by_statement_hash(records: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        indexed[record_hash(row)].append(dict(row))
    return dict(indexed)


def duplicate_rows(indexed: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {key: rows for key, rows in indexed.items() if len(rows) > 1}


def overlap_rows(
    left: Mapping[str, list[dict[str, Any]]],
    right: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    shared = set(left) & set(right)
    return {key: {"left": left[key], "right": right[key]} for key in sorted(shared)}


def compact_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "line_number": row.get("_line_number"),
            "lean_statement": row.get("lean_statement"),
        }
        for row in rows
    ]


def compact_duplicates(duplicates: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        key: compact_rows(rows)
        for key, rows in sorted(duplicates.items())
    }


def compact_overlap(overlap: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        key: {
            "left": compact_rows(value["left"]),
            "right": compact_rows(value["right"]),
        }
        for key, value in overlap.items()
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = read_records(args.train_file)
    validation = read_records(args.validation_file)
    benchmark = read_records(args.benchmark_file)
    indexes = {
        "train": index_by_statement_hash(train),
        "validation": index_by_statement_hash(validation),
        "benchmark": index_by_statement_hash(benchmark),
    }
    duplicates = {
        name: duplicate_rows(index)
        for name, index in indexes.items()
    }
    overlaps = {
        "train_validation": overlap_rows(indexes["train"], indexes["validation"]),
        "train_benchmark": overlap_rows(indexes["train"], indexes["benchmark"]),
        "validation_benchmark": overlap_rows(indexes["validation"], indexes["benchmark"]),
    }
    return {
        "counts": {
            "train_records": len(train),
            "validation_records": len(validation),
            "benchmark_records": len(benchmark),
            "train_unique_statements": len(indexes["train"]),
            "validation_unique_statements": len(indexes["validation"]),
            "benchmark_unique_statements": len(indexes["benchmark"]),
        },
        "duplicates": {
            name: {
                "count": len(value),
                "rows": compact_duplicates(value),
            }
            for name, value in duplicates.items()
        },
        "overlaps": {
            name: {
                "count": len(value),
                "rows": compact_overlap(value),
            }
            for name, value in overlaps.items()
        },
    }


def enforce_policy(report: Mapping[str, Any], args: argparse.Namespace) -> None:
    overlaps = report["overlaps"]
    violations: dict[str, int] = {}
    if overlaps["train_validation"]["count"] and not args.allow_train_validation_overlap:
        violations["train_validation"] = overlaps["train_validation"]["count"]
    if overlaps["train_benchmark"]["count"] and not args.allow_train_benchmark_overlap:
        violations["train_benchmark"] = overlaps["train_benchmark"]["count"]
    if (
        overlaps["validation_benchmark"]["count"]
        and not args.allow_validation_benchmark_overlap
    ):
        violations["validation_benchmark"] = overlaps["validation_benchmark"]["count"]
    if violations:
        raise ValueError(
            "statement leakage detected; pass explicit --allow_* flags to continue: "
            + json.dumps(violations, ensure_ascii=False)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prepared Lean dataset leakage.")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--benchmark_file", default=None)
    parser.add_argument("--report_output", default=None)
    parser.add_argument("--allow_train_validation_overlap", action="store_true")
    parser.add_argument("--allow_train_benchmark_overlap", action="store_true")
    parser.add_argument("--allow_validation_benchmark_overlap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report_output:
        Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_output).write_text(text, encoding="utf-8")
    print(text, end="")
    enforce_policy(report, args)


if __name__ == "__main__":
    main()
