"""Read-only diagnostics for the temporary NuminaMath failure index or a batch."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def inspect_index(path: Path, limit: int) -> None:
    with sqlite3.connect(path) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )]
        schema = {
            table: connection.execute(f"PRAGMA table_info({table})").fetchall()
            for table in tables
        }
        payload: dict[str, object] = {"tables": tables, "schema": schema}
        if "records" in tables:
            columns = {row[1] for row in schema["records"]}
            payload["row_count"] = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            for column in (
                "classification",
                "error_family",
                "easy_family",
                "repair_lane",
                "error_head",
            ):
                if column in columns:
                    payload[f"top_{column}"] = connection.execute(
                        f"SELECT {column}, COUNT(*) AS n FROM records "
                        f"GROUP BY {column} ORDER BY n DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
        _print(payload)


def inspect_batch(path: Path, limit: int, examples: int, diagnostic_contains: str) -> None:
    result_path = path / "verification_results.jsonl"
    rows = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = {
        str(row.get("record_id") or ""): row
        for row in (
            json.loads(line)
            for line in (path / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    diagnostics: Counter[str] = Counter()
    failure_examples: list[dict[str, object]] = []
    for row in rows:
        if row.get("success"):
            continue
        errors = row.get("errors") if isinstance(row.get("errors"), list) else []
        message = "\n".join(str(item) for item in errors)
        if "Unknown constant" in message:
            tail = message.split("Unknown constant", 1)[1].splitlines()[0].strip()
            key = f"unknown_constant:{tail}"
        elif "No goals to be solved" in message:
            key = "no_goals_to_be_solved"
        elif "Tactic `simp` failed" in message:
            key = "simp_failed"
        elif "declaration has metavariables" in message:
            key = "declaration_has_metavariables"
        elif "timeout" in message.casefold():
            key = "timeout"
        else:
            key = (message.splitlines()[0] if message else "empty_error")[:300]
        diagnostics[key] += 1
        if (
            len(failure_examples) < examples
            and (not diagnostic_contains or diagnostic_contains.casefold() in message.casefold())
        ):
            candidate = candidates.get(str(row.get("record_id") or ""), {})
            failure_examples.append({
                "record_id": row.get("record_id"),
                "diagnostic": message,
                "variants": candidate.get("variants") or [],
            })
    _print({
        "batch": str(path.resolve()),
        "result_rows": len(rows),
        "success_rows": sum(bool(row.get("success")) for row in rows),
        "failure_rows": sum(not row.get("success") for row in rows),
        "failure_diagnostics": diagnostics.most_common(limit),
        "failure_examples": failure_examples,
        "read_only": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("path", type=Path)
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("path", type=Path)
    batch_parser.add_argument("--examples", type=int, default=0)
    batch_parser.add_argument("--diagnostic-contains", default="")
    for subparser in (index_parser, batch_parser):
        subparser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.command == "index":
        inspect_index(args.path, args.limit)
    else:
        inspect_batch(
            args.path,
            args.limit,
            args.examples,
            args.diagnostic_contains,
        )


if __name__ == "__main__":
    main()
