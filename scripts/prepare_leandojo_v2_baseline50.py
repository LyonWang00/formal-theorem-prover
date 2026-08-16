"""Prepare and compare the deterministic 50-record Pantograph baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(value) for value in row.get("imports") or ())].append(row)
    selected: list[dict[str, Any]] = []
    for _, group in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        take = min(len(group), count - len(selected))
        selected.extend(group[:take])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)} of {count} rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--compare-results", type=Path)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.sample))
    previous = {
        str(row["sample_id"]): row for row in iter_jsonl(args.previous_results)
    }
    selected = choose_rows(rows, args.count)
    missing = [str(row["id"]) for row in selected if str(row["id"]) not in previous]
    if missing:
        raise RuntimeError(f"baseline ids missing previous results: {missing[:5]}")

    output_root = args.output_root.resolve()
    manifest = output_root / "audit/baseline50_manifest.jsonl"
    expected = output_root / "audit/baseline50_expected.jsonl"
    write_jsonl(manifest, selected)
    write_jsonl(
        expected,
        (
            {
                "sample_id": str(row["id"]),
                "compile_success": bool(previous[str(row["id"])]["compile_success"]),
                "error_category": str(
                    previous[str(row["id"])].get("error_category") or ""
                ),
                "assembled_source_hash": str(
                    previous[str(row["id"])]["assembled_source_hash"]
                ),
            }
            for row in selected
        ),
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "count": len(selected),
        "selection": "largest import groups, deterministic lexical tie-break",
        "import_group_count": len(
            {
                tuple(str(value) for value in row.get("imports") or ())
                for row in selected
            }
        ),
        "manifest_path": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "expected_path": str(expected),
        "all_selected_ids_present_in_previous_results": not missing,
        "reproduced": None,
    }
    if args.compare_results:
        actual = {
            str(row["sample_id"]): row
            for row in iter_jsonl(args.compare_results)
        }
        mismatches: list[dict[str, Any]] = []
        cache_hits = 0
        for row in iter_jsonl(expected):
            sample_id = str(row["sample_id"])
            got = actual.get(sample_id)
            if got is None:
                mismatches.append(
                    {"sample_id": sample_id, "reason": "missing_result"}
                )
                continue
            cache_hits += int(bool(got.get("cache_hit")))
            for field in (
                "compile_success",
                "error_category",
                "assembled_source_hash",
            ):
                if got.get(field) != row.get(field):
                    mismatches.append(
                        {
                            "sample_id": sample_id,
                            "field": field,
                            "expected": row.get(field),
                            "actual": got.get(field),
                        }
                    )
        payload.update(
            {
                "actual_results_path": str(args.compare_results.resolve()),
                "actual_result_count": len(actual),
                "first_run_cache_hits": cache_hits,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "reproduced": len(actual) == args.count and not mismatches,
            }
        )
    write_json(output_root / "audit/reproduction.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
