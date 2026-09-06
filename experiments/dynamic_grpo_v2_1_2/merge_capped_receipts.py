#!/usr/bin/env python3
"""Merge pass@8 plus capped top-ups into one exact attempt history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass8", type=Path, required=True)
    parser.add_argument("--pass16", type=Path, required=True)
    parser.add_argument("--pass32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path, offset, expected_per_problem in (
        (args.pass8, 0, 8),
        (args.pass16, 8, 8),
        (args.pass32, 16, 16),
    ):
        rows = read(path)
        counts: dict[str, int] = {}
        for source in rows:
            row = dict(source)
            problem_id = str(row["problem_id"])
            local_attempt = int(row["attempt_index"])
            if not 0 <= local_attempt < expected_per_problem:
                raise ValueError(f"{path}: invalid local attempt {local_attempt}")
            row["source_attempt_index"] = local_attempt
            row["attempt_index"] = offset + local_attempt
            row["capped_stage"] = {0: "pass8", 8: "pass16", 16: "pass32"}[offset]
            key = (problem_id, int(row["attempt_index"]))
            if key in seen:
                raise ValueError(f"duplicate merged attempt {key}")
            seen.add(key)
            counts[problem_id] = counts.get(problem_id, 0) + 1
            merged.append(row)
        if any(count != expected_per_problem for count in counts.values()):
            raise ValueError(f"{path}: incomplete per-problem attempt set")

    merged.sort(key=lambda row: (str(row["problem_id"]), int(row["attempt_index"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema": "adaptive_capped_receipts_merge_v1",
        "rows": len(merged),
        "problems": len({str(row["problem_id"]) for row in merged}),
        "pass8_sha256": sha(args.pass8),
        "pass16_sha256": sha(args.pass16),
        "pass32_sha256": sha(args.pass32),
        "output_sha256": sha(args.output),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
