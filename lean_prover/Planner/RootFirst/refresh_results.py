#!/usr/bin/env python3
"""Upgrade saved RootFirst records without making any API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_minif2f import load_jsonl, refresh_result_record, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = [
        refresh_result_record(row)
        for row in load_jsonl(args.output_dir / "results.jsonl")
    ]
    upgraded = args.output_dir / "results_v2.jsonl"
    upgraded.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary = summarize(rows)
    invariant_errors: list[object] = []
    for row in rows:
        if int(row["nodes_added"]) > 2:
            invariant_errors.append(
                [row["problem_id"], "nodes_added", row["nodes_added"]]
            )
        if int(row["frozen_reprove_violation_count"]):
            invariant_errors.append(
                [
                    row["problem_id"],
                    "frozen_reprove_violation_count",
                    row["frozen_reprove_violation_count"],
                ]
            )
        for node in row["result"]["blueprint"]["nodes"]:
            grouped: dict[tuple[int, int], int] = {}
            for attempt in node["attempts"]:
                if attempt["kind"] != "proof_repair":
                    continue
                key = (
                    int(attempt["proof_invocation"]),
                    int(attempt["attempt_index"]),
                )
                grouped[key] = grouped.get(key, 0) + 1
            for key, count in grouped.items():
                if count > 3:
                    invariant_errors.append(
                        [row["problem_id"], node["id"], list(key), count]
                    )
    summary["invariant_errors"] = invariant_errors
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
