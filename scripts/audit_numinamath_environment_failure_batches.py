#!/usr/bin/env python3
"""Read-only audit for staged batches affected by a missing Mathlib cache."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-batch", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for batch_dir in sorted(args.root.glob("batch_*")):
        match = re.match(r"batch_(\d+)", batch_dir.name)
        if not match or int(match.group(1)) < args.min_batch:
            continue
        candidates = read_jsonl(batch_dir / "candidate_manifest.jsonl")
        results = read_jsonl(batch_dir / "verification_results.jsonl")
        environment_failures = 0
        successes = 0
        record_ids: set[str] = set()
        for result in results:
            record_ids.add(str(result.get("record_id") or ""))
            successes += int(bool(result.get("success")))
            diagnostic = "\n".join(str(item) for item in (result.get("errors") or []))
            environment_failures += int("unknown module prefix 'Mathlib'" in diagnostic)
        rows.append({
            "batch_id": batch_dir.name,
            "candidate_rows": len(candidates),
            "result_rows": len(results),
            "result_records": len(record_ids - {""}),
            "success_rows": successes,
            "mathlib_environment_failure_rows": environment_failures,
            "affected": bool(results) and environment_failures == len(results),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
