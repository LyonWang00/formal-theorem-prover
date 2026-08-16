"""Summarize runtime telemetry and durable expert-iteration artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def summarize(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    memory = read_jsonl(root / "runtime/memory.jsonl")
    lifecycle = read_jsonl(root / "runtime/lifecycle.jsonl")
    isolated = read_jsonl(root / "runtime/isolated_processes.jsonl")
    generations = read_jsonl(root / "iteration_000/discovery/generations.jsonl")
    verifications = read_jsonl(root / "iteration_000/discovery/verifications.jsonl")
    cache_rows = 0
    cache_path = root / "verification_cache.sqlite"
    if cache_path.exists():
        with sqlite3.connect(cache_path) as connection:
            cache_rows = int(
                connection.execute(
                    "SELECT COUNT(*) FROM verification_cache"
                ).fetchone()[0]
            )
    lifecycle_names = (
        "vllm_start",
        "vllm_stop",
        "pantograph_start",
        "pantograph_stop",
        "trainer_start",
        "trainer_stop",
        "cleanup_all",
    )
    return {
        "peak_system_used_ratio": max(
            (row.get("system_used_ratio", 0) for row in memory), default=0
        ),
        "peak_process_rss_bytes": max(
            (row.get("process_rss_bytes", 0) for row in memory), default=0
        ),
        "peak_gpu_used_mib": max(
            (
                gpu.get("used_mib", 0)
                for row in memory
                for gpu in row.get("gpu", [])
            ),
            default=0,
        ),
        "memory_samples": len(memory),
        "memory_levels": {
            level: sum(row.get("level") == level for row in memory)
            for level in ("normal", "warning", "critical")
        },
        "lifecycle_counts": {
            event: sum(row.get("event") == event for row in lifecycle)
            for event in lifecycle_names
        },
        "generation_process_starts": sum(
            row.get("event") == "started" and row.get("stage") == "generation"
            for row in isolated
        ),
        "generations": len(generations),
        "unique_generation_ids": len(
            {row["generation_id"] for row in generations}
        ),
        "verifications": len(verifications),
        "unique_verification_generation_ids": len(
            {row["generation_id"] for row in verifications}
        ),
        "verification_successes": sum(
            bool(row.get("verified")) for row in verifications
        ),
        "success_rows_with_full_source": sum(
            bool(row.get("assembled_source"))
            for row in verifications
            if row.get("verified")
        ),
        "failure_rows_with_full_source": sum(
            bool(row.get("assembled_source"))
            for row in verifications
            if not row.get("verified")
        ),
        "sqlite_cache_rows": cache_rows,
        "retained_failure_task_files": len(
            list((root / "runtime/verification_tasks").glob("*.json"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
