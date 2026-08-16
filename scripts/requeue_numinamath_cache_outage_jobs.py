#!/usr/bin/env python3
"""Requeue jobs failed only because the frozen Mathlib build cache was unavailable."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_ERROR = "PantographUnavailableError: failed to start Pantograph"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--min-failed-unix", type=int, required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    failed_dir = args.queue_dir / "failed"
    pending_dir = args.queue_dir / "pending"
    archive_dir = args.queue_dir / "failed_archive" / args.archive_name
    selected = []
    for error_path in sorted(failed_dir.glob("*.error.json")):
        error = json.loads(error_path.read_text(encoding="utf-8"))
        failed_unix = int(error.get("failed_unix") or 0)
        if failed_unix < args.min_failed_unix or EXPECTED_ERROR not in str(error.get("error") or ""):
            continue
        job_path = error_path.with_name(error_path.name.removesuffix(".error.json") + ".json")
        if not job_path.is_file():
            raise FileNotFoundError(f"paired job descriptor missing: {job_path}")
        job = json.loads(job_path.read_text(encoding="utf-8"))
        batch_dir = Path(str(job["batch_dir"]))
        manifest = batch_dir / "candidate_manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"candidate manifest missing: {manifest}")
        target = pending_dir / job_path.name
        if target.exists():
            raise FileExistsError(f"pending target already exists: {target}")
        selected.append({
            "descriptor": job_path.name,
            "batch_id": job.get("batch_id"),
            "failed_unix": failed_unix,
            "error_file": error_path.name,
        })

    report = {
        "schema_version": "numinamath_cache_outage_requeue_v1",
        "min_failed_unix": args.min_failed_unix,
        "selected_jobs": len(selected),
        "jobs": selected,
        "executed": args.execute,
    }
    if args.execute:
        pending_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=False)
        for item in selected:
            job_path = failed_dir / str(item["descriptor"])
            error_path = failed_dir / str(item["error_file"])
            shutil.move(str(error_path), str(archive_dir / error_path.name))
            shutil.move(str(job_path), str(pending_dir / job_path.name))
        report["archive_dir"] = str(archive_dir)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
