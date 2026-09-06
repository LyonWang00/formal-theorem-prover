#!/usr/bin/env python3
"""Recompile a fixed verified-data golden set and publish it only on 30/30."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lean_prover.lean_training.runtime.environment import environment_identity
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from scripts.curate_reference_proofs import make_task, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fixture-output", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    groups = Counter(str(row.get("group")) for row in rows)
    proofs = [str(row.get("proof") or row.get("completion") or "") for row in rows]
    statements = [str(row.get("lean_statement") or "") for row in rows]
    shape_ok = (
        len(rows) == 30
        and groups == {"train": 10, "eval": 10, "verified_extra": 10}
        and len(set(statements)) == 30
        and len(set(proofs)) == 30
    )
    tasks = []
    for index, row in enumerate(rows):
        task, _ = make_task(
            row,
            source_file=input_path,
            source_index=index,
            problem_index=index,
            generation_id=f"golden-roundtrip:{index}",
            group=str(row.get("group")),
        )
        tasks.append(task)

    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.startup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=8,
            max_worker_restarts=3,
            max_task_retries=1,
            shutdown_timeout=15,
        )
    )
    try:
        run = pool.run_batch(tasks)
        passed = sum(bool(result.get("success")) for result in run.results)
        report = {
            "success": shape_ok and passed == 30 and not run.fatal_errors,
            "passed": passed,
            "total": len(run.results),
            "shape_ok": shape_ok,
            "groups": dict(groups),
            "unique_statements": len(set(statements)),
            "unique_proofs": len(set(proofs)),
            "coverage": {
                "simp": sum("simp" in proof for proof in proofs),
                "norm_num": sum("norm_num" in proof for proof in proofs),
                "linarith": sum("linarith" in proof for proof in proofs),
                "multi_step": sum(
                    len(
                        [
                            line
                            for line in proof.splitlines()
                            if line.strip() and line.strip() != "by"
                        ]
                    )
                    > 1
                    for proof in proofs
                ),
            },
            "environment": environment_identity(args.project, ("Mathlib",)),
            "runtime_stats": run.runtime_stats,
            "fatal_errors": run.fatal_errors,
            "results": run.results,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if report["success"]:
            fixture_path = Path(args.fixture_output)
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "results"},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(0 if report["success"] else 1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
