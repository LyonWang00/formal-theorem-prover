#!/usr/bin/env python3
"""Run the three fixed proofs through two persistent workers, then close all IPC."""

from __future__ import annotations

import argparse
import json

from lean_prover.lean_training.expert_iteration.precheck import (
    build_fixed_smoke_tasks,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean_project_path", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--warmup_timeout", type=int, default=1200)
    args = parser.parse_args()
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.lean_project_path,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.warmup_timeout,
            num_workers=args.num_workers,
            queue_maxsize=8,
            shutdown_timeout=15,
        )
    )
    try:
        run = pool.run_batch(build_fixed_smoke_tasks(("Mathlib",)))
        report = {
            "success": not run.fatal_errors
            and len(run.results) == 3
            and all(row.get("success") for row in run.results),
            "results": run.results,
            "warmup_reports": run.warmup_reports,
            "runtime_stats": run.runtime_stats,
            "fatal_errors": run.fatal_errors,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["success"]:
            raise SystemExit(1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
