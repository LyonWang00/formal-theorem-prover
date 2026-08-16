"""Recompile all verified anchor proofs to collect current Pantograph timings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lean_prover.lean_training.data.preparation import compose_lean_theorem
from lean_prover.lean_training.verification.pool import VerificationPoolConfig, run_verification_pool
from lean_prover.lean_training.verification.schema import VerificationTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", default="lean_project")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = []
    for index, row in enumerate(rows):
        proof = str(row.get("proof") or row.get("completion") or "").strip()
        statement = str(row.get("lean_statement") or row.get("statement") or "").strip()
        tasks.append(VerificationTask(priority=index, problem_index=index, attempt_index=0,
            problem_id=str(row.get("record_id") or row.get("id") or index), prompt=str(row.get("prompt") or ""),
            generated_proof=proof, raw_completion=proof, lean_code=compose_lean_theorem(statement, proof),
            imports=tuple(row.get("imports") or ("Mathlib",)), context_lines=tuple(row.get("context_lines") or ()),
            enqueue_time=time.monotonic(), payload={"statement_id": row.get("statement_id"), "record_id": row.get("record_id") or row.get("id")}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle = args.output.open("w", encoding="utf-8")
    def record(result):
        handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
    try:
        run = run_verification_pool(tasks, VerificationPoolConfig(
            lean_project_path=args.project, imports=("Mathlib",), timeout=args.timeout, warmup_timeout=900,
            num_workers=args.workers, queue_maxsize=32, heartbeat_interval=5, heartbeat_timeout=30,
            max_worker_restarts=3, max_task_retries=1, shutdown_timeout=15,
            task_spool_dir=str(args.output.parent / "anchor_compile_tasks"), save_full_source_on_failure_only=True), on_result=record)
    finally:
        handle.close()
    summary = {"records": len(rows), "results": len(run.results),
        "successes": sum(bool(row.get("success")) for row in run.results),
        "failures": sum(not bool(row.get("success")) for row in run.results),
        "warmup_reports": run.warmup_reports, "fatal_errors": run.fatal_errors, "runtime_stats": run.runtime_stats}
    (args.output.parent / "anchor_compile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["successes"] == len(rows) else 1)


if __name__ == "__main__": main()
