"""Compile deterministic prepared reference proofs through the production assembler."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.preparation import compose_lean_theorem
from lean_prover.lean_training.verification.pool import (
    VerificationPoolConfig,
    run_verification_pool,
)
from lean_prover.lean_training.verification.schema import VerificationTask


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_rows(path: Path, count: int, rng: random.Random, excluded: set[str]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in read_jsonl(path)
        if row.get("lean_statement")
        and (row.get("proof") or row.get("completion"))
        and str(row.get("id")) not in excluded
    ]
    selected = rng.sample(candidates, count)
    excluded.update(str(row.get("id")) for row in selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--project", default="lean_project")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    excluded: set[str] = set()
    groups = {
        "train": select_rows(Path(args.train), 10, rng, excluded),
        "eval": select_rows(Path(args.eval), 10, rng, excluded),
        "train_extra": select_rows(Path(args.train), 10, rng, excluded),
    }
    tasks: list[VerificationTask] = []
    selected: list[dict[str, Any]] = []
    for problem_index, (group, row) in enumerate(
        (item for group, rows in groups.items() for item in ((group, row) for row in rows))
    ):
        proof = str(row.get("proof") or row.get("completion") or "").strip()
        statement = str(row["lean_statement"]).strip()
        assembled = compose_lean_theorem(statement, proof)
        imports = tuple(row.get("imports") or ("Mathlib",))
        context_lines = tuple(row.get("context_lines") or [])
        selected.append(
            {
                "group": group,
                "id": row.get("id"),
                "statement": statement,
                "reference_proof": proof,
                "assembled_source": assembled,
                "imports": list(imports),
                "context_lines": list(context_lines),
            }
        )
        tasks.append(
            VerificationTask(
                priority=problem_index,
                problem_index=problem_index,
                attempt_index=0,
                problem_id=str(row.get("id") or problem_index),
                prompt=str(row.get("prompt") or ""),
                generated_proof=proof,
                raw_completion=proof,
                lean_code=assembled,
                imports=imports,
                context_lines=context_lines,
                reject_forbidden=True,
            )
        )

    run = run_verification_pool(
        tasks,
        VerificationPoolConfig(
            lean_project_path=args.project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.startup_timeout,
            num_workers=1,
            queue_maxsize=32,
            heartbeat_interval=5,
            heartbeat_timeout=30,
            max_worker_restarts=1,
            max_task_retries=0,
            shutdown_timeout=15,
        ),
    )
    by_problem = {str(row["problem_id"]): row for row in run.results}
    for row in selected:
        result = by_problem.get(str(row["id"]), {})
        row.update(
            {
                "success": bool(result.get("success")),
                "diagnostics": result.get("diagnostics"),
                "compile_errors": result.get("compile_errors") or [],
            }
        )
    report = {
        "sample_plan": {name: len(rows) for name, rows in groups.items()},
        "benchmark_reference_note": (
            "The local miniF2F files contain sorry placeholders and no formal reference proofs; "
            "ten disjoint additional train references are used instead."
        ),
        "passed": sum(bool(row["success"]) for row in selected),
        "total": len(selected),
        "warmup_reports": run.warmup_reports,
        "fatal_errors": run.fatal_errors,
        "records": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
