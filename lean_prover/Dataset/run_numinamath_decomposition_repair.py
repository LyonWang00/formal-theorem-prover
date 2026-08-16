"""Run a bounded Planner -> Blueprint -> Prover pilot for routed NuminaMath rows.

This module is deliberately staged and resumable.  It never mutates the
canonical success/fail files: only an all-node Pantograph-verified Prover result
is eligible for a later campaign commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.repair_numinamath_failures import iter_jsonl, write_jsonl
from lean_prover.Planner.client import create_deepseek_client_from_env
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import PantographDeclarationCheckingBackend
from lean_prover.Planner.schemas import RawTheoremInput
from lean_prover.Planner.service import PlannerService
from lean_prover.Prover.service import (
    BlueprintProver,
    NodeProofVerifier,
    create_openai_compatible_proof_generator_from_env,
)
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


SCHEMA = "numinamath_decomposition_repair_v1"


def _ids(paths: Iterable[Path]) -> set[str]:
    return {
        str(row["record_id"])
        for path in paths
        for row in iter_jsonl(path)
        if row.get("record_id")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen = _ids(args.frozen_manifest)
    routes = [
        row for row in iter_jsonl(args.routing_manifest)
        if str(row["record_id"]) not in frozen
        and "missing_proof_high_statement_complexity" in row.get("reason_codes", [])
    ]
    routes.sort(key=lambda row: (int(row.get("statement_chars") or 0), str(row["record_id"])))
    existing = {
        str(row["record_id"]): row for row in (
            iter_jsonl(args.output) if args.output.exists() else []
        )
    }
    selected = [row for row in routes if str(row["record_id"]) not in existing]
    if args.limit:
        selected = selected[: args.limit]
    selected_ids = {str(row["record_id"]) for row in selected}
    source_rows = {
        str(row["record_id"]): row
        for row in iter_jsonl(args.fail_file)
        if str(row["record_id"]) in selected_ids
    }
    if set(source_rows) != selected_ids:
        raise RuntimeError("selected decomposition rows are not all present in current fail")

    client = create_deepseek_client_from_env()
    for route in selected:
        record_id = str(route["record_id"])
        row = source_rows[record_id]
        statement = str(row.get("lean_statement") or row.get("formal_statement") or "").strip()
        raw_input = RawTheoremInput(
            input_text=statement,
            problem_id=record_id,
            imports=["Mathlib"],
        )
        with PantographDeclarationCheckingBackend(
            project_path=args.lean_project,
            timeout=args.timeout,
        ) as planner_backend:
            planner = PlannerService(
                client=client,
                lean_checker=BlueprintLeanChecker(planner_backend),
                max_attempts=args.planner_attempts,
            )
            planner_result = planner.plan_input(raw_input)

        entry: dict[str, Any] = {
            "schema_version": SCHEMA,
            "record_id": record_id,
            "record_hash": row.get("record_hash"),
            "routing_hash": route.get("routing_hash"),
            "planner_success": bool(planner_result.success),
            "planner_stage": planner_result.stage,
            "planner_result": planner_result.model_dump(mode="json", by_alias=True),
            "prover_success": False,
            "repair_status": "planner_failure",
        }
        if planner_result.success and planner_result.problem and planner_result.blueprint:
            theorem_verifier = PantographTheoremVerifier(
                args.lean_project,
                imports=("Mathlib",),
                timeout=args.timeout,
                startup_timeout=900,
            )
            try:
                prover = BlueprintProver(
                    generator=create_openai_compatible_proof_generator_from_env(),
                    verifier=NodeProofVerifier(theorem_verifier),
                )
                prover_result = prover.prove(
                    problem=planner_result.problem,
                    blueprint=planner_result.blueprint,
                )
                entry["prover_success"] = bool(prover_result.success)
                entry["repair_status"] = (
                    "prover_success" if prover_result.success else "prover_failure"
                )
                entry["prover_result"] = prover_result.model_dump(mode="json")
            finally:
                theorem_verifier.close()
        existing[record_id] = entry
        write_jsonl(args.output, existing.values())

    counts: dict[str, int] = {}
    for row in existing.values():
        status = str(row.get("repair_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    report = {
        "schema_version": SCHEMA,
        "selected_rows": len(selected),
        "completed_rows": len(existing),
        "status_counts": counts,
        "frozen_rows": len(frozen),
        "canonical_dataset_mutated": False,
        "all_successes_require_all_node_pantograph_verification": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fail-file", type=Path, default=root / "verified_data" / "numinamath_verified_fail.jsonl")
    value.add_argument("--routing-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    value.add_argument("--limit", type=int, default=1)
    value.add_argument("--timeout", type=int, default=120)
    value.add_argument("--planner-attempts", type=int, default=1)
    value.add_argument("--frozen-manifest", type=Path, action="append", default=[])
    return value


def main() -> None:
    report = run(parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
