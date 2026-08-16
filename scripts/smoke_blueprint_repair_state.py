"""Run one real, auditable BluePrintRepair state-contract smoke test."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from lean_prover.Planner.client import create_deepseek_client_from_env
from lean_prover.Planner.schemas import LeanEnvironmentIdentity, TheoremProblem
from lean_prover.Repair.blueprint import (
    BluePrintRepairer,
    JsonlBlueprintRepairStore,
    deterministic_blueprint_issues,
)


def first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-results", type=Path, required=True)
    parser.add_argument("--repair-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    problem_row = first_jsonl(args.problem_results)
    repair_row = first_jsonl(args.repair_results)
    planner = problem_row["pipeline_result"]["planner_result"]
    problem = TheoremProblem.model_validate(planner["problem"])
    environment = LeanEnvironmentIdentity.model_validate(planner["environment"])
    candidate = copy.deepcopy(repair_row["input_snapshot"])
    if not isinstance(candidate, dict) or not candidate.get("nodes"):
        raise ValueError("source repair row must contain a nonempty input snapshot")

    original_root_dependencies = list(candidate["root_dependencies"])
    candidate["root_dependencies"] = []
    repaired, rounds = BluePrintRepairer(
        create_deepseek_client_from_env(),
        store=JsonlBlueprintRepairStore(args.output),
    ).run(
        problem=problem,
        environment=environment,
        candidate=candidate,
    )
    report = {
        "problem_id": problem.problem_id,
        "injected_defect": "root_dependencies cleared",
        "original_root_dependencies": original_root_dependencies,
        "repaired_root_dependencies": repaired.root_dependencies,
        "rounds": [
            {
                "round_id": item.round_id,
                "state": item.state.value if item.state else None,
                "changed": item.changed,
                "changed_fields": item.changed_fields,
                "output_error": item.output_error,
            }
            for item in rounds
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    _, final_issues = deterministic_blueprint_issues(
        problem=problem,
        environment=environment,
        candidate=repaired,
    )
    if final_issues:
        raise RuntimeError("model did not produce a valid repaired DAG")
    if not rounds or rounds[0].state is None:
        raise RuntimeError("first repair response was not structurally valid")
    if rounds[0].state.value != "failed" or not rounds[0].changed:
        raise RuntimeError("first round did not execute a failed-state repair")
    if not any(
        path.startswith(("root_dependencies", "nodes"))
        for path in rounds[0].changed_fields
    ):
        raise RuntimeError("DAG repair was not audit-visible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
