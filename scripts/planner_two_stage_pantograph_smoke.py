#!/usr/bin/env python3
"""Real local Pantograph smoke for the two-stage Planner statement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanPreamble,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)


def node(node_id: str, lean_statement: str, *, depends_on: list[str] | None = None) -> BlueprintNode:
    return BlueprintNode(
        id=node_id,
        title=f"Statement {node_id}",
        informal_statement=(
            "For every natural number n, n equals itself."
            if node_id == "L1"
            else "For every natural number n, adding zero to n gives n."
        ),
        lean_statement=lean_statement,
        preamble=LeanPreamble(
            imports=["Mathlib"],
            namespaces=["PlannerSmoke"],
            open_namespaces=["Nat"],
            open_scoped=["BigOperators"],
            variable_declarations=["variable (n : Nat)"],
            local_context=["noncomputable section"],
        ),
        semantic_alignment=SemanticAlignment(
            objects=["n, a natural number"],
            hypotheses=[],
            conclusion="the stated equality holds",
            alignment_notes="The Lean equality has the same object and conclusion.",
        ),
        estimated_proof_length=ProofLengthEstimate(
            estimated_lines=2,
            estimated_tokens=8,
            rationale="The equality should have a direct simplification proof.",
        ),
        depends_on=depends_on or [],
        proof_strategy="Use reflexivity or simplification.",
        difficulty=1,
    )


def blueprint(environment, *, failing: bool) -> Blueprint:
    return Blueprint(
        blueprint_summary="Check two elementary natural-number statements.",
        nodes=[
            node("L1", "lemma L1 : n = n"),
            node(
                "L2",
                "lemma L2 : MissingPlannerType" if failing else "lemma L2 : n + 0 = n",
                depends_on=["L1"],
            ),
        ],
        root_dependencies=["L2"],
        environment=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    problem = TheoremProblem(
        problem_id="planner_pantograph_smoke",
        imports=["Mathlib"],
        natural_language_statement="For every natural number n, n + 0 = n.",
        target_lean_decl="theorem target (n : Nat) : n + 0 = n",
    )
    with PantographDeclarationCheckingBackend(
        project_path=args.project,
        timeout=args.timeout,
    ) as backend:
        checker = BlueprintLeanChecker(backend)
        environment = checker.environment_identity(problem)
        passing = checker.check_blueprint_detailed(
            problem,
            blueprint(environment, failing=False),
        )
        failing = checker.check_blueprint_detailed(
            problem,
            blueprint(environment, failing=True),
        )
    if not passing.success or len(passing.node_results) != 2:
        raise RuntimeError(f"passing Blueprint failed: {passing.model_dump()}")
    if failing.success or failing.failed_node_ids != ["L2"]:
        raise RuntimeError(f"atomic failure gate failed: {failing.model_dump()}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "environment": environment.model_dump(mode="json"),
                "passing_node_results": [
                    {"node_id": row.node_id, "success": row.success}
                    for row in passing.node_results
                ],
                "failing_node_results": [
                    {"node_id": row.node_id, "success": row.success}
                    for row in failing.node_results
                ],
                "atomic_failed_node_ids": failing.failed_node_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
