"""Directed smoke for deterministic Verify dependency/Pantograph gates."""

from __future__ import annotations

import json

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


def _node(
    node_id: str,
    statement: str,
    *,
    dependencies: list[str] | None = None,
) -> BlueprintNode:
    return BlueprintNode(
        id=node_id,
        title=node_id,
        informal_statement="For every natural number n, n + 0 = n.",
        lean_statement=statement,
        preamble=LeanPreamble(
            imports=["Mathlib"],
            raw_header="import Mathlib\nopen Nat",
        ),
        semantic_alignment=SemanticAlignment(
            objects=["n is a natural number"],
            hypotheses=[],
            conclusion="n + 0 = n",
            alignment_notes="The Nat object and equality match exactly.",
        ),
        estimated_proof_length=ProofLengthEstimate(
            estimated_lines=1,
            estimated_tokens=8,
            rationale="Direct use of the predecessor.",
        ),
        depends_on=dependencies or [],
    )


def main() -> int:
    problem = TheoremProblem(
        problem_id="verify-deterministic-pantograph-smoke",
        imports=["Mathlib"],
        natural_language_statement="For every natural number n, n + 0 = n.",
        target_lean_decl="theorem target (n : Nat) : n + 0 = n",
        header="import Mathlib\nopen Nat",
    )
    backend = PantographDeclarationCheckingBackend(
        project_path="lean_project",
        timeout=120,
    )
    try:
        environment = backend.environment_identity(problem.imports)
        blueprint = Blueprint(
            blueprint_summary="The second node cites the exact first node.",
            nodes=[
                _node("L1", "lemma L1 (n : Nat) : n + 0 = n"),
                _node(
                    "L2",
                    "lemma L2 (n : Nat) : (L1 n) = (L1 n)",
                    dependencies=["L1"],
                ),
            ],
            root_dependencies=["L2"],
            environment=environment,
            problem_hash=problem.problem_hash,
        )
        result = BlueprintLeanChecker(backend).check_blueprint_detailed(
            problem,
            blueprint,
        )
        print(
            json.dumps(
                {
                    "success": result.success,
                    "nodes": [
                        {
                            "node_id": row.node_id,
                            "success": row.success,
                            "diagnostics": (
                                row.result.error_message
                                or row.result.stderr
                                or row.result.stdout
                            ),
                        }
                        for row in result.node_results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.success else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
