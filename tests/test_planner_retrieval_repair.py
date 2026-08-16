from typing import Any

from lean_prover.Planner.schemas import (
    Blueprint,
    LeanCheckResult,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.tests.helpers import environment, full_node
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeChecker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def check_blueprint_detailed(self, problem, blueprint):
        from lean_prover.Planner.schemas import (
            BlueprintLeanCheckResult,
            NodeLeanCheckResult,
        )

        statement = blueprint.nodes[0].lean_decl
        self.calls.append(statement)
        success = "True" in statement
        result = LeanCheckResult(
            success=success,
            declaration=statement,
            error_message=None if success else "unknown identifier Bad",
        )
        return BlueprintLeanCheckResult(
            success=success,
            node_results=[
                NodeLeanCheckResult(
                    node_id="L1",
                    success=success,
                    lean_statement=statement,
                    preamble=blueprint.nodes[0].preamble,
                    result=result,
                )
            ],
            failed_node_ids=[] if success else ["L1"],
            issues=[] if success else [
                ValidationIssue(
                    stage="lean",
                    code="declaration_failed",
                    node_id="L1",
                    message="unknown identifier Bad",
                )
            ],
        )


def _patch(statement: str) -> dict[str, Any]:
    node = full_node("L1", statement)
    return {
        "node_id": "L1",
        "lean_statement": statement,
        "preamble": node.preamble.model_dump(mode="json"),
        "semantic_alignment": node.semantic_alignment.model_dump(mode="json"),
        "mathlib_hints": [],
    }


def test_repair_generates_exactly_one_candidate_and_compiles_it() -> None:
    previous = Blueprint(
        blueprint_summary="one",
        nodes=[full_node("L1", "lemma L1 : Bad")],
        root_dependencies=["L1"],
        environment=environment(),
    )
    client = FakeClient(
        {
            "candidates": [
                {
                    "candidate_id": "C1",
                    "node_patches": [_patch("lemma L1 : True")],
                    "rationale": "single grounded candidate",
                },
            ]
        }
    )
    checker = FakeChecker()
    repaired = PlannerSubproblemRepairer(
        client,
        lean_checker=checker,
    ).repair(
        problem=TheoremProblem(
            problem_id="p",
            imports=["Mathlib"],
            natural_language_statement="Truth.",
            target_lean_decl="theorem target : True",
        ),
        previous_blueprint=previous,
        issues=[
            ValidationIssue(
                stage="lean",
                code="declaration_failed",
                node_id="L1",
                message="unknown identifier Bad",
            )
        ],
        environment=environment(),
    )
    assert checker.calls == ["lemma L1 : True"]
    assert repaired.nodes[0].lean_decl == "lemma L1 : True"
    assert repaired.nodes[0].formal_statement_verified is None
    audit = repaired.metadata["formalization_repair"][-1]
    assert [row["candidate_id"] for row in audit["candidate_compilation"]] == ["C1"]
    prompt = client.calls[0]["user_prompt"]
    assert environment().mathlib_commit in prompt
    assert "Original root natural-language statement" in prompt
    assert "Full Pantograph/Verify/static errors" in prompt
    assert '"candidates"' in prompt
