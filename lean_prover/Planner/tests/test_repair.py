from typing import Any

from lean_prover.Planner.repair import BlueprintRepairer
from lean_prover.Planner.schemas import (
    Blueprint,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.tests.helpers import environment


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history=None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "empty_response_message": empty_response_message,
                "history": history,
            }
        )
        return self.response


def test_repair_returns_validated_blueprint_from_client_response() -> None:
    client = FakeClient(
        {
            "blueprint_summary": "fixed",
            "nodes": [],
            "root_dependencies": [],
            "environment": {
                "lean_version": "Lean test",
                "lean_commit": "f72c35b3f637c8c6571d353742168ab66cc22c00",
                "mathlib_commit": "5e932f97dd25535344f80f9dd8da3aab83df0fe6",
                "environment_hash": "4" * 64,
            },
        }
    )
    repairer = BlueprintRepairer(client)
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    issue = ValidationIssue(
        stage="graph",
        code="invalid_graph",
        message="missing node",
    )

    blueprint = repairer.repair(
        problem=problem,
        previous_blueprint={
            "blueprint_summary": "broken",
            "nodes": [],
            "root_dependencies": [],
            "environment": environment().model_dump(mode="json"),
        },
        issues=[issue],
        environment=environment(),
    )

    assert isinstance(blueprint, Blueprint)
    assert blueprint.blueprint_summary == "fixed"
    assert "missing node" in client.calls[0]["user_prompt"]
    assert client.calls[0]["empty_response_message"] == "修复分解模型空响应"
    assert client.calls[0]["history"]
    assert "missing node" in str(client.calls[0]["history"])
