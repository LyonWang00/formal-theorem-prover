from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    BlueprintPlan,
    BlueprintPlanNode,
    LeanPreamble,
    ProofLengthEstimate,
    RawTheoremInput,
    SemanticAlignment,
)
from lean_prover.Planner.service import PlannerService


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="real Pantograph integration runs in the WSL Lean project",
)


class SequenceClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history=None,
    ):
        if "You are BluePrintRepair" in system_prompt:
            import json

            problem = json.loads(
                user_prompt.split("Exact root problem:\n", 1)[1].split(
                    "\n\nExact local environment:", 1
                )[0]
            )
            candidate = json.loads(
                user_prompt.split("BEGIN_CURRENT_BLUEPRINT_CANDIDATE\n", 1)[1]
                .split("\nEND_CURRENT_BLUEPRINT_CANDIDATE", 1)[0]
            )
            normalized = Blueprint.model_validate(candidate)
            normalized.problem_hash = problem["problem_hash"]
            complete = normalized.model_dump(mode="json", by_alias=True)
            return {
                **complete,
                "state": "success" if candidate == complete else "failed",
            }
        if "compare one natural-language mathematical" in system_prompt:
            return {
                "formal_statement_correct": True,
                "corrected_lean_statement": "",
                "issue_message": "",
                "repair_reference": "",
                "verification_notes": "Exact fixture alignment.",
            }
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "empty_response_message": empty_response_message,
                "history": history,
            }
        )
        return self.responses.pop(0)


def estimate() -> ProofLengthEstimate:
    return ProofLengthEstimate(
        estimated_lines=2,
        estimated_tokens=8,
        rationale="A direct proof is expected.",
    )


def node_payload(environment) -> dict[str, Any]:
    node = BlueprintNode(
        id="L1",
        title="Truth",
        informal_statement="Truth holds.",
        informal_proof="Use the constructor of True.",
        logical_ideas=["Construct True"],
        lean_statement="lemma L1 : True",
        preamble=LeanPreamble(imports=["Mathlib"]),
        semantic_alignment=SemanticAlignment(
            objects=["True"],
            hypotheses=[],
            conclusion="True",
            alignment_notes="The Lean proposition is exactly True.",
        ),
        estimated_proof_length=estimate(),
        proof_strategy="Use the constructor of True.",
        difficulty=1,
    )
    return {
        "blueprint_summary": "Establish truth directly.",
        "nodes": [node.model_dump(mode="json", by_alias=True)],
        "root_dependencies": ["L1"],
        "environment": environment.model_dump(mode="json"),
        "warnings": [],
    }


def test_both_planner_modes_pass_real_pantograph() -> None:
    pytest.importorskip("pantograph")
    project = Path.cwd() / "lean_project"
    if not (project / "lean-toolchain").is_file():
        pytest.skip("Lean project toolchain is unavailable")

    with PantographDeclarationCheckingBackend(project_path=project) as backend:
        environment = backend.environment_identity(["Mathlib"])
        checker = BlueprintLeanChecker(backend)

        plan = BlueprintPlan(
            blueprint_summary="Establish truth directly.",
            nodes=[
                BlueprintPlanNode(
                    id="L1",
                    title="Truth",
                    informal_statement="Truth holds.",
                    informal_proof="Use the constructor of True.",
                    logical_ideas=["Construct True"],
                    lean_statement="",
                    proof_strategy="Use the constructor of truth.",
                    estimated_proof_length=estimate(),
                    difficulty=1,
                )
            ],
            root_dependencies=["L1"],
        )
        natural_blueprint = node_payload(environment)
        natural_blueprint["nodes"][0]["proof_strategy"] = (
            "Use the constructor of truth."
        )
        natural_client = SequenceClient(
            [
                {
                    "input_kind": "natural_language",
                    "confidence": 1.0,
                    "rationale": "The input is ordinary mathematical prose.",
                },
                plan.model_dump(mode="json"),
                {
                    "target_lean_decl": "theorem target : True",
                    "semantic_alignment_notes": "Truth is represented by True.",
                    "warnings": [],
                },
                natural_blueprint,
            ]
        )
        natural = PlannerService(
            client=natural_client,
            lean_checker=checker,
            max_attempts=1,
            environment=environment,
        ).plan_input(RawTheoremInput(input_text="Prove that truth holds."))

        lean_client = SequenceClient(
            [
                {
                    "input_kind": "lean",
                    "confidence": 1.0,
                    "rationale": "The input is a Lean theorem declaration.",
                },
                node_payload(environment),
            ]
        )
        lean = PlannerService(
            client=lean_client,
            lean_checker=checker,
            max_attempts=1,
            environment=environment,
        ).plan_input(RawTheoremInput(input_text="theorem supplied : True"))

    assert natural.success
    assert natural.planner_mode and natural.planner_mode.value == "natural_language"
    assert natural.target_check and natural.target_check.success
    assert all(check.success for check in natural.node_checks)
    assert lean.success
    assert lean.planner_mode and lean.planner_mode.value == "lean"
    assert lean.plan is None
    assert lean.translation_attempts == 0
    assert lean.target_check and lean.target_check.success
    assert all(check.success for check in lean.node_checks)
    assert len(lean_client.calls) == 2
