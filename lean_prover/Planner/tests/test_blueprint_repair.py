from __future__ import annotations

import json

import pytest

from lean_prover.Planner.schemas import Blueprint, TheoremProblem
from lean_prover.Planner.tests.helpers import environment, full_node
from lean_prover.Repair.blueprint import BluePrintRepairer
from lean_prover.Repair.blueprint import BLUEPRINT_REPAIR_SYSTEM_PROMPT


def blueprint(problem: TheoremProblem) -> Blueprint:
    return Blueprint(
        blueprint_summary="Use truth.",
        nodes=[full_node("L1")],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
        warnings=[],
        metadata={},
    )


class RepairClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def with_state(value: Blueprint, state: str) -> dict:
    return {**value.model_dump(mode="json", by_alias=True), "state": state}


def test_blueprint_repair_prompt_assigns_state_after_concrete_repair() -> None:
    prompt = BLUEPRINT_REPAIR_SYSTEM_PROMPT
    execution_order = [
        prompt.index("1. SNAPSHOT"),
        prompt.index("2. INSPECT"),
        prompt.index("3. REPAIR"),
        prompt.index("4. COMPARE"),
        prompt.index("5. ASSIGN STATE LAST"),
    ]
    assert execution_order == sorted(execution_order)
    assert 'NEVER return `state="failed"` with an unchanged Blueprint' in prompt
    assert 'NEVER return `state="failed"` as a substitute' in prompt
    assert "The complete returned Blueprint\n  IS the executed repair" in prompt
    assert "serialize `state` as the\nFIRST top-level JSON key" in prompt


def test_blueprint_repair_self_loop_is_stateless_and_stops_on_success() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
        natural_language_statement="Truth holds.",
    )
    fixed = blueprint(problem)
    raw = fixed.model_dump(mode="json", by_alias=True)
    raw.pop("metadata")
    client = RepairClient([
        with_state(fixed, "failed"),
        with_state(fixed, "success"),
    ])

    result, rounds = BluePrintRepairer(client).run(
        problem=problem,
        environment=environment(),
        candidate=raw,
    )

    assert result == fixed
    assert [item.state.value for item in rounds] == ["failed", "success"]
    assert rounds[0].changed_fields == ["metadata"]
    assert rounds[1].changed_fields == []
    assert rounds[0].input_snapshot == raw
    assert all(call["history"] is None for call in client.calls)
    second_candidate = client.calls[1]["user_prompt"].split(
        "BEGIN_CURRENT_BLUEPRINT_CANDIDATE\n", 1
    )[1].split("\nEND_CURRENT_BLUEPRINT_CANDIDATE", 1)[0]
    assert json.loads(second_candidate) == fixed.model_dump(
        mode="json", by_alias=True
    )


def test_blueprint_repair_rechecks_success_that_changes_valid_input() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    original = blueprint(problem)
    changed = original.model_copy(deep=True)
    changed.blueprint_summary = "Changed despite success."
    client = RepairClient([
        with_state(changed, "success"),
        with_state(changed, "success"),
    ])

    result, rounds = BluePrintRepairer(client).run(
        problem=problem,
        environment=environment(),
        candidate=original,
    )
    assert result == changed
    assert len(rounds) == 2
    assert rounds[0].output_error == "state=success must not modify Blueprint data"


def test_blueprint_repair_constructs_success_for_unchanged_valid_input() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    fixed = blueprint(problem)
    client = RepairClient([with_state(fixed, "failed")])

    result, rounds = BluePrintRepairer(client).run(
        problem=problem,
        environment=environment(),
        candidate=fixed,
    )
    assert result == fixed
    assert len(rounds) == 1
    assert rounds[0].state.value == "success"
    assert rounds[0].changed is False
    assert rounds[0].output_error is None
    assert rounds[0].raw_output == with_state(fixed, "failed")


def test_blueprint_repair_rechecks_its_own_invalid_output_next_round() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    fixed = blueprint(problem)
    invalid = with_state(fixed, "failed")
    invalid["nodes"][0]["semantic_alignment"]["objects"] = []
    client = RepairClient([
        invalid,
        with_state(fixed, "failed"),
        with_state(fixed, "success"),
    ])

    result, rounds = BluePrintRepairer(client).run(
        problem=problem,
        environment=environment(),
        candidate={"broken": True},
    )

    assert result == fixed
    assert len(rounds) == 3
    assert rounds[0].blueprint is None
    assert rounds[0].state is None
    assert rounds[0].raw_output == invalid
    assert "objects" in (rounds[0].output_error or "")
    assert all(call["history"] is None for call in client.calls)
    assert json.dumps(invalid, ensure_ascii=False, indent=2) in client.calls[1][
        "user_prompt"
    ]
