from __future__ import annotations

import json
import threading
import time

from lean_prover.Planner.schemas import Blueprint, TheoremProblem
from lean_prover.Planner.tests.helpers import environment, full_node
from lean_prover.Verify.blueprint import (
    VERIFY_SYSTEM_PROMPT,
    BlueprintVerificationBatchError,
    BlueprintVerifier,
    JsonlVerifyStore,
    _semantic_prompt,
)


class VerifyClient:
    def __init__(self, *, formal_correct: bool = True) -> None:
        self.formal_correct = formal_correct
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def generate_json(self, *, user_prompt, empty_response_message, **_):
        payload = json.loads(
            user_prompt.split("SEMANTIC_VERIFY_INPUT_JSON:\n", 1)[1]
        )
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        issue = {
            "component": "formal_statement",
            "message": "Goal differs.",
            "repair_reference": "Use the exact requested goal.",
        }
        node_id = payload["node_id_must_be_preserved"]
        return {
            "formal_statement_correct": self.formal_correct,
            "corrected_lean_statement": (
                "" if self.formal_correct else f"lemma {node_id} : False"
            ),
            "issue_message": "" if self.formal_correct else issue["message"],
            "repair_reference": (
                "" if self.formal_correct else issue["repair_reference"]
            ),
            "verification_notes": "Audited exact objects, hypotheses, and goal.",
        }


def test_verify_runs_nodes_in_parallel_and_writes_jsonl(tmp_path) -> None:
    client = VerifyClient()
    store = JsonlVerifyStore(tmp_path / "verify.jsonl")
    verifier = BlueprintVerifier(client, store=store, max_workers=2)
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="two",
        nodes=[full_node("L1"), full_node("L2", depends_on=["L1"])],
        root_dependencies=["L2"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    results = verifier.verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )

    assert [result.node_id for result in results] == ["L1", "L2"]
    assert client.maximum_active == 2
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert {row["node_id"] for row in rows} == {"L1", "L2"}


def test_verify_false_requires_actionable_repair_reference() -> None:
    client = VerifyClient(formal_correct=False)
    verifier = BlueprintVerifier(client)
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[full_node("L1")],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    result = verifier.verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert result.formal_statement_correct is False
    assert result.formal_statement_issues[0].repair_reference


def test_verify_prompt_uses_value_only_template_and_few_shots() -> None:
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    node = full_node("L1")
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    prompt = _semantic_prompt(
        problem=problem,
        node=node,
        environment=environment(),
    )
    template_text = prompt.split(
        "SEMANTIC_VALUES_TEMPLATE_JSON:\n", 1
    )[1].split("\n\nSEMANTIC_VERIFY_INPUT_JSON:\n", 1)[0]
    template = json.loads(template_text)

    assert list(template) == [
        "formal_statement_correct",
        "corrected_lean_statement",
        "issue_message",
        "repair_reference",
        "verification_notes",
    ]
    assert "program, not you, constructs BlueprintNodeVerification" in prompt
    assert "Few-shot -- exact match" in VERIFY_SYSTEM_PROMPT
    assert "Few-shot -- semantic mismatch" in VERIFY_SYSTEM_PROMPT


def test_program_constructs_dependency_failure_without_model_values() -> None:
    client = VerifyClient()
    verifier = BlueprintVerifier(client)
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    node = full_node("L1")
    node.depends_on = ["L9"]
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    result = verifier.verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert result.dependency_statements_correct is False
    assert result.dependency_issues[0].component == "dependencies"
    assert result.corrected_preamble == node.preamble


def test_program_deduplicates_header_and_preamble_values() -> None:
    from lean_prover.Planner.schemas import LeanPreamble

    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
        header="import Mathlib\nopen Nat\nopen Nat",
    )
    node = full_node("L1")
    node.preamble = LeanPreamble(
        imports=["Mathlib"],
        raw_header=problem.header,
        open_namespaces=["Nat", "Nat"],
    )
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    result = BlueprintVerifier(VerifyClient()).verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert result.corrected_preamble.raw_header.count("open Nat") == 1
    assert result.corrected_preamble.open_namespaces == []
    assert result.header_changes


class OneFailingVerifyClient(VerifyClient):
    def generate_json(self, *, user_prompt, **kwargs):
        payload = json.loads(
            user_prompt.split("SEMANTIC_VERIFY_INPUT_JSON:\n", 1)[1]
        )
        if payload["node_id_must_be_preserved"] == "L1":
            raise RuntimeError("bad semantic response")
        return super().generate_json(user_prompt=user_prompt, **kwargs)


class NoOpSemanticPatchClient(VerifyClient):
    def generate_json(self, *, user_prompt, **kwargs):
        payload = json.loads(
            user_prompt.split("SEMANTIC_VERIFY_INPUT_JSON:\n", 1)[1]
        )
        return {
            "formal_statement_correct": False,
            "corrected_lean_statement": payload["node_lean_statement"],
            "issue_message": "The formalization is semantically wrong.",
            "repair_reference": "Reformalize it from the natural language.",
            "verification_notes": "The proposed patch was a no-op.",
        }


def test_noop_semantic_patch_becomes_issue_for_formalization_repair() -> None:
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    node = full_node("L1")
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    result = BlueprintVerifier(NoOpSemanticPatchClient()).verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert not result.formal_statement_correct
    assert blueprint.nodes[0].lean_decl == "lemma L1 : True"
    assert blueprint.nodes[0].metadata["verify_semantic_repair_count"] == 1
    assert "verify_semantic_patch_pending" not in blueprint.nodes[0].metadata


class SparseMismatchClient(VerifyClient):
    def generate_json(self, **kwargs):
        return {
            "formal_statement_correct": False,
            "corrected_lean_statement": "",
            "unexpected_model_comment": "ignored",
        }


def test_sparse_semantic_values_are_completed_by_program() -> None:
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    node = full_node("L1")
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    result = BlueprintVerifier(SparseMismatchClient()).verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert not result.formal_statement_correct
    assert result.formal_statement_issues[0].message
    assert result.formal_statement_issues[0].repair_reference
    assert result.verification_notes


class MissingVerdictOnceClient(VerifyClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.histories: list[list[dict[str, str]]] = []

    def generate_json(self, *, history=None, **kwargs):
        self.calls += 1
        self.histories.append(list(history or []))
        if self.calls == 1:
            return {"verification_notes": "forgot the verdict"}
        return {"formal_statement_correct": True}


def test_verify_retries_json_missing_required_verdict() -> None:
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[full_node("L1")],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )
    client = MissingVerdictOnceClient()

    result = BlueprintVerifier(client).verify_blueprint(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )[0]

    assert result.formal_statement_correct
    assert client.calls == 2
    assert client.histories[1][-1]["role"] == "user"


def test_verify_collects_all_parallel_nodes_before_raising(tmp_path) -> None:
    store = JsonlVerifyStore(tmp_path / "verify.jsonl")
    verifier = BlueprintVerifier(
        OneFailingVerifyClient(), store=store, max_workers=2
    )
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="two",
        nodes=[full_node("L1"), full_node("L2")],
        root_dependencies=["L1", "L2"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    try:
        verifier.verify_blueprint(
            problem=problem,
            blueprint=blueprint,
            environment=environment(),
        )
    except BlueprintVerificationBatchError as error:
        assert set(error.failures) == {"L1"}
        assert [result.node_id for result in error.partial_results] == ["L2"]
    else:
        raise AssertionError("one failing node must raise after collection")
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert {row["record_type"] for row in rows} == {
        "verification_result",
        "verification_error",
    }


def test_third_semantic_mismatch_raises_after_two_repairs() -> None:
    client = VerifyClient(formal_correct=False)
    verifier = BlueprintVerifier(client, max_semantic_repairs=2)
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
    )
    node = full_node("L1")
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )

    verifier.verify_blueprint(
        problem=problem, blueprint=blueprint, environment=environment()
    )
    node.lean_decl = "lemma L1 : True"
    verifier.verify_blueprint(
        problem=problem, blueprint=blueprint, environment=environment()
    )
    node.lean_decl = "lemma L1 : True"
    try:
        verifier.verify_blueprint(
            problem=problem,
            blueprint=blueprint,
            environment=environment(),
        )
    except BlueprintVerificationBatchError as error:
        failure = error.failures["L1"]
        from lean_prover.Verify import SemanticVerificationExhaustedError

        assert isinstance(failure, SemanticVerificationExhaustedError)
        assert "2轮verify-repair后语义仍不一致" in str(failure)
    else:
        raise AssertionError("a third mismatch must fail the node")


def test_preamble_renderer_removes_raw_header_duplicates() -> None:
    from lean_prover.Planner.preamble import wrap_source_with_preamble
    from lean_prover.Planner.schemas import LeanPreamble

    source = wrap_source_with_preamble(
        "lemma L1 : True",
        LeanPreamble(
            imports=["Mathlib"],
            raw_header="import Mathlib\nopen BigOperators\nopen scoped Real",
            open_namespaces=["BigOperators"],
            open_scoped=["Real"],
        ),
    )

    assert source.count("open BigOperators") == 1
    assert source.count("open scoped Real") == 1


def test_bare_hypothesis_is_rejected_from_local_context() -> None:
    import pytest
    from pydantic import ValidationError
    from lean_prover.Planner.schemas import LeanPreamble

    with pytest.raises(ValidationError, match="bare hypothesis"):
        LeanPreamble(imports=["Mathlib"], local_context=["h : True"])
