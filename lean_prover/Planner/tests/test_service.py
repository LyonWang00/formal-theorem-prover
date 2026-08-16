import json
from typing import Any

from lean_prover.Planner.schemas import (
    BlueprintLeanCheckResult,
    Blueprint,
    BlueprintNodeVerification,
    BlueprintPlan,
    BlueprintPlanNode,
    InputClassification,
    LeanCheckResult,
    NodeLeanCheckResult,
    PlannerResult,
    ProblemInputKind,
    RawTheoremInput,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Planner.service import PlannerService
from lean_prover.Planner.tests.helpers import environment, full_node, proof_length
from lean_prover.Verify import (
    BlueprintVerificationBatchError,
    SemanticVerificationExhaustedError,
)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history=None,
    ) -> dict[str, Any]:
        if "You are BluePrintRepair" in system_prompt:
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
            payload = json.loads(
                user_prompt.split("SEMANTIC_VERIFY_INPUT_JSON:\n", 1)[1]
            )
            return {
                "formal_statement_correct": True,
                "corrected_lean_statement": "",
                "issue_message": "",
                "repair_reference": "",
                "verification_notes": "Exact test fixture match.",
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


class FakeLeanChecker:
    def __init__(
        self,
        failed_ids: set[str] | None = None,
        *,
        target_success: bool = True,
    ) -> None:
        self.failed_ids = failed_ids or set()
        self.target_success = target_success
        self.calls = []

    def check_blueprint_detailed(self, problem, blueprint):
        self.calls.append((problem, blueprint))
        node_results = []
        issues = []
        for node in blueprint.nodes:
            success = node.id not in self.failed_ids
            result = LeanCheckResult(
                success=success,
                declaration=node.lean_decl,
                error_message=None if success else f"{node.id} failed",
            )
            node_results.append(
                NodeLeanCheckResult(
                    node_id=node.id,
                    success=success,
                    lean_statement=node.lean_decl,
                    preamble=node.preamble,
                    result=result,
                )
            )
            if not success:
                issues.append(
                    ValidationIssue(
                        stage="lean",
                        code="declaration_failed",
                        node_id=node.id,
                        message=f"{node.id} failed",
                    )
                )
        return BlueprintLeanCheckResult(
            success=not issues,
            node_results=node_results,
            failed_node_ids=[issue.node_id for issue in issues if issue.node_id],
            issues=issues,
        )

    def check_target_declaration(self, problem):
        return LeanCheckResult(
            success=self.target_success,
            declaration=problem.target_lean_decl,
            stdout="target accepted" if self.target_success else "",
            error_message=None if self.target_success else "target rejected",
        )


def test_verify_cannot_erase_authoritative_problem_header() -> None:
    problem = TheoremProblem(
        problem_id="header-demo",
        target_lean_decl="theorem target : True",
        header="import Mathlib\n\nopen Nat",
    )
    node = full_node("L1")
    node.preamble.raw_header = problem.header
    bp = Blueprint(
        blueprint_summary="direct",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )
    verification = BlueprintNodeVerification(
        node_id="L1",
        corrected_preamble=node.preamble.model_copy(update={"raw_header": ""}),
        dependency_statements_correct=True,
        formal_statement_correct=True,
        verification_notes="No semantic issue.",
    )

    issues = PlannerService._apply_verify_results(bp, [verification])

    assert not issues
    assert bp.nodes[0].preamble.raw_header == problem.header


class SemanticPatchClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.semantic_calls = 0

    def generate_json(self, *, system_prompt, user_prompt, **kwargs):
        if "compare one natural-language mathematical" in system_prompt:
            self.semantic_calls += 1
            if self.semantic_calls == 1:
                return {
                    "formal_statement_correct": False,
                    "corrected_lean_statement": "lemma L1 : True",
                    "issue_message": "The original formal goal differed.",
                    "repair_reference": "Use the requested truth goal.",
                    "verification_notes": "Natural language is authoritative.",
                }
            return {
                "formal_statement_correct": True,
                "corrected_lean_statement": "",
                "issue_message": "",
                "repair_reference": "",
                "verification_notes": "The repaired goal matches exactly.",
            }
        return super().generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )


def test_semantic_patch_compiles_then_reenters_verify_without_repairer() -> None:
    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
        natural_language_statement="Truth holds.",
    )
    node = full_node("L1", "lemma L1 : False")
    node.informal_statement = "Truth holds."
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )
    client = SemanticPatchClient()
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )
    repair_calls = 0

    def unexpected_repair(**kwargs):
        nonlocal repair_calls
        repair_calls += 1
        return kwargs["previous_blueprint"]

    service.repairer.repair = unexpected_repair
    results1, issues1 = service._verify_blueprint_nodes(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )
    compile_result = service._compile_verified_blueprint(
        problem=problem,
        blueprint=blueprint,
    )
    results2, issues2 = service._verify_blueprint_nodes(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
        node_ids={"L1"},
    )

    assert not results1[0].formal_statement_correct
    assert issues1[0].code == "formal_statement_verification_failed"
    assert blueprint.nodes[0].lean_decl == "lemma L1 : True"
    assert compile_result.success
    assert results2[0].formal_statement_correct
    assert not issues2
    assert repair_calls == 0
    assert blueprint.nodes[0].metadata["verify_semantic_repair_count"] == 1
    assert blueprint.nodes[0].metadata["verify_pantograph_checks"][0][
        "success"
    ]


def test_uncompilable_semantic_patch_enters_formalization_repair() -> None:
    class UncompilablePatchClient(SemanticPatchClient):
        def generate_json(self, *, system_prompt, user_prompt, **kwargs):
            if "compare one natural-language mathematical" in system_prompt:
                self.semantic_calls += 1
                if self.semantic_calls == 1:
                    return {
                        "formal_statement_correct": False,
                        "corrected_lean_statement": "lemma L1 : MissingType",
                        "issue_message": "The formal goal differs.",
                        "repair_reference": "Use the requested truth goal.",
                        "verification_notes": "A semantic patch is required.",
                    }
                return {
                    "formal_statement_correct": True,
                    "corrected_lean_statement": "",
                    "issue_message": "",
                    "repair_reference": "",
                    "verification_notes": "The repaired goal matches exactly.",
                }
            return super().generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **kwargs,
            )

    class StatementChecker(FakeLeanChecker):
        def check_blueprint_detailed(self, problem, blueprint):
            self.failed_ids = {
                node.id
                for node in blueprint.nodes
                if "MissingType" in node.lean_decl
            }
            return super().check_blueprint_detailed(problem, blueprint)

    problem = TheoremProblem(
        problem_id="p",
        target_lean_decl="theorem target : True",
        natural_language_statement="Truth holds.",
    )
    node = full_node("L1", "lemma L1 : False")
    node.informal_statement = "Truth holds."
    blueprint = Blueprint(
        blueprint_summary="one",
        nodes=[node],
        root_dependencies=["L1"],
        environment=environment(),
        problem_hash=problem.problem_hash,
    )
    client = UncompilablePatchClient()
    service = PlannerService(
        client=client,
        lean_checker=StatementChecker(),
        environment=environment(),
    )
    service.lean_decomposition_api.decompose_candidate = lambda **_: {}
    service._run_blueprint_repair = lambda **_: (blueprint, [])
    repair_calls = 0

    def repair(**kwargs):
        nonlocal repair_calls
        repair_calls += 1
        repaired = kwargs["previous_blueprint"].model_copy(deep=True)
        repaired.nodes[0].lean_decl = "lemma L1 : True"
        return repaired

    service.repairer.repair = repair
    classification = InputClassification(
        input_kind=ProblemInputKind.LEAN,
        confidence=1.0,
        rationale="formal_statement",
    )

    result = service._decompose_lean_and_check(
        problem=problem,
        environment=environment(),
        classification=classification,
        base_attempts=1,
    )

    assert result.success
    assert repair_calls == 1
    assert result.blueprint.nodes[0].lean_decl == "lemma L1 : True"


def test_semantic_repair_exhaustion_is_a_node_failure_not_process_error() -> None:
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
    service = PlannerService(
        client=FakeClient([]),
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )

    def exhausted(**kwargs):
        raise BlueprintVerificationBatchError(
            failures={
                "L1": SemanticVerificationExhaustedError(
                    "节点L1经过2轮verify-repair后语义仍不一致"
                )
            },
            partial_results=[],
        )

    service.verifier.verify_blueprint = exhausted
    results, issues = service._verify_blueprint_nodes(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )

    assert results == []
    assert issues[0].node_id == "L1"
    assert issues[0].code == "semantic_repair_exhausted"


def test_mixed_parallel_verify_failures_are_collected_as_node_issues() -> None:
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
    service = PlannerService(
        client=FakeClient([]),
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )

    def mixed_failures(**kwargs):
        raise BlueprintVerificationBatchError(
            failures={
                "L1": SemanticVerificationExhaustedError("semantic mismatch"),
                "L2": RuntimeError("invalid JSON after five attempts"),
            },
            partial_results=[],
        )

    service.verifier.verify_blueprint = mixed_failures
    results, issues = service._verify_blueprint_nodes(
        problem=problem,
        blueprint=blueprint,
        environment=environment(),
    )

    assert results == []
    assert [(issue.node_id, issue.code) for issue in issues] == [
        ("L1", "semantic_repair_exhausted"),
        ("L2", "verify_node_failed"),
    ]
    assert "invalid JSON" in issues[1].message

def plan_node(node_id: str, *, depends_on: list[str] | None = None) -> BlueprintPlanNode:
    return BlueprintPlanNode(
        id=node_id,
        title=node_id,
        informal_statement=f"Statement for {node_id}",
        informal_proof=(
            "Use " + ", ".join(f"`{item}`" for item in (depends_on or []))
            + " and apply the stated inference."
            if depends_on
            else "Apply the stated direct inference."
        ),
        logical_ideas=["Apply the stated inference"],
        lean_statement="",
        depends_on=depends_on or [],
        proof_strategy="Use the assumptions directly.",
        estimated_proof_length=proof_length(),
        difficulty=1,
    )


def two_node_responses() -> list[dict[str, Any]]:
    plan = BlueprintPlan(
        blueprint_summary="two steps",
        nodes=[plan_node("L1"), plan_node("L2", depends_on=["L1"])],
        root_dependencies=["L2"],
    )
    nodes = [
        full_node("L1"),
        full_node("L2", depends_on=["L1"]),
    ]
    return [
        plan.model_dump(mode="json"),
        {
            "blueprint_summary": plan.blueprint_summary,
            "nodes": [node.model_dump(mode="json", by_alias=True) for node in nodes],
            "root_dependencies": plan.root_dependencies,
            "environment": environment().model_dump(mode="json"),
            "warnings": [],
        },
    ]


def test_plan_runs_two_api_stages_and_returns_success() -> None:
    direct_plan = BlueprintPlan(
        blueprint_summary="direct",
        nodes=[],
        root_dependencies=[],
    )
    client = FakeClient(
        [
            direct_plan.model_dump(mode="json"),
            {
                "blueprint_summary": "direct",
                "nodes": [],
                "root_dependencies": [],
                "environment": environment().model_dump(mode="json"),
            },
        ]
    )
    lean_checker = FakeLeanChecker()
    service = PlannerService(
        client=client,
        lean_checker=lean_checker,
        max_attempts=2,
        environment=environment(),
    )
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )

    result = service.plan(problem)

    assert isinstance(result, PlannerResult)
    assert result.success is True
    assert result.stage == "completed"
    assert result.attempts == 2
    assert result.decomposition_attempts == 1
    assert result.formalization_attempts == 1
    assert result.target_check is not None
    assert result.target_check.success is True
    assert result.plan is not None
    assert result.blueprint is not None
    assert len(client.calls) == 2
    assert "lean_statement must be exactly the empty string" in client.calls[0]["system_prompt"]
    assert environment().lean_commit in client.calls[1]["user_prompt"]
    assert client.calls[0]["empty_response_message"] == "分解模型空响应"
    assert client.calls[1]["empty_response_message"] == "节点形式化模型空响应"
    assert lean_checker.calls


def test_plan_returns_decomposition_schema_error_before_formalization() -> None:
    client = FakeClient([{"nodes": "not a list"}])
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(),
        max_attempts=2,
        environment=environment(),
    )
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )

    result = service.plan(problem)

    assert result.success is False
    assert result.stage == "decomposition"
    assert result.attempts == 1
    assert result.issues[0].code == "schema_error"
    assert len(client.calls) == 1


def test_invalid_root_target_stops_before_decomposition_api() -> None:
    client = FakeClient([])
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(target_success=False),
        environment=environment(),
    )
    problem = TheoremProblem(
        problem_id="bad-target",
        target_lean_decl="theorem target : MissingType",
    )

    result = service.plan(problem)

    assert result.success is False
    assert result.stage == "target_lean_validation"
    assert result.issues[0].code == "target_declaration_failed"
    assert client.calls == []


def test_any_failed_node_marks_the_whole_problem_failed() -> None:
    responses = two_node_responses()
    responses.append(responses[-1])
    service = PlannerService(
        client=FakeClient(responses),
        lean_checker=FakeLeanChecker({"L1"}),
        max_attempts=1,
        environment=environment(),
    )
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )

    result = service.plan(problem)

    assert result.success is False
    assert result.stage == "lean_validation"
    assert result.failed_node_ids == ["L1"]
    assert result.formalization_attempts == 2
    assert "L1节点形式化失败，错误为：" in result.issues[-1].message
    assert len(result.node_checks) == 2
    assert {row.node_id: row.success for row in result.node_checks} == {
        "L1": False,
        "L2": True,
    }


def routed_responses(*, input_kind: str) -> list[dict[str, Any]]:
    plan = BlueprintPlan(
        blueprint_summary="one intermediate step",
        nodes=[plan_node("L1")],
        root_dependencies=["L1"],
    )
    blueprint = {
        "blueprint_summary": plan.blueprint_summary,
        "nodes": [full_node("L1").model_dump(mode="json", by_alias=True)],
        "root_dependencies": ["L1"],
        "environment": environment().model_dump(mode="json"),
        "warnings": [],
    }
    classification = {
        "input_kind": input_kind,
        "confidence": 0.99,
        "rationale": "The syntax is unambiguous.",
    }
    if input_kind == "natural_language":
        return [
            classification,
            plan.model_dump(mode="json"),
            {
                "target_lean_decl": "theorem target : True",
                "semantic_alignment_notes": "Truth maps to True.",
                "warnings": [],
            },
            blueprint,
        ]
    return [
        classification,
        blueprint,
    ]


def test_plan_input_routes_natural_language_after_decomposition() -> None:
    client = FakeClient(routed_responses(input_kind="natural_language"))
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )

    result = service.plan_input(
        RawTheoremInput(input_text="Prove that truth holds.")
    )

    assert result.success is True
    assert result.problem is not None
    assert result.blueprint is not None
    assert result.problem.target_lean_decl == "theorem target : True"
    assert result.problem.problem_hash == result.blueprint.problem_hash
    assert result.classification_attempts == 1
    assert result.target_formalization_attempts == 1
    assert result.translation_attempts == 0
    assert "input gate" in client.calls[0]["system_prompt"]
    assert "task-decomposition" in client.calls[1]["system_prompt"]
    assert "formalize one natural-language theorem target" in client.calls[2]["system_prompt"]
    assert [call["empty_response_message"] for call in client.calls] == [
        "输入分类模型空响应",
        "分解模型空响应",
        "目标形式化模型空响应",
        "形式化节点L1时模型空响应",
    ]


def test_plan_input_routes_lean_through_dedicated_lean_decomposition() -> None:
    client = FakeClient(routed_responses(input_kind="lean"))
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )

    result = service.plan_input(
        RawTheoremInput(
            input_text="theorem given (n : Nat) : n = n",
        )
    )

    assert result.success is True
    assert result.problem is not None
    assert result.problem.target_lean_decl.startswith("theorem given")
    assert result.problem.natural_language_statement is None
    assert result.plan is None
    assert result.planner_mode is not None
    assert result.planner_mode.value == "lean"
    assert result.translation_attempts == 0
    assert result.target_formalization_attempts == 0
    assert len(client.calls) == 2
    assert client.calls[0]["empty_response_message"] == "输入分类模型空响应"
    assert client.calls[1]["empty_response_message"] == "分解模型空响应"
    assert "Lean-native task-decomposition API" in client.calls[1]["system_prompt"]
    assert "Do not translate" in " ".join(
        client.calls[1]["system_prompt"].split()
    )
    assert "theorem given" in client.calls[1]["user_prompt"]


def test_explicit_formal_statement_takes_priority_and_keeps_informal_reference() -> None:
    blueprint = {
        "blueprint_summary": "direct",
        "nodes": [],
        "root_dependencies": [],
        "environment": environment().model_dump(mode="json"),
        "warnings": [],
    }
    client = FakeClient([blueprint])
    service = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(),
        environment=environment(),
    )

    result = service.plan_input(
        RawTheoremInput(
            formal_statement="theorem supplied : True := sorry",
            informal_stmt="Truth holds.",
            header="import Mathlib\n\nopen Nat",
            imports=["Mathlib"],
        )
    )

    assert result.success
    assert result.planner_mode is not None
    assert result.planner_mode.value == "lean"
    assert result.classification_attempts == 0
    assert result.problem is not None
    assert result.problem.target_lean_decl == "theorem supplied : True"
    assert result.problem.natural_language_statement == "Truth holds."
    assert result.problem.header == "import Mathlib\n\nopen Nat"
    assert len(client.calls) == 1
    assert "Truth holds." in client.calls[0]["user_prompt"]
    assert "import Mathlib" in client.calls[0]["user_prompt"]


def test_natural_language_root_uses_pantograph_feedback_history_for_repair() -> None:
    plan = BlueprintPlan(
        blueprint_summary="direct",
        nodes=[],
        root_dependencies=[],
    )
    final_blueprint = {
        "blueprint_summary": "direct",
        "nodes": [],
        "root_dependencies": [],
        "environment": environment().model_dump(mode="json"),
    }
    client = FakeClient(
        [
            {
                "input_kind": "natural_language",
                "confidence": 1.0,
                "rationale": "prose",
            },
            plan.model_dump(mode="json"),
            {
                "target_lean_decl": "theorem target : MissingType",
                "semantic_alignment_notes": "first",
                "warnings": [],
            },
            {
                "target_lean_decl": "theorem target : True",
                "semantic_alignment_notes": "repaired",
                "warnings": [],
            },
            final_blueprint,
        ]
    )
    checker = FakeLeanChecker(target_success=False)
    calls = 0

    def check_target(problem):
        nonlocal calls
        calls += 1
        success = calls > 1
        return LeanCheckResult(
            success=success,
            declaration=problem.target_lean_decl,
            stderr="unknown identifier MissingType" if not success else "",
            error_message=(
                "unknown identifier MissingType" if not success else None
            ),
        )

    checker.check_target_declaration = check_target
    result = PlannerService(
        client=client,
        lean_checker=checker,
        max_attempts=3,
        environment=environment(),
    ).plan_input(RawTheoremInput(input_text="Truth holds."))

    assert result.success
    assert result.target_formalization_attempts == 2
    repair_call = client.calls[3]
    assert repair_call["history"]
    assert "unknown identifier MissingType" in str(repair_call["history"])


def test_natural_language_root_reports_failure_after_three_repairs() -> None:
    plan = BlueprintPlan(
        blueprint_summary="direct",
        nodes=[],
        root_dependencies=[],
    )
    target = {
        "target_lean_decl": "theorem target : MissingType",
        "semantic_alignment_notes": "attempt",
        "warnings": [],
    }
    client = FakeClient(
        [
            {
                "input_kind": "natural_language",
                "confidence": 1.0,
                "rationale": "prose",
            },
            plan.model_dump(mode="json"),
            target,
            target,
            target,
            target,
        ]
    )
    result = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(target_success=False),
        max_attempts=3,
        environment=environment(),
    ).plan_input(RawTheoremInput(input_text="Truth holds."))

    assert not result.success
    assert result.stage == "target_formalization_repair"
    assert result.target_formalization_attempts == 4
    assert "ROOT节点形式化失败，错误为：target rejected" in result.issues[-1].message
    assert len(client.calls) == 6


def test_explicit_lean_root_failure_is_not_repaired() -> None:
    client = FakeClient([])
    result = PlannerService(
        client=client,
        lean_checker=FakeLeanChecker(target_success=False),
        max_attempts=3,
        environment=environment(),
    ).plan_input(
        RawTheoremInput(
            formal_statement="theorem supplied : MissingType := sorry",
            informal_stmt="Reference prose.",
        )
    )

    assert not result.success
    assert result.stage == "target_lean_validation"
    assert result.classification_attempts == 0
    assert result.target_formalization_attempts == 0
    assert client.calls == []
