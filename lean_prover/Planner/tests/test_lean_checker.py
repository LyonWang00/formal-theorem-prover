from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanCheckResult,
    LeanPreamble,
    NodeStatus,
    TheoremProblem,
)
from lean_prover.Planner.tests.helpers import environment, full_node


class FakeBackend:
    def __init__(self, results: list[LeanCheckResult]) -> None:
        self.results = results
        self.calls = []

    def check_declaration(
        self,
        *,
        imports: list[str],
        declaration: str,
        preceding_declarations: list[str],
        preamble: LeanPreamble,
    ) -> LeanCheckResult:
        self.calls.append(
            {
                "imports": imports,
                "declaration": declaration,
                "preceding_declarations": list(preceding_declarations),
                "preamble": preamble,
            }
        )
        return self.results.pop(0)


def make_node(node_id: str) -> BlueprintNode:
    return full_node(node_id)


def test_check_blueprint_collects_lean_failures_and_preserves_context() -> None:
    backend = FakeBackend(
        [
            LeanCheckResult(success=True, declaration="lemma L1 : True"),
            LeanCheckResult(
                success=False,
                declaration="lemma L2 : True",
                error_message="type mismatch",
            ),
        ]
    )
    checker = BlueprintLeanChecker(backend)
    problem = TheoremProblem(
        problem_id="demo",
        imports=["Mathlib"],
        target_lean_decl="theorem target : True",
    )
    l1 = make_node("L1")
    l2 = make_node("L2")
    l2.depends_on = ["L1"]
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[l1, l2],
        root_dependencies=["L2"],
        environment=environment(),
    )

    issues = checker.check_blueprint(problem, blueprint)

    assert len(issues) == 1
    assert issues[0].stage == "lean"
    assert issues[0].node_id == "L2"
    assert "type mismatch" in issues[0].message
    assert backend.calls[1]["preceding_declarations"] == ["lemma L1 : True"]


def test_check_blueprint_only_materializes_declared_dependencies() -> None:
    backend = FakeBackend(
        [
            LeanCheckResult(success=True, declaration="lemma L1 : True"),
            LeanCheckResult(success=True, declaration="lemma L2 : True"),
        ]
    )
    first = make_node("L1")
    second = make_node("L2")
    checker = BlueprintLeanChecker(backend)
    checker.check_blueprint_detailed(
        TheoremProblem(
            problem_id="demo",
            imports=["Mathlib"],
            target_lean_decl="theorem target : True",
        ),
        Blueprint(
            blueprint_summary="independent",
            nodes=[first, second],
            root_dependencies=["L1", "L2"],
            environment=environment(),
        ),
    )

    assert backend.calls[1]["preceding_declarations"] == []


def test_detailed_check_is_atomic_and_marks_every_failed_node() -> None:
    backend = FakeBackend(
        [
            LeanCheckResult(
                success=False,
                declaration="lemma L1 : BadOne",
                error_message="unknown BadOne",
            ),
            LeanCheckResult(
                success=False,
                declaration="lemma L2 : BadTwo",
                error_message="unknown BadTwo",
            ),
        ]
    )
    checker = BlueprintLeanChecker(backend)
    problem = TheoremProblem(
        problem_id="demo",
        imports=["Mathlib"],
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[make_node("L1"), make_node("L2")],
        root_dependencies=["L2"],
        environment=environment(),
    )

    result = checker.check_blueprint_detailed(problem, blueprint)

    assert result.success is False
    assert result.failed_node_ids == ["L1", "L2"]
    assert len(result.node_results) == 2
    assert [node.status for node in blueprint.nodes] == [
        NodeStatus.FAILED,
        NodeStatus.FAILED,
    ]
