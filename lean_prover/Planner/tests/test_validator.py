from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    BlueprintPlan,
    BlueprintPlanNode,
    TheoremProblem,
)
from lean_prover.Planner.tests.helpers import environment, full_node, proof_length
from lean_prover.Planner.validator import (
    validate_blueprint,
    validate_blueprint_plan,
    validate_formalized_blueprint,
)


def make_node(
    node_id: str,
    *,
    statement: str | None = None,
    lean_decl: str | None = None,
    depends_on: list[str] | None = None,
) -> BlueprintNode:
    node = full_node(
        node_id,
        lean_decl or f"lemma {node_id} : True",
        depends_on=depends_on,
    )
    node.informal_statement = statement or f"Statement for {node_id}"
    return node


def test_validate_blueprint_accepts_simple_valid_blueprint() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[
            make_node("L1"),
            make_node("L2", depends_on=["L1"]),
        ],
        root_dependencies=["L2"],
        environment=environment(),
    )

    result = validate_blueprint(problem, blueprint)

    assert result.valid is True
    assert result.issues == []


def test_validate_blueprint_reports_common_static_issues() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[
            make_node("L1", statement="duplicate"),
            make_node(
                "L2",
                statement="duplicate",
                lean_decl="lemma L2 : theorem target : True",
            ),
            make_node("L3"),
        ],
        root_dependencies=["L1"],
        environment=environment(),
    )

    result = validate_blueprint(problem, blueprint, max_nodes=2)
    codes = {issue.code for issue in result.issues}

    assert result.valid is False
    assert "too_many_nodes" in codes
    assert "disconnected_node" in codes
    assert "duplicate_statement" in codes
    assert "target_restatement" in codes


def test_formalization_cannot_change_frozen_informal_statement() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    planned_node = BlueprintPlanNode(
        id="L1",
        title="Original",
        informal_statement="The proposition True holds.",
        informal_proof="Use the constructor of True.",
        logical_ideas=["Construct True"],
        lean_statement="",
        proof_strategy="Use the constructor of True.",
        estimated_proof_length=proof_length(),
        difficulty=1,
    )
    plan = BlueprintPlan(
        blueprint_summary="direct lemma",
        nodes=[planned_node],
        root_dependencies=["L1"],
    )
    formalized = full_node("L1")
    formalized.title = planned_node.title
    formalized.informal_statement = "A changed, weaker statement."
    formalized.proof_strategy = planned_node.proof_strategy
    blueprint = Blueprint(
        blueprint_summary=plan.blueprint_summary,
        nodes=[formalized],
        root_dependencies=plan.root_dependencies,
        environment=environment(),
    )

    plan_result = validate_blueprint_plan(problem, plan)
    formalized_result = validate_formalized_blueprint(
        problem,
        plan,
        blueprint,
        environment(),
    )

    assert plan_result.valid is True
    assert formalized_result.valid is False
    assert any(
        issue.code == "frozen_node_drift"
        and issue.node_id == "L1"
        and "informal_statement" in issue.message
        for issue in formalized_result.issues
    )


def test_rejects_informal_proof_that_omits_dependency_reference() -> None:
    problem = TheoremProblem(
        problem_id="demo",
        target_lean_decl="theorem target : True",
    )
    first = full_node("L1")
    second = full_node("L2", depends_on=["L1"])
    second.informal_proof = "Apply the stated inference without naming its premise."
    blueprint = Blueprint(
        blueprint_summary="two steps",
        nodes=[first, second],
        root_dependencies=["L2"],
        environment=environment(),
    )

    result = validate_blueprint(problem, blueprint)

    assert any(
        issue.code == "invalid_informal_proof_dependencies"
        and issue.node_id == "L2"
        for issue in result.issues
    )
