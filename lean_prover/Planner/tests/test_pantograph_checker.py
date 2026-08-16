from pathlib import Path

from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanPreamble,
    TheoremProblem,
)
from lean_prover.Planner.tests.helpers import environment, full_node, preamble


class FakePantographServer:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.sources = []
        self.closed = False

    def load_sorry(self, source: str, ignore_values: bool = False):
        self.sources.append(
            {
                "source": source,
                "ignore_values": ignore_values,
            }
        )
        if self.fail_on and self.fail_on in source:
            raise RuntimeError("fake elaboration error")
        return [object()]

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True


def make_checker(
    server: FakePantographServer,
) -> PantographDeclarationCheckingBackend:
    return PantographDeclarationCheckingBackend(
        project_path=Path("."),
        _server=server,
        _api={
            "ServerError": RuntimeError,
            "TacticFailure": RuntimeError,
            "get_version": lambda: "fake",
        },
    )


def make_node(node_id: str, lean_decl: str) -> BlueprintNode:
    return full_node(node_id, lean_decl)


def test_check_declaration_builds_context_with_sorry_bodies() -> None:
    server = FakePantographServer()
    checker = make_checker(server)

    result = checker.check_declaration(
        imports=["Mathlib"],
        preceding_declarations=["lemma L1 : True"],
        declaration="lemma L2 : True",
        preamble=preamble(),
    )

    assert result.success is True
    assert server.sources[0]["ignore_values"] is True
    source = server.sources[0]["source"]
    assert "import Mathlib" not in source
    assert "lemma L1 : True := by\n  sorry" in source
    assert "lemma L2 : True := by\n  sorry" in source


def test_check_declaration_returns_failed_result_on_pantograph_error() -> None:
    server = FakePantographServer(fail_on="bad_symbol")
    checker = make_checker(server)

    result = checker.check_declaration(
        imports=["Mathlib"],
        preceding_declarations=[],
        declaration="lemma L1 : bad_symbol",
        preamble=preamble(),
    )

    assert result.success is False
    assert "fake elaboration error" in result.error_message


def test_check_declaration_materializes_complete_structured_preamble() -> None:
    server = FakePantographServer()
    checker = make_checker(server)
    node_preamble = LeanPreamble(
        imports=["Mathlib"],
        namespaces=["PlannerDemo"],
        open_namespaces=["Nat"],
        open_scoped=["BigOperators"],
        variable_declarations=["variable (α : Type*)"],
        local_context=["noncomputable section"],
    )

    result = checker.check_declaration(
        imports=["Mathlib"],
        preceding_declarations=[],
        declaration="lemma L1 : True",
        preamble=node_preamble,
    )

    assert result.success is True
    source = server.sources[0]["source"]
    assert source.startswith("namespace PlannerDemo")
    assert "open Nat" in source
    assert "open scoped BigOperators" in source
    assert "variable (α : Type*)" in source
    assert "noncomputable section" in source
    assert "\nend\nend PlannerDemo" in source
    assert source.rstrip().endswith("end PlannerDemo")


def test_check_declaration_places_raw_header_before_dependencies_and_target() -> None:
    server = FakePantographServer()
    checker = make_checker(server)
    header = "import Mathlib\n\nopen Nat"

    result = checker.check_declaration(
        imports=["Mathlib"],
        preceding_declarations=["lemma L1 : True"],
        declaration="lemma L2 : True",
        preamble=LeanPreamble(imports=["Mathlib"], raw_header=header),
    )

    assert result.success
    source = server.sources[0]["source"]
    assert "import Mathlib" not in source
    assert source.count("open Nat") == 1
    assert source.index("open Nat") < source.index("lemma L1")
    assert source.index("lemma L1") < source.index("lemma L2")


def test_check_blueprint_aggregates_failed_node_as_validation_issue() -> None:
    server = FakePantographServer(fail_on="bad_symbol")
    checker = make_checker(server)
    problem = TheoremProblem(
        problem_id="demo",
        imports=["Mathlib"],
        target_lean_decl="theorem target : True",
    )
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[
            make_node("L1", "lemma L1 : True"),
            make_node("L2", "lemma L2 : bad_symbol"),
        ],
        root_dependencies=["L2"],
        environment=environment(),
    )
    blueprint.nodes[1].depends_on = ["L1"]

    issues = checker.check_blueprint(problem, blueprint)

    assert len(issues) == 1
    assert issues[0].stage == "lean"
    assert issues[0].code == "declaration_failed"
    assert issues[0].node_id == "L2"
    assert len(server.sources) == 2
    assert "lemma L1 : True := by\n  sorry" in server.sources[1]["source"]
