import pytest

from lean_prover.Planner.graph import (
    BlueprintGraphError,
    build_dependency_graph,
    dependents,
    find_disconnected_nodes,
    ready_nodes,
    to_mermaid,
    topological_order,
)
from lean_prover.Planner.schemas import Blueprint, BlueprintNode
from lean_prover.Planner.tests.helpers import environment, full_node


def make_node(
    node_id: str,
    *,
    depends_on: list[str] | None = None,
    title: str | None = None,
) -> BlueprintNode:
    node = full_node(node_id, depends_on=depends_on)
    node.title = title or node_id
    return node


def test_build_dependency_graph_and_topological_order() -> None:
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[
            make_node("L1"),
            make_node("L2", depends_on=["L1"]),
        ],
        root_dependencies=["L2"],
        environment=environment(),
    )

    graph = build_dependency_graph(blueprint)
    order = topological_order(blueprint)

    assert graph == {"L1": set(), "L2": {"L1"}, "ROOT": {"L2"}}
    assert order.index("L1") < order.index("L2") < order.index("ROOT")


def test_missing_dependency_raises_graph_error() -> None:
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[make_node("L1", depends_on=["L2"])],
        root_dependencies=["L1"],
        environment=environment(),
    )

    with pytest.raises(BlueprintGraphError, match="missing nodes"):
        build_dependency_graph(blueprint)


def test_ready_nodes_dependents_disconnected_and_mermaid() -> None:
    blueprint = Blueprint(
        blueprint_summary="demo",
        nodes=[
            make_node("L1"),
            make_node("L2", depends_on=["L1"]),
            make_node("L3", title='Quote "safe"'),
        ],
        root_dependencies=["L2"],
        environment=environment(),
    )

    assert [node.id for node in ready_nodes(blueprint, set())] == ["L1", "L3"]
    assert [node.id for node in ready_nodes(blueprint, {"L1"})] == ["L2", "L3"]
    assert dependents(blueprint)["L2"] == ["ROOT"]
    assert find_disconnected_nodes(blueprint) == {"L3"}

    mermaid = to_mermaid(blueprint)
    assert "graph TD" in mermaid
    assert "L1 --> L2" in mermaid
    assert "L2 --> ROOT" in mermaid
    assert "Quote 'safe'" in mermaid
