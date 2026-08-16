"""Manage the dependency graph of a blueprint, including topological sorting and cycle detection."""
from __future__ import annotations

from collections import defaultdict, deque
from graphlib import CycleError, TopologicalSorter

from .schemas import Blueprint, BlueprintNode


class BlueprintGraphError(ValueError):
    pass


def nodes_by_id(
    blueprint: Blueprint,
) -> dict[str, BlueprintNode]:
    result: dict[str, BlueprintNode] = {}

    for node in blueprint.nodes:
        if node.id in result:
            raise BlueprintGraphError(
                f"Duplicate node ID: {node.id}"
            )
        result[node.id] = node

    return result


def build_dependency_graph(
    blueprint: Blueprint,
) -> dict[str, set[str]]:
    known_nodes = nodes_by_id(blueprint)
    all_ids = set(known_nodes)

    graph: dict[str, set[str]] = {}

    for node in blueprint.nodes:
        dependencies = set(node.depends_on)
        missing = dependencies - all_ids

        if missing:
            raise BlueprintGraphError(
                f"{node.id} depends on missing nodes: {missing}"
            )

        if node.id in dependencies:
            raise BlueprintGraphError(
                f"{node.id} depends on itself"
            )

        graph[node.id] = dependencies

    root_dependencies = set(blueprint.root_dependencies)
    missing_root = root_dependencies - all_ids

    if missing_root:
        raise BlueprintGraphError(
            f"ROOT depends on missing nodes: {missing_root}"
        )

    graph["ROOT"] = root_dependencies
    return graph


def topological_order(blueprint: Blueprint) -> list[str]:
    graph = build_dependency_graph(blueprint)

    try:
        order = list(
            TopologicalSorter(graph).static_order()
        )
    except CycleError as error:
        raise BlueprintGraphError(
            f"Blueprint contains a cycle: {error}"
        ) from error

    return order


def find_disconnected_nodes(
    blueprint: Blueprint,
) -> set[str]:
    graph = build_dependency_graph(blueprint)
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return

        visited.add(node_id)

        for dependency in graph[node_id]:
            visit(dependency)

    visit("ROOT")

    all_node_ids = {node.id for node in blueprint.nodes}
    return all_node_ids - visited


def ready_nodes(
    blueprint: Blueprint,
    proved_node_ids: set[str],
) -> list[BlueprintNode]:
    return [
        node
        for node in blueprint.nodes
        if node.id not in proved_node_ids
        and set(node.depends_on) <= proved_node_ids
    ]


def dependents(
    blueprint: Blueprint,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)

    for node in blueprint.nodes:
        for dependency in node.depends_on:
            result[dependency].append(node.id)

    for dependency in blueprint.root_dependencies:
        result[dependency].append("ROOT")

    return dict(result)


def to_mermaid(blueprint: Blueprint) -> str:
    lines = ["graph TD"]

    for node in blueprint.nodes:
        safe_title = node.title.replace('"', "'")
        lines.append(
            f'    {node.id}["{node.id}: {safe_title}"]'
        )

    lines.append('    ROOT["Target theorem"]')

    for node in blueprint.nodes:
        for dependency in node.depends_on:
            lines.append(
                f"    {dependency} --> {node.id}"
            )

    for dependency in blueprint.root_dependencies:
        lines.append(f"    {dependency} --> ROOT")

    return "\n".join(lines)
