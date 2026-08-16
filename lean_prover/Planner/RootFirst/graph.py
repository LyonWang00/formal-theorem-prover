"""Deterministic graph checks for father -> child RootFirst Blueprints."""

from __future__ import annotations

from collections import deque

from .schemas import RootFirstBlueprint


class GraphValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(" | ".join(errors))


def graph_errors(blueprint: RootFirstBlueprint) -> list[str]:
    errors: list[str] = []
    ids = [node.id for node in blueprint.nodes]
    known = set(ids)
    if len(ids) != len(known):
        errors.append("duplicate node IDs")
    if blueprint.root_id not in known:
        errors.append("L0 root node is missing")
        return errors
    nodes = blueprint.node_map()
    root = nodes[blueprint.root_id]
    if root.children:
        errors.append("L0 must have no children and be the unique sink")

    for node in blueprint.nodes:
        if node.id in node.father_nodes or node.id in node.children:
            errors.append(f"{node.id} contains a self-loop")
        for father in node.father_nodes:
            if father not in known:
                errors.append(f"{node.id} references missing father {father}")
            elif node.id not in nodes[father].children:
                errors.append(
                    f"edge mismatch: {node.id} names father {father}, but "
                    f"{father}.children omits {node.id}"
                )
        for child in node.children:
            if child not in known:
                errors.append(f"{node.id} references missing child {child}")
            elif node.id not in nodes[child].father_nodes:
                errors.append(
                    f"edge mismatch: {node.id}.children contains {child}, but "
                    f"{child}.father_nodes omits {node.id}"
                )

    indegree = {node_id: 0 for node_id in known}
    for node in blueprint.nodes:
        for child in node.children:
            if child in indegree:
                indegree[child] += 1
    queue = deque(sorted(node_id for node_id, value in indegree.items() if value == 0))
    visited: list[str] = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for child in nodes[current].children:
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(known):
        errors.append("graph contains a directed cycle")

    sinks = sorted(node.id for node in blueprint.nodes if not node.children)
    if sinks != [blueprint.root_id]:
        errors.append(f"L0 must be the unique sink; actual sinks={sinks}")

    reachable: set[str] = set()

    def reaches_root(node_id: str, active: set[str]) -> bool:
        if node_id == blueprint.root_id:
            return True
        if node_id in active:
            return False
        active = set(active)
        active.add(node_id)
        return any(
            child in nodes and reaches_root(child, active)
            for child in nodes[node_id].children
        )

    for node_id in sorted(known):
        if reaches_root(node_id, set()):
            reachable.add(node_id)
    unreachable = sorted(known - reachable)
    if unreachable:
        errors.append(f"nodes cannot reach L0: {unreachable}")
    return errors


def validate_graph(blueprint: RootFirstBlueprint) -> None:
    errors = graph_errors(blueprint)
    if errors:
        raise GraphValidationError(errors)


def topological_order(blueprint: RootFirstBlueprint) -> list[str]:
    validate_graph(blueprint)
    nodes = blueprint.node_map()
    indegree = {node.id: len(node.father_nodes) for node in blueprint.nodes}
    queue = deque(sorted(node_id for node_id, value in indegree.items() if value == 0))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for child in nodes[current].children:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return order

