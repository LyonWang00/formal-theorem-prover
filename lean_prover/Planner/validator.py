"""Static validation for decomposition and formalization Planner stages."""

from __future__ import annotations

import re
from typing import Protocol

from .graph import BlueprintGraphError, find_disconnected_nodes, topological_order
from .schemas import (
    Blueprint,
    BlueprintPlan,
    BlueprintValidationResult,
    LeanEnvironmentIdentity,
    TheoremProblem,
    ValidationIssue,
)


class _GraphNode(Protocol):
    id: str
    informal_statement: str
    informal_proof: str
    logical_ideas: list[str]
    depends_on: list[str]


class _BlueprintGraph(Protocol):
    nodes: list[_GraphNode]
    root_dependencies: list[str]


def normalize_lean_target_input(
    value: str,
    *,
    require_target_name: bool = False,
) -> str:
    """Validate a proof-free Lean target and wrap bare propositions."""

    target = value.strip()
    target = re.sub(
        r"\s*:=\s*(?:by\s+)?sorry\s*\Z",
        "",
        target,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not target:
        raise ValueError("Lean target must not be empty")
    forbidden = re.compile(
        r"\b(sorry|admit|axiom|unsafe)\b|:=|\bby\b",
        flags=re.IGNORECASE,
    )
    match = forbidden.search(target)
    if match:
        raise ValueError(
            f"Lean target contains forbidden proof token: {match.group(0)}"
        )
    declaration = re.match(r"^(theorem|lemma)\s+([^\s(:]+)", target)
    if declaration is None:
        target = f"theorem target : {target}"
        declaration_name = "target"
    else:
        declaration_name = declaration.group(2)
    if require_target_name and declaration_name != "target":
        raise ValueError(
            "generated natural-language target must be named `target`"
        )
    return target


def _validate_common_graph(
    blueprint: _BlueprintGraph,
    *,
    max_nodes: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(blueprint.nodes) > max_nodes:
        issues.append(
            ValidationIssue(
                stage="schema",
                code="too_many_nodes",
                message=(
                    f"Blueprint has {len(blueprint.nodes)} nodes; "
                    f"maximum is {max_nodes}"
                ),
            )
        )

    expected_ids = [f"L{i}" for i in range(1, len(blueprint.nodes) + 1)]
    actual_ids = [node.id for node in blueprint.nodes]
    if actual_ids != expected_ids:
        issues.append(
            ValidationIssue(
                stage="schema",
                code="invalid_node_order",
                message=f"Expected node IDs {expected_ids}, got {actual_ids}",
            )
        )

    try:
        order = topological_order(blueprint)  # type: ignore[arg-type]
        positions = {node_id: index for index, node_id in enumerate(order)}
        for node in blueprint.nodes:
            for dependency in node.depends_on:
                if dependency in positions and positions[dependency] > positions[node.id]:
                    issues.append(
                        ValidationIssue(
                            stage="graph",
                            code="invalid_topological_order",
                            node_id=node.id,
                            message=f"{dependency} must appear before {node.id}",
                        )
                    )
    except BlueprintGraphError as error:
        issues.append(
            ValidationIssue(
                stage="graph",
                code="invalid_graph",
                message=str(error),
            )
        )

    try:
        disconnected = find_disconnected_nodes(blueprint)  # type: ignore[arg-type]
        for node_id in sorted(disconnected):
            issues.append(
                ValidationIssue(
                    stage="graph",
                    code="disconnected_node",
                    node_id=node_id,
                    message=f"{node_id} does not contribute to ROOT",
                )
            )
    except BlueprintGraphError:
        pass

    normalized_statements: dict[str, str] = {}
    for node in blueprint.nodes:
        normalized = " ".join(node.informal_statement.casefold().split())
        if normalized in normalized_statements:
            issues.append(
                ValidationIssue(
                    stage="semantic",
                    code="duplicate_statement",
                    node_id=node.id,
                    message=(
                        f"{node.id} appears to duplicate "
                        f"{normalized_statements[normalized]}"
                    ),
                )
            )
        else:
            normalized_statements[normalized] = node.id
        if not 1 <= len(node.logical_ideas) <= 3:
            issues.append(
                ValidationIssue(
                    stage="decomposition",
                    code="invalid_logical_idea_count",
                    node_id=node.id,
                    message=(
                        f"{node.id} must list one to three new logical ideas; "
                        f"got {len(node.logical_ideas)}"
                    ),
                )
            )
        if any(not idea.strip() for idea in node.logical_ideas):
            issues.append(
                ValidationIssue(
                    stage="decomposition",
                    code="empty_logical_idea",
                    node_id=node.id,
                    message="logical_ideas entries must be non-empty",
                )
            )
        cited = set(re.findall(r"`(L[1-9][0-9]*)`", node.informal_proof))
        expected = set(node.depends_on)
        missing_citations = sorted(expected - cited)
        if missing_citations:
            issues.append(
                ValidationIssue(
                    stage="decomposition",
                    code="invalid_informal_proof_dependencies",
                    node_id=node.id,
                    message=(
                        "informal_proof must cite every declared dependency by "
                        "exact backticked ID: missing=" + str(missing_citations)
                    ),
                )
            )
    return issues


def validate_blueprint_plan(
    problem: TheoremProblem,
    plan: BlueprintPlan,
    *,
    max_nodes: int = 10,
) -> BlueprintValidationResult:
    """Validate the natural-language-only decomposition output."""

    issues = _validate_common_graph(plan, max_nodes=max_nodes)
    if plan.problem_hash and plan.problem_hash != problem.problem_hash:
        issues.append(
            ValidationIssue(
                stage="decomposition",
                code="problem_hash_mismatch",
                message="BlueprintPlan problem_hash differs from the problem",
            )
        )
    for node in plan.nodes:
        if node.lean_statement:
            issues.append(
                ValidationIssue(
                    stage="decomposition",
                    code="lean_statement_not_empty",
                    node_id=node.id,
                    message="Stage-one lean_statement must be empty",
                )
            )
        if problem.natural_language_statement:
            target = " ".join(problem.natural_language_statement.casefold().split())
            statement = " ".join(node.informal_statement.casefold().split())
            if statement == target:
                issues.append(
                    ValidationIssue(
                        stage="semantic",
                        code="target_restatement",
                        node_id=node.id,
                        message="Intermediate node exactly restates the target theorem",
                    )
                )
    return BlueprintValidationResult(valid=not issues, issues=issues)


def validate_blueprint(
    problem: TheoremProblem,
    blueprint: Blueprint,
    *,
    max_nodes: int = 10,
) -> BlueprintValidationResult:
    """Validate the complete formalized Blueprint independent of stage one."""

    issues = _validate_common_graph(blueprint, max_nodes=max_nodes)
    if blueprint.problem_hash and blueprint.problem_hash != problem.problem_hash:
        issues.append(
            ValidationIssue(
                stage="formalization",
                code="problem_hash_mismatch",
                message="Blueprint problem_hash differs from the problem",
            )
        )
    required_imports = set(problem.imports)
    for node in blueprint.nodes:
        declaration_match = re.match(r"^(?:theorem|lemma)\s+([^\s(:]+)", node.lean_decl)
        declaration_name = declaration_match.group(1) if declaration_match else ""
        if declaration_name != node.id:
            issues.append(
                ValidationIssue(
                    stage="formalization",
                    code="declaration_name_mismatch",
                    node_id=node.id,
                    message=(
                        f"Lean declaration name must be {node.id}, "
                        f"got {declaration_name or 'missing'}"
                    ),
                )
            )
        missing_imports = sorted(required_imports - set(node.preamble.imports))
        if missing_imports:
            issues.append(
                ValidationIssue(
                    stage="formalization",
                    code="incomplete_preamble_imports",
                    node_id=node.id,
                    message=f"Node preamble is missing required imports: {missing_imports}",
                )
            )
        if problem.target_lean_decl.strip() in node.lean_decl:
            issues.append(
                ValidationIssue(
                    stage="semantic",
                    code="target_restatement",
                    node_id=node.id,
                    message="Intermediate node appears to restate the target theorem",
                )
            )
    return BlueprintValidationResult(valid=not issues, issues=issues)


def validate_formalized_blueprint(
    problem: TheoremProblem,
    plan: BlueprintPlan,
    blueprint: Blueprint,
    environment: LeanEnvironmentIdentity,
    *,
    max_nodes: int = 10,
) -> BlueprintValidationResult:
    """Enforce that stage two only fills formalization fields of the frozen plan."""

    issues = list(validate_blueprint(problem, blueprint, max_nodes=max_nodes).issues)
    if blueprint.environment != environment:
        issues.append(
            ValidationIssue(
                stage="formalization",
                code="environment_identity_drift",
                message="Final Blueprint environment differs from the local environment",
            )
        )
    top_level_fields = (
        "blueprint_summary",
        "root_dependencies",
        "problem_hash",
        "warnings",
    )
    for field in top_level_fields:
        if getattr(blueprint, field) != getattr(plan, field):
            issues.append(
                ValidationIssue(
                    stage="formalization",
                    code="frozen_plan_drift",
                    message=f"Stage two changed frozen field {field}",
                )
            )
    if len(blueprint.nodes) != len(plan.nodes):
        issues.append(
            ValidationIssue(
                stage="formalization",
                code="node_count_drift",
                message=(
                    f"Frozen plan has {len(plan.nodes)} nodes; "
                    f"formalized Blueprint has {len(blueprint.nodes)}"
                ),
            )
        )
    frozen_fields = (
        "id",
        "title",
        "informal_statement",
        "informal_proof",
        "logical_ideas",
        "depends_on",
        "proof_strategy",
        "estimated_proof_length",
        "difficulty",
    )
    for index, (planned, formalized) in enumerate(
        zip(plan.nodes, blueprint.nodes, strict=False)
    ):
        for field in frozen_fields:
            if getattr(planned, field) != getattr(formalized, field):
                issues.append(
                    ValidationIssue(
                        stage="formalization",
                        code="frozen_node_drift",
                        node_id=formalized.id,
                        message=(
                            f"Stage two changed nodes[{index}].{field} "
                            f"from the frozen plan"
                        ),
                    )
                )
    return BlueprintValidationResult(valid=not issues, issues=issues)
