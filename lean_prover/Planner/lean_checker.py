"""Check lean declaration."""
from __future__ import annotations

from typing import Protocol

from .schemas import (
    Blueprint,
    BlueprintLeanCheckResult,
    LeanCheckResult,
    LeanEnvironmentIdentity,
    LeanFeedback,
    LeanPreamble,
    NodeLeanCheckResult,
    NodeStatus,
    TheoremProblem,
    ValidationIssue,
)


class DeclarationCheckingBackend(Protocol):
    def check_declaration(
        self,
        *,
        imports: list[str],
        declaration: str,
        preceding_declarations: list[str],
        preamble: LeanPreamble,
    ) -> LeanCheckResult:
        ...


class BlueprintLeanChecker:
    def __init__(
        self,
        backend: DeclarationCheckingBackend,
    ) -> None:
        self.backend = backend

    def environment_identity(
        self,
        problem: TheoremProblem,
    ) -> LeanEnvironmentIdentity:
        resolver = getattr(self.backend, "environment_identity", None)
        if not callable(resolver):
            raise RuntimeError(
                "Lean checking backend cannot resolve the local environment identity"
            )
        return resolver(problem.imports)

    @staticmethod
    def _ordered_imports(problem: TheoremProblem, preamble: LeanPreamble) -> list[str]:
        imports: list[str] = []
        for module in [*problem.imports, *preamble.imports]:
            if module not in imports:
                imports.append(module)
        return imports

    def check_target_declaration(
        self,
        problem: TheoremProblem,
    ) -> LeanCheckResult:
        """Compile the immutable/root target before node formalization."""

        preamble = LeanPreamble(
            imports=problem.imports,
            raw_header=problem.header,
        )
        return self.backend.check_declaration(
            imports=list(problem.imports),
            declaration=problem.target_lean_decl,
            preceding_declarations=list(problem.available_definitions),
            preamble=preamble,
        )

    def check_blueprint_detailed(
        self,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> BlueprintLeanCheckResult:
        """Compile every node and retain an explicit per-node atomic audit."""

        issues: list[ValidationIssue] = []
        node_results: list[NodeLeanCheckResult] = []
        accepted_declarations: dict[str, str] = {}

        for node in blueprint.nodes:
            dependency_declarations = [
                accepted_declarations[dependency]
                for dependency in node.depends_on
                if dependency in accepted_declarations
            ]
            result = self.backend.check_declaration(
                imports=self._ordered_imports(problem, node.preamble),
                declaration=node.lean_decl,
                preceding_declarations=[
                    *problem.available_definitions,
                    *dependency_declarations,
                ],
                preamble=node.preamble,
            )
            node_results.append(
                NodeLeanCheckResult(
                    node_id=node.id,
                    success=result.success,
                    lean_statement=node.lean_decl,
                    preamble=node.preamble,
                    result=result,
                )
            )
            feedback_message = (
                result.stdout
                if result.success
                else result.error_message or result.stderr or "Unknown Lean error"
            )
            node.lean_feedback.append(
                LeanFeedback(
                    stage="lean_statement_elaboration",
                    success=result.success,
                    message=feedback_message,
                    diagnostics=result.stderr,
                )
            )
            if result.success:
                node.status = NodeStatus.STATEMENT_VALID
                accepted_declarations[node.id] = node.lean_decl
                continue

            node.status = NodeStatus.FAILED
            issues.append(
                ValidationIssue(
                    stage="lean",
                    code="declaration_failed",
                    node_id=node.id,
                    message=feedback_message,
                )
            )

        failed_node_ids = [row.node_id for row in node_results if not row.success]
        return BlueprintLeanCheckResult(
            success=(
                len(node_results) == len(blueprint.nodes)
                and not failed_node_ids
            ),
            node_results=node_results,
            failed_node_ids=failed_node_ids,
            issues=issues,
        )

    def check_blueprint(
        self,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> list[ValidationIssue]:
        return self.check_blueprint_detailed(problem, blueprint).issues
