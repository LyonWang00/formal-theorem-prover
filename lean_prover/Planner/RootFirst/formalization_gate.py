"""Verify -> Pantograph -> retrieval repair -> Verify gate for new helpers."""

from __future__ import annotations

from typing import Any

from lean_prover.Planner.RootFirst.graph import topological_order
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    NodeStatus,
    TheoremProblem,
    ValidationIssue,
)
from lean_prover.Repair.planner_subproblem import PlannerSubproblemRepairer
from lean_prover.Verify import BlueprintVerifier

from .schemas import RootFirstBlueprint


class RootFirstFormalizationGate:
    """Apply the same semantic and compiler gate used by the whole Planner."""

    def __init__(
        self,
        *,
        verifier: BlueprintVerifier,
        lean_checker: BlueprintLeanChecker,
        repairer: PlannerSubproblemRepairer,
        max_repair_rounds: int = 2,
    ) -> None:
        if max_repair_rounds < 1:
            raise ValueError("max_repair_rounds must be positive")
        self.verifier = verifier
        self.lean_checker = lean_checker
        self.repairer = repairer
        self.max_repair_rounds = max_repair_rounds

    @staticmethod
    def _as_blueprint(source: RootFirstBlueprint) -> Blueprint:
        nodes = source.node_map()
        converted: list[BlueprintNode] = []
        for node_id in topological_order(source):
            if node_id == "L0":
                continue
            node = nodes[node_id]
            converted.append(
                BlueprintNode(
                    id=node.id,
                    title=node.title,
                    informal_statement=node.informal_statement,
                    informal_proof=node.informal_proof,
                    logical_ideas=list(node.logical_ideas),
                    lean_statement=node.lean_statement,
                    preamble=node.preamble.model_copy(deep=True),
                    semantic_alignment=node.semantic_alignment.model_copy(deep=True),
                    estimated_proof_length=node.estimated_proof_length.model_copy(
                        deep=True
                    ),
                    depends_on=list(node.father_nodes),
                    proof_strategy=node.proof_strategy,
                    mathlib_hints=list(node.mathlib_hints),
                    difficulty=3,
                    status=(
                        NodeStatus.PROVED
                        if node.verified_proof
                        else NodeStatus.PLANNED
                    ),
                    children=[item for item in node.children if item != "L0"],
                    verified_proof=node.verified_proof,
                    metadata=dict(node.metadata),
                )
            )
        return Blueprint(
            blueprint_summary="Program-owned RootFirst formalization gate.",
            nodes=converted,
            root_dependencies=[
                node.id for node in source.nodes if "L0" in node.children
            ],
            environment=source.environment,
            problem_hash=source.problem_hash,
            metadata={"root_first_gate": True},
        )

    @staticmethod
    def _copy_back(
        target: RootFirstBlueprint,
        checked: Blueprint,
        *,
        node_id: str,
    ) -> None:
        source_node = next(node for node in checked.nodes if node.id == node_id)
        target_node = target.node_map()[node_id]
        target_node.lean_statement = source_node.lean_decl
        target_node.preamble = source_node.preamble.model_copy(deep=True)
        target_node.semantic_alignment = source_node.semantic_alignment.model_copy(
            deep=True
        )
        target_node.mathlib_hints = list(source_node.mathlib_hints)
        # Preserve retrieval/compile repair history produced on the temporary
        # whole-Blueprint representation when copying the accepted value back
        # into the RootFirst node.
        target_node.metadata.update(dict(source_node.metadata))

    @staticmethod
    def _verification_issues(results: list[Any]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for result in results:
            for item in result.dependency_issues:
                issues.append(
                    ValidationIssue(
                        stage="verify",
                        code="dependency_verification_failed",
                        node_id=result.node_id,
                        message=item.message + " Repair reference: " + item.repair_reference,
                    )
                )
            for item in result.formal_statement_issues:
                issues.append(
                    ValidationIssue(
                        stage="verify",
                        code="formal_statement_verification_failed",
                        node_id=result.node_id,
                        message=item.message + " Repair reference: " + item.repair_reference,
                    )
                )
        return issues

    def validate_and_repair(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node_id: str,
    ) -> RootFirstBlueprint:
        updated = blueprint.model_copy(deep=True)
        formal = self._as_blueprint(updated)
        audit: list[dict[str, object]] = []
        last_issues: list[ValidationIssue] = []

        for gate_round in range(0, self.max_repair_rounds + 1):
            results = self.verifier.verify_blueprint(
                problem=problem,
                blueprint=formal,
                environment=updated.environment,
                node_ids={node_id},
            )
            formal_node = next(node for node in formal.nodes if node.id == node_id)
            result = results[0]
            formal_node.preamble = result.corrected_preamble.model_copy(deep=True)
            verify_issues = self._verification_issues(results)
            check = self.lean_checker.check_blueprint_detailed(problem, formal)
            compile_issues = [
                issue for issue in check.issues if issue.node_id == node_id
            ]
            last_issues = [*verify_issues, *compile_issues]
            semantic_patch_pending = bool(
                formal_node.metadata.pop("verify_semantic_patch_pending", False)
            )
            audit.append(
                {
                    "round": gate_round,
                    "verify": result.model_dump(mode="json"),
                    "pantograph_success": not compile_issues,
                    "pantograph_issues": [
                        issue.model_dump(mode="json") for issue in compile_issues
                    ],
                    "semantic_patch_pending": semantic_patch_pending,
                }
            )
            self._copy_back(updated, formal, node_id=node_id)
            if not last_issues:
                node = updated.node_map()[node_id]
                node.metadata["formalization_gate"] = audit
                node.metadata["formal_statement_verified"] = True
                node.metadata["dependency_statements_verified"] = True
                return updated
            if gate_round >= self.max_repair_rounds:
                break
            if semantic_patch_pending and not compile_issues:
                # Verify supplied the Lean value; compile passed. Re-enter
                # Verify without a generative repair, matching PlannerService.
                continue
            formal = self.repairer.repair(
                problem=problem,
                previous_blueprint=formal,
                issues=last_issues,
                environment=updated.environment,
                plan=None,
            )
            self._copy_back(updated, formal, node_id=node_id)

        message = " | ".join(issue.message for issue in last_issues)
        raise ValueError(
            f"{node_id}节点形式化失败，错误为：{message or 'unknown gate failure'}"
        )


__all__ = ["RootFirstFormalizationGate"]
