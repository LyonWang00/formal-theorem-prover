"""Root-first prove/refine state machine."""

from __future__ import annotations

from lean_prover.Planner.schemas import (
    LeanEnvironmentIdentity,
    LeanPreamble,
    ProofLengthEstimate,
    RawTheoremInput,
    SemanticAlignment,
    TheoremProblem,
)

from .graph import topological_order, validate_graph
from .effectiveness import dependency_usage, proof_mentions_name
from .proving import RootFirstProver
from .refinement import BlueprintRefinement
from .schemas import (
    NodeState,
    RefinementAudit,
    RootFirstBlueprint,
    RootFirstNode,
    RootFirstRunResult,
)


class RootFirstService:
    """Start at L0 and add at most one helper per refinement round."""

    def __init__(
        self,
        *,
        prover: RootFirstProver,
        refinement: BlueprintRefinement,
        environment: LeanEnvironmentIdentity,
        max_refinement_rounds: int = 3,
    ) -> None:
        if max_refinement_rounds != 3:
            raise ValueError("RootFirst max_refinement_rounds is fixed at 3")
        self.prover = prover
        self.refinement = refinement
        self.environment = environment
        self.max_refinement_rounds = max_refinement_rounds

    @staticmethod
    def problem_from_raw(raw: RawTheoremInput) -> TheoremProblem:
        target = (raw.formal_statement or raw.input_text).strip()
        if not target.startswith(("theorem ", "lemma ")):
            raise ValueError(
                "RootFirst currently requires a proof-free formal_statement"
            )
        return TheoremProblem(
            problem_id=raw.problem_id or "root_first_problem",
            imports=list(raw.imports or ["Mathlib"]),
            natural_language_statement=raw.informal_stmt,
            target_lean_decl=target,
            header=raw.header,
            available_definitions=list(raw.available_definitions),
            input_hash=raw.input_hash,
        )

    def initial_blueprint(self, problem: TheoremProblem) -> RootFirstBlueprint:
        root = RootFirstNode(
            id="L0",
            title="Original theorem target",
            informal_statement=(
                problem.natural_language_statement
                or "Prove the supplied immutable Lean theorem."
            ),
            informal_proof=(
                "First attempt a direct proof from the theorem's own hypotheses."
            ),
            logical_ideas=["Prove the original target directly"],
            lean_statement=problem.target_lean_decl,
            father_nodes=[],
            children=[],
            preamble=LeanPreamble(
                imports=list(problem.imports),
                raw_header=problem.header,
            ),
            semantic_alignment=SemanticAlignment(
                objects=["All objects in the original theorem"],
                hypotheses=["All hypotheses in the original theorem"],
                conclusion="The exact original theorem conclusion",
                alignment_notes="L0 is byte-identical to the input target.",
            ),
            estimated_proof_length=ProofLengthEstimate(
                estimated_lines=20,
                estimated_tokens=256,
                rationale="Direct root-first attempt before decomposition.",
            ),
            proof_strategy="Try the exact root directly before adding helpers.",
            state=NodeState.PENDING,
            metadata={"immutable_root": True},
        )
        blueprint = RootFirstBlueprint(
            problem_id=problem.problem_id,
            problem_hash=problem.problem_hash,
            original_target=problem.target_lean_decl,
            nodes=[root],
            environment=self.environment,
        )
        validate_graph(blueprint)
        return blueprint

    @staticmethod
    def _attempt_count(blueprint: RootFirstBlueprint) -> int:
        return sum(len(node.attempts) for node in blueprint.nodes)

    @staticmethod
    def _next_failure(blueprint: RootFirstBlueprint) -> str | None:
        nodes = blueprint.node_map()
        for node_id in topological_order(blueprint):
            node = nodes[node_id]
            if node.state in {NodeState.DISPROVED, NodeState.FAILED}:
                return node_id
        return None

    def _prove_open_frontier(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
    ) -> RootFirstBlueprint:
        updated = blueprint
        while True:
            nodes = updated.node_map()
            progress = False
            for node_id in topological_order(updated):
                node = nodes[node_id]
                if node.state != NodeState.PENDING:
                    continue
                if all(
                    nodes[father].state == NodeState.SUCCESS
                    for father in node.father_nodes
                ):
                    updated = self.prover.prove_node(
                        problem=problem,
                        blueprint=updated,
                        node_id=node_id,
                    )
                    progress = True
                    break
            if not progress:
                return updated

    @staticmethod
    def _assess_new_helpers(
        blueprint: RootFirstBlueprint,
        *,
        helper_ids: list[str],
        attempt_counts_before: dict[str, int],
        round_index: int,
    ) -> None:
        """Audit whether a compiled helper was explicitly used by its child."""

        nodes = blueprint.node_map()
        audit_log = blueprint.metadata.setdefault(
            "refinement_effectiveness", []
        )
        for helper_id in helper_ids:
            helper = nodes[helper_id]
            usage = dependency_usage(helper)
            children = list(helper.children)
            child_id = children[0] if len(children) == 1 else ""
            child = nodes.get(child_id)
            new_child_attempts = (
                child.attempts[attempt_counts_before.get(child_id, 0) :]
                if child is not None
                else []
            )
            referenced_attempt_ids = [
                attempt.attempt_id
                for attempt in new_child_attempts
                if proof_mentions_name(attempt.proof, helper_id)
            ]
            successful_attempt_ids = [
                attempt.attempt_id
                for attempt in new_child_attempts
                if attempt.success
            ]
            successful_references = sorted(
                set(referenced_attempt_ids) & set(successful_attempt_ids)
            )
            if helper.state != NodeState.SUCCESS:
                status = "helper_unproved"
                effective = False
            elif not bool(usage["all_declared_dependencies_used"]):
                status = "helper_ignored_declared_dependency"
                effective = False
            elif child is None or not new_child_attempts:
                status = "child_not_probed"
                effective = False
            elif successful_references:
                status = "child_proved_using_helper"
                effective = True
            elif successful_attempt_ids:
                status = "child_proved_without_helper"
                effective = False
            elif referenced_attempt_ids:
                status = "helper_referenced_child_failed"
                effective = False
            else:
                status = "helper_not_referenced_child_failed"
                effective = False
            record = {
                "round": round_index,
                "helper_id": helper_id,
                "child_id": child_id or None,
                "helper_state": helper.state.value,
                "dependency_usage": usage,
                "child_probe_attempt_ids": [
                    attempt.attempt_id for attempt in new_child_attempts
                ],
                "child_attempts_referencing_helper": referenced_attempt_ids,
                "successful_child_attempt_ids": successful_attempt_ids,
                "successful_child_attempts_referencing_helper": (
                    successful_references
                ),
                "status": status,
                "effective_refinement": effective,
            }
            helper.metadata["dependency_usage"] = usage
            helper.metadata["successor_effectiveness"] = record
            audit_log.append(record)
            if child is not None and not effective:
                feedback = (
                    f"Round {round_index} helper {helper_id} effectiveness gate: "
                    f"{status}. The next refinement must add a distinct, "
                    "nontrivial bridge that the child proof can explicitly cite."
                )
                child.metadata.setdefault("refinement_feedback", []).append(
                    feedback
                )

    def run_problem(self, problem: TheoremProblem) -> RootFirstRunResult:
        blueprint = self.initial_blueprint(problem)
        audits: list[RefinementAudit] = []

        # Direct L0 pass@k followed by the same shared node-level repair
        # policy used by both whole-Blueprint comparison arms.
        blueprint = self.prover.prove_node(
            problem=problem,
            blueprint=blueprint,
            node_id="L0",
        )
        if blueprint.node_map()["L0"].state == NodeState.SUCCESS:
            return RootFirstRunResult(
                success=True,
                stage="root_direct_success",
                problem_id=problem.problem_id,
                problem_hash=problem.problem_hash,
                blueprint=blueprint,
                total_attempts=self._attempt_count(blueprint),
            )
        if blueprint.node_map()["L0"].state == NodeState.DISPROVED:
            return RootFirstRunResult(
                success=False,
                stage="immutable_root_disproved",
                problem_id=problem.problem_id,
                problem_hash=problem.problem_hash,
                blueprint=blueprint,
                total_attempts=self._attempt_count(blueprint),
                warnings=[
                    "L0's exact negation compiled. Changing L0 would no longer "
                    "prove the input theorem, so refinement stops."
                ],
            )

        target_id = "L0"
        for round_index in range(1, self.max_refinement_rounds + 1):
            if blueprint.nodes_added >= self.max_refinement_rounds:
                break
            target = blueprint.node_map()[target_id]
            node_ids_before = set(blueprint.node_map())
            attempt_counts_before = {
                node.id: len(node.attempts) for node in blueprint.nodes
            }
            if target.state == NodeState.DISPROVED and target_id == "L0":
                break
            try:
                blueprint, round_audits = self.refinement.refine(
                    problem=problem,
                    blueprint=blueprint,
                    target_id=target_id,
                    round_index=round_index,
                )
            except Exception as error:
                return RootFirstRunResult(
                    success=False,
                    stage="refinement_failure",
                    problem_id=problem.problem_id,
                    problem_hash=problem.problem_hash,
                    blueprint=blueprint,
                    refinements=audits,
                    total_attempts=self._attempt_count(blueprint),
                    warnings=[str(error)],
                )
            audits.extend(round_audits)
            validate_graph(blueprint)
            blueprint = self._prove_open_frontier(
                problem=problem,
                blueprint=blueprint,
            )
            helper_ids = sorted(set(blueprint.node_map()) - node_ids_before)
            self._assess_new_helpers(
                blueprint,
                helper_ids=helper_ids,
                attempt_counts_before=attempt_counts_before,
                round_index=round_index,
            )
            root = blueprint.node_map()["L0"]
            if root.state == NodeState.SUCCESS:
                return RootFirstRunResult(
                    success=True,
                    stage="refined_success",
                    problem_id=problem.problem_id,
                    problem_hash=problem.problem_hash,
                    blueprint=blueprint,
                    refinements=audits,
                    total_attempts=self._attempt_count(blueprint),
                )
            target_id = self._next_failure(blueprint) or ""
            if not target_id:
                return RootFirstRunResult(
                    success=False,
                    stage="blocked_open_frontier",
                    problem_id=problem.problem_id,
                    problem_hash=problem.problem_hash,
                    blueprint=blueprint,
                    refinements=audits,
                    total_attempts=self._attempt_count(blueprint),
                    warnings=["No failed node found but L0 remains unsolved."],
                )
            if (
                blueprint.node_map()[target_id].state == NodeState.DISPROVED
                and target_id == "L0"
            ):
                return RootFirstRunResult(
                    success=False,
                    stage="immutable_root_disproved",
                    problem_id=problem.problem_id,
                    problem_hash=problem.problem_hash,
                    blueprint=blueprint,
                    refinements=audits,
                    total_attempts=self._attempt_count(blueprint),
                )

        return RootFirstRunResult(
            success=False,
            stage="refinement_budget_exhausted",
            problem_id=problem.problem_id,
            problem_hash=problem.problem_hash,
            blueprint=blueprint,
            refinements=audits,
            total_attempts=self._attempt_count(blueprint),
        )

    def run_input(self, raw: RawTheoremInput) -> RootFirstRunResult:
        return self.run_problem(self.problem_from_raw(raw))
