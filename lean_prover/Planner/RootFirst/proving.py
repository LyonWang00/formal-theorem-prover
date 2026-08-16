"""Pass@k proving with one shared node repair budget and formal disproof."""

from __future__ import annotations

import re
import uuid
from typing import Protocol

from lean_prover.Planner.schemas import (
    LeanFeedback,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.Prover.service import (
    DependencyContext,
    GeneratedProof,
    NodeProofRequest,
    ProofGenerator,
)
from lean_prover.Prover.repair_policy import (
    enforce_repair_grounding,
    proof_repair_eligible,
)

from .schemas import NodeState, RootFirstAttempt, RootFirstBlueprint, RootFirstNode


class ProofVerifier(Protocol):
    def verify(self, *, request: NodeProofRequest, proof: str) -> LeanFeedback:
        ...


class CandidateRepairer(Protocol):
    def generate_candidates(
        self,
        *,
        request: NodeProofRequest,
        failed_proof: str,
        feedback: LeanFeedback,
        round_index: int,
        repair_history: list[dict[str, object]],
    ) -> list[GeneratedProof]:
        ...


def _failed_feedback(message: str) -> LeanFeedback:
    return LeanFeedback(
        stage="proof_generation",
        success=False,
        message=message,
        diagnostics=message,
        error_type="internal_error",
        error_detail="internal_exception",
    )


def _top_level_signature_colon(declaration: str) -> int:
    """Find the declaration/result separator without parsing binder colons."""

    depth = 0
    in_string = False
    escaped = False
    candidates: list[int] = []
    for index, char in enumerate(declaration):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            candidates.append(index)
    if not candidates:
        raise ValueError("could not locate theorem conclusion separator")
    return candidates[-1]


def negated_declaration(node: RootFirstNode) -> str:
    """Construct the exact negation of the declaration's full Pi proposition."""

    declaration = node.lean_statement.strip()
    separator = _top_level_signature_colon(declaration)
    prefix = declaration[:separator].rstrip()
    conclusion = declaration[separator + 1 :].strip()
    match = re.match(r"^(?:theorem|lemma)\s+[^\s(:]+(?P<binders>.*)$", prefix, re.S)
    if match is None:
        raise ValueError("could not parse declaration name and binders")
    binders = match.group("binders").strip()
    proposition = f"∀ {binders}, {conclusion}" if binders else conclusion
    return f"theorem {node.id}_formal_disproof : ¬ ({proposition})"


class RootFirstProver:
    """Prove one node with pass@k followed by one shared repair budget."""

    def __init__(
        self,
        *,
        generator: ProofGenerator,
        verifier: ProofVerifier,
        repairer: CandidateRepairer | None = None,
        attempts_per_node: int = 4,
        max_repair_rounds: int = 3,
        max_repair_candidates_per_node: int | None = None,
        attempt_disproof: bool = True,
        complete_pass_batch: bool = False,
    ) -> None:
        if attempts_per_node < 1:
            raise ValueError("attempts_per_node must be positive")
        if max_repair_rounds < 1:
            raise ValueError("max_repair_rounds must be positive")
        if (
            max_repair_candidates_per_node is not None
            and max_repair_candidates_per_node < 1
        ):
            raise ValueError("max_repair_candidates_per_node must be positive")
        self.generator = generator
        self.verifier = verifier
        self.repairer = repairer
        self.attempts_per_node = attempts_per_node
        self.max_repair_rounds = max_repair_rounds
        self.max_repair_candidates_per_node = (
            max_repair_candidates_per_node
            if max_repair_candidates_per_node is not None
            else max_repair_rounds
        )
        self.attempt_disproof = attempt_disproof
        self.complete_pass_batch = complete_pass_batch

    @staticmethod
    def build_request(
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node: RootFirstNode,
        target_decl: str | None = None,
        disproof: bool = False,
    ) -> NodeProofRequest:
        nodes = blueprint.node_map()
        dependencies = tuple(
            DependencyContext(
                node_id=father_id,
                declaration=nodes[father_id].lean_statement,
                preamble=nodes[father_id].preamble,
                verified_proof=nodes[father_id].verified_proof,
            )
            for father_id in node.father_nodes
        )
        return NodeProofRequest(
            problem_id=problem.problem_id,
            problem_hash=problem.problem_hash,
            node_id=node.id + ("_DISPROOF" if disproof else ""),
            imports=tuple(dict.fromkeys([*problem.imports, *node.preamble.imports])),
            target_decl=target_decl or node.lean_statement,
            informal_statement=(
                "Formally prove the exact logical negation of: "
                + node.informal_statement
                if disproof
                else node.informal_statement
            ),
            informal_proof=(
                "Search for a compiler-verifiable contradiction or counterexample."
                if disproof
                else node.informal_proof
            ),
            logical_ideas=(
                ("Establish the exact negation under the same hypotheses",)
                if disproof
                else tuple(node.logical_ideas)
            ),
            proof_strategy=(
                "Prove only the program-constructed negated declaration."
                if disproof
                else node.proof_strategy
            ),
            available_definitions=tuple(problem.available_definitions),
            dependencies=dependencies,
            preamble=node.preamble,
            semantic_alignment=node.semantic_alignment,
            estimated_proof_length=node.estimated_proof_length,
            environment=blueprint.environment,
            mathlib_hints=tuple(node.mathlib_hints),
            dependency_statements_verified=True,
            formal_statement_verified=True,
        )

    @staticmethod
    def _attempt(
        *,
        index: int,
        kind: str,
        generated: GeneratedProof | None,
        feedback: LeanFeedback,
        proof_invocation: int,
        repair_round: int | None = None,
    ) -> RootFirstAttempt:
        metadata = dict(generated.metadata or {}) if generated else {}
        return RootFirstAttempt(
            attempt_id=f"RF-{uuid.uuid4().hex[:10]}",
            proof_invocation=proof_invocation,
            attempt_index=index,
            kind=kind,
            repair_round=repair_round,
            candidate_id=(
                str(metadata.get("candidate_id"))
                if metadata.get("candidate_id") is not None
                else None
            ),
            proof=generated.proof if generated else "",
            success=feedback.success,
            pantograph=feedback,
            model=generated.model if generated else "",
            metadata=metadata,
        )

    def _prove_request(
        self,
        *,
        request: NodeProofRequest,
        node: RootFirstNode,
        attempt_index: int,
        proof_invocation: int,
        disproof: bool = False,
    ) -> str | None:
        kind_prefix = "disproof" if disproof else "proof"
        try:
            generated = self.generator.generate(request)
            feedback = self.verifier.verify(request=request, proof=generated.proof)
        except Exception as error:
            generated = None
            feedback = _failed_feedback(str(error))
        node.attempts.append(
            self._attempt(
                index=attempt_index,
                kind=kind_prefix if generated else "generation_error",
                generated=generated,
                feedback=feedback,
                proof_invocation=proof_invocation,
            )
        )
        if feedback.success and generated is not None:
            return generated.proof
        node.last_diagnostics = feedback.diagnostics or feedback.message
        return None

    def _repair_node(
        self,
        *,
        request: NodeProofRequest,
        node: RootFirstNode,
        failed_attempt: RootFirstAttempt,
        proof_invocation: int,
        disproof: bool = False,
    ) -> str | None:
        """Run one shared, history-preserving repair budget for the node."""

        if self.repairer is None or not proof_repair_eligible(
            failed_attempt.pantograph
        ):
            return None
        failed_proof = failed_attempt.proof
        failed_feedback = failed_attempt.pantograph
        history: list[dict[str, object]] = []
        kind = "disproof_repair" if disproof else "proof_repair"
        candidates_checked = 0
        for repair_round in range(1, self.max_repair_rounds + 1):
            try:
                candidates = self.repairer.generate_candidates(
                    request=request,
                    failed_proof=failed_proof,
                    feedback=failed_feedback,
                    round_index=repair_round,
                    repair_history=history,
                )
            except Exception as error:
                node.last_diagnostics = str(error)
                break
            if not candidates:
                break
            if candidates_checked >= self.max_repair_candidates_per_node:
                break
            candidate = candidates[0]
            candidates_checked += 1
            candidate_feedback = self.verifier.verify(
                request=request,
                proof=candidate.proof,
            )
            candidate_feedback = enforce_repair_grounding(
                candidate_feedback,
                dict(candidate.metadata or {}),
            )
            node.attempts.append(
                self._attempt(
                    index=failed_attempt.attempt_index,
                    kind=kind,
                    generated=candidate,
                    feedback=candidate_feedback,
                    proof_invocation=proof_invocation,
                    repair_round=repair_round,
                )
            )
            outcome = {
                "candidate_id": (candidate.metadata or {}).get("candidate_id"),
                "success": candidate_feedback.success,
                "diagnostics": candidate_feedback.diagnostics,
                "error_detail": candidate_feedback.error_detail,
            }
            history.append(
                {"round": repair_round, "candidate_outcomes": [outcome]}
            )
            if candidate_feedback.success:
                node.metadata["proof_repair_history"] = history
                return candidate.proof
            failed_proof = candidate.proof
            failed_feedback = candidate_feedback
            if not proof_repair_eligible(candidate_feedback):
                break
        node.metadata["proof_repair_history"] = history
        node.last_diagnostics = failed_feedback.diagnostics or failed_feedback.message
        return None

    def prove_node(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node_id: str,
    ) -> RootFirstBlueprint:
        updated = blueprint.model_copy(deep=True)
        node = updated.node_map()[node_id]
        if node.state == NodeState.SUCCESS:
            # This guard is deliberately before request construction and model
            # invocation: a proved node is immutable proof context, never a new
            # proof-search target.
            return updated
        if any(
            updated.node_map()[father].state != NodeState.SUCCESS
            for father in node.father_nodes
        ):
            node.state = NodeState.FAILED
            node.last_diagnostics = "not all father nodes are proved"
            return updated
        node.metadata["proof_invocation_count"] = (
            int(node.metadata.get("proof_invocation_count", 0)) + 1
        )
        proof_invocation = int(node.metadata["proof_invocation_count"])
        request = self.build_request(problem=problem, blueprint=updated, node=node)
        first_verified_proof: str | None = None
        for attempt_index in range(1, self.attempts_per_node + 1):
            proof = self._prove_request(
                request=request,
                node=node,
                attempt_index=attempt_index,
                proof_invocation=proof_invocation,
            )
            if proof is not None and first_verified_proof is None:
                first_verified_proof = proof
                if not self.complete_pass_batch:
                    node.verified_proof = first_verified_proof
                    node.state = NodeState.SUCCESS
                    node.frozen = True
                    node.metadata["frozen_fingerprint"] = (
                        node.frozen_fingerprint()
                    )
                    node.metadata["attempt_count_at_freeze"] = len(node.attempts)
                    return updated
        if first_verified_proof is None:
            base_failures = [
                attempt
                for attempt in node.attempts
                if attempt.proof_invocation == proof_invocation
                and attempt.kind == "proof"
                and not attempt.success
                and proof_repair_eligible(attempt.pantograph)
            ]
            if base_failures:
                first_verified_proof = self._repair_node(
                    request=request,
                    node=node,
                    failed_attempt=base_failures[-1],
                    proof_invocation=proof_invocation,
                )
        if first_verified_proof is not None:
            # pass@k is an evaluation batch, not an early-stopping search.
            # Complete all independent base samples, then at most one shared
            # node-level compatibility-repair loop. A later refinement round
            # observes SUCCESS and never proves this node again.
            node.verified_proof = first_verified_proof
            node.state = NodeState.SUCCESS
            node.frozen = True
            node.metadata["frozen_fingerprint"] = node.frozen_fingerprint()
            node.metadata["attempt_count_at_freeze"] = len(node.attempts)
            return updated
        node.state = NodeState.FAILED
        node.metadata["diagnosis"] = "UNKNOWN"

        # A failed proof search is not evidence that a theorem is false.  Only
        # spend the formal-negation budget when Pantograph/model diagnostics
        # explicitly raise statement suspicion; every `disproved` state still
        # requires a compiled proof of the exact negation.
        suspicion = (
            "statement_wrong" in node.last_diagnostics.casefold()
            or "statement appears false" in node.last_diagnostics.casefold()
            or "counterexample" in node.last_diagnostics.casefold()
        )
        if self.attempt_disproof and suspicion:
            try:
                negated = negated_declaration(node)
                disproof_request = self.build_request(
                    problem=problem,
                    blueprint=updated,
                    node=node,
                    target_decl=negated,
                    disproof=True,
                )
                disproof = self._prove_request(
                    request=disproof_request,
                    node=node,
                    attempt_index=1,
                    proof_invocation=proof_invocation,
                    disproof=True,
                )
                if disproof is not None:
                    node.formal_disproof = disproof
                    node.state = NodeState.DISPROVED
                    node.frozen = False
                    node.metadata["diagnosis"] = "DISPROVED"
                    node.metadata["negated_lean_statement"] = negated
            except ValueError as error:
                node.metadata["disproof_skipped"] = str(error)
        return updated
