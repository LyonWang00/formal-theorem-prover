"""One-node-at-a-time Blueprint refinement with program-owned assembly."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from lean_prover.Planner.schemas import (
    LeanCheckResult,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)
from lean_prover.model_api import MAX_EMPTY_RESPONSE_ATTEMPTS

from .effectiveness import helper_quality_errors
from .graph import graph_errors
from .schemas import (
    NodeState,
    RefinementAudit,
    RefinementPatch,
    RootFirstBlueprint,
    RootFirstNode,
)


class RefinementClient(Protocol):
    def generate_text_or_json(self, **kwargs: Any) -> dict[str, Any] | str:
        ...


class StatementChecker(Protocol):
    def check(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node: RootFirstNode,
    ) -> LeanCheckResult:
        ...


class FormalizationGate(Protocol):
    def validate_and_repair(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        node_id: str,
    ) -> RootFirstBlueprint:
        ...


REFINEMENT_SYSTEM_PROMPT = r"""
You are BlueprintRefinement for a root-first Lean proof pipeline. Return exactly
one JSON object matching the requested patch schema. The program, not you,
constructs the final Blueprint.

For state=failed, perform exactly one local decomposition: add exactly one new
node immediately before target_node_id. The new node's `children` must be the
one-element list [target_node_id]. Its `father_nodes` must be the currently
proved facts that it explicitly uses. If the failed target currently depends
on one or more proved fathers, the new node normally sits between them and the
target. The program will replace those direct father->target edges with
father->new_node->target. If the target has no fathers, use father_nodes=[].

For state=disproved, do not add any node. Revise only the target node's
mathematical statement so that it is a correct useful lemma under its existing
context. Return action=revise_disproved_node and revised_node. Never alter L0's
immutable original target; a formal disproof of L0 is reported as a terminal
inconsistency instead of being revised.

For each new node return ONLY id, informal_statement, lean_statement,
father_nodes, and children. For a revised node return ONLY informal_statement
and lean_statement. The program constructs titles, proof hints, dependency
contexts, preambles, graph updates, and all runtime fields. Do not output those
program-owned values. lean_statement is a proof-free declaration: no :=, by,
sorry, admit, axiom, unsafe, tactics, comments, or markdown. Do not return an
entire Blueprint, modify successful frozen nodes, or add more than one node.

The new helper must be mathematically nontrivial and add a genuine bridge. Its
conclusion must not repeat one of its own hypotheses, and its Lean or informal
proposition must not duplicate an existing node (renaming only the declaration
does not make it new). State one local fact that can materially advance its
child; do not create tautologies or merely restate the target.

Keep the patch compact. Return only the completed JSON fields, never hidden
reasoning, scratch work, repeated paragraphs, or an essay. Limits:
informal_statement 1200; lean_statement 2400. If your first idea is uncertain, still choose one
concrete locally useful lemma and finish the JSON instead of continuing to
deliberate in the output.

Few-shot: first failure of L0
{
  "state":"failed",
  "action":"add_node",
  "target_node_id":"L0",
  "new_node":{
    "id":"L1",
    "informal_statement":"Under the root hypotheses, the expression equals -39.",
    "lean_statement":"lemma L1 (a b : ℝ) (ha : a = -1) (hb : b = 5) : -a - b^2 + 3 * (a*b) = -39",
    "father_nodes":[],
    "children":["L0"]
  },
  "revised_node":null
}

Few-shot: L1 is proved but L0 still fails
{
  "state":"failed",
  "action":"add_node",
  "target_node_id":"L0",
  "new_node":{
    "id":"L2",
    "informal_statement":"The exact target follows from the evaluated expression.",
    "lean_statement":"lemma L2 (a b : ℝ) (ha : a = -1) (hb : b = 5) : -a - b^2 + 3 * (a*b) = -39",
    "father_nodes":["L1"],
    "children":["L0"]
  },
  "revised_node":null
}

Few-shot: a non-root node L2 has a Pantograph-verified disproof
{
  "state":"disproved",
  "action":"revise_disproved_node",
  "target_node_id":"L2",
  "new_node":null,
  "revised_node":{
    "informal_statement":"Under the stated hypotheses, the weaker bridge equality holds.",
    "lean_statement":"lemma L2 (x : ℝ) (h : x = 1) : x = 1"
  }
}
""".strip()


class BlueprintRefinement:
    def __init__(
        self,
        *,
        client: RefinementClient,
        statement_checker: StatementChecker,
        formalization_gate: FormalizationGate | None = None,
        max_generation_attempts: int | None = None,
    ) -> None:
        self.client = client
        self.statement_checker = statement_checker
        self.formalization_gate = formalization_gate
        self.max_generation_attempts = (
            max_generation_attempts or MAX_EMPTY_RESPONSE_ATTEMPTS
        )

    @staticmethod
    def _next_id(blueprint: RootFirstBlueprint) -> str:
        used = {int(node.id[1:]) for node in blueprint.nodes}
        candidate = 1
        while candidate in used:
            candidate += 1
        return f"L{candidate}"

    @staticmethod
    def _prompt(
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        target_id: str,
        expected_new_id: str,
        feedback_history: list[str],
    ) -> str:
        target = blueprint.node_map()[target_id]
        compact_nodes = [
            {
                "id": node.id,
                "state": node.state.value,
                "frozen": node.frozen,
                "informal_statement": node.informal_statement,
                "informal_proof": node.informal_proof,
                "lean_statement": node.lean_statement,
                "father_nodes": node.father_nodes,
                "children": node.children,
                "verified_proof_available": bool(node.verified_proof),
                "last_diagnostics": node.last_diagnostics,
                "refinement_feedback": node.metadata.get(
                    "refinement_feedback", []
                ),
            }
            for node in blueprint.nodes
        ]
        return f"""
Original immutable theorem target:
{problem.target_lean_decl}

Natural-language problem:
{problem.natural_language_statement or "(not supplied)"}

Current complete RootFirst Blueprint:
{json.dumps(compact_nodes, ensure_ascii=False, indent=2)}

Target node to refine: {target_id}
Target state: {target.state.value}
Target Pantograph/prover diagnostics:
{target.last_diagnostics}

For an add_node action, the only permitted new ID is {expected_new_id}.
The only permitted child is {target_id}. father_nodes must be selected from the
target's current fathers that are state=success, namely:
{json.dumps([father for father in target.father_nodes if blueprint.node_map()[father].state == NodeState.SUCCESS], ensure_ascii=False)}

Program/graph/compiler rejection history for this SAME refinement round:
{json.dumps(feedback_history, ensure_ascii=False, indent=2)}

Return the concrete patch now. If a prior candidate was rejected, actually fix
the reported field; do not merely repeat or describe the error.
""".strip()

    @staticmethod
    def _make_node(
        *,
        values: Any,
        target: RootFirstNode,
    ) -> RootFirstNode:
        return RootFirstNode(
            id=values.id,
            title=f"Refinement helper {values.id}",
            informal_statement=values.informal_statement,
            informal_proof=(
                "Use the verified predecessor propositions "
                + ", ".join(f"`{item}`" for item in values.father_nodes)
                + " as exact available facts, then establish the stated conclusion."
                if values.father_nodes
                else "Establish the stated conclusion directly from its explicit hypotheses."
            ),
            logical_ideas=["Establish the exact stated local proposition"],
            lean_statement=values.lean_statement,
            father_nodes=values.father_nodes,
            children=values.children,
            preamble=target.preamble.model_copy(deep=True),
            semantic_alignment=SemanticAlignment(
                objects=["Objects and binders in the generated Lean statement"],
                hypotheses=["Hypotheses in the generated Lean statement"],
                conclusion=values.informal_statement,
                alignment_notes="RootFirst program-owned refinement template.",
            ),
            estimated_proof_length=ProofLengthEstimate(
                estimated_lines=12,
                estimated_tokens=160,
                rationale="One local refinement node with at most three ideas.",
            ),
            proof_strategy="Use exact hypotheses and verified predecessor declarations.",
            mathlib_hints=[],
        )

    @staticmethod
    def _declaration_name(statement: str) -> str:
        match = re.match(r"^(?:theorem|lemma)\s+([^\s(:]+)", statement.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _check_frozen(before: RootFirstBlueprint, after: RootFirstBlueprint) -> list[str]:
        errors: list[str] = []
        after_nodes = after.node_map()
        for node in before.nodes:
            if not node.frozen:
                continue
            candidate = after_nodes.get(node.id)
            if candidate is None:
                errors.append(f"frozen node {node.id} was deleted")
            elif candidate.frozen_fingerprint() != node.frozen_fingerprint():
                errors.append(f"frozen content or proof changed for {node.id}")
        return errors

    def _apply(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        patch: RefinementPatch,
        expected_new_id: str,
    ) -> tuple[RootFirstBlueprint, list[str], str]:
        updated = blueprint.model_copy(deep=True)
        nodes = updated.node_map()
        target = nodes.get(patch.target_node_id)
        if target is None:
            return updated, [f"unknown target {patch.target_node_id}"], ""
        errors: list[str] = []
        statement_error = ""
        if patch.state != target.state.value:
            errors.append(
                f"patch state {patch.state} differs from target state {target.state.value}"
            )

        if patch.action == "add_node" and patch.new_node is not None:
            values = patch.new_node
            if values.id != expected_new_id:
                errors.append(
                    f"new node ID must be {expected_new_id}, got {values.id}"
                )
            if values.children != [target.id]:
                errors.append(
                    f"new node children must be exactly ['{target.id}']"
                )
            eligible_fathers = {
                father
                for father in target.father_nodes
                if nodes[father].state == NodeState.SUCCESS
            }
            if set(values.father_nodes) != eligible_fathers:
                errors.append(
                    "new node fathers must be exactly all proved current fathers "
                    "of target, so it is inserted between them and the target; "
                    f"eligible={sorted(eligible_fathers)}"
                )
            if self._declaration_name(values.lean_statement) != values.id:
                errors.append(
                    f"new Lean declaration name must be {values.id}"
                )
            if errors:
                return updated, errors, statement_error
            new_node = self._make_node(values=values, target=target)
            errors.extend(helper_quality_errors(new_node, blueprint.nodes))
            if errors:
                return updated, errors, statement_error
            updated.nodes.append(new_node)
            # Insert the node between every selected father and the target.
            for father_id in values.father_nodes:
                father = nodes[father_id]
                father.children = [
                    new_node.id if child == target.id else child
                    for child in father.children
                ]
                target.father_nodes = [
                    father for father in target.father_nodes if father != father_id
                ]
            if new_node.id not in target.father_nodes:
                target.father_nodes.append(new_node.id)
            target.state = NodeState.PENDING
            target.last_diagnostics = ""
            updated.nodes_added += 1
        elif patch.action == "revise_disproved_node" and patch.revised_node is not None:
            if target.id == "L0":
                errors.append("L0 is immutable and cannot be revised after disproof")
                return updated, errors, statement_error
            values = patch.revised_node
            if self._declaration_name(values.lean_statement) != target.id:
                errors.append(
                    f"revised Lean declaration name must remain {target.id}"
                )
                return updated, errors, statement_error
            target.informal_statement = values.informal_statement
            target.lean_statement = values.lean_statement
            target.state = NodeState.PENDING
            target.formal_disproof = None
            target.last_diagnostics = ""

        errors.extend(graph_errors(updated))
        errors.extend(self._check_frozen(blueprint, updated))
        if not errors:
            try:
                if self.formalization_gate is not None:
                    updated = self.formalization_gate.validate_and_repair(
                        problem=problem,
                        blueprint=updated,
                        node_id=patch.target_node_id if patch.action != "add_node" else expected_new_id,
                    )
                    if patch.action == "add_node":
                        errors.extend(
                            helper_quality_errors(
                                updated.node_map()[expected_new_id],
                                blueprint.nodes,
                            )
                        )
                else:
                    checked_node = updated.node_map()[
                        patch.target_node_id if patch.action != "add_node" else expected_new_id
                    ]
                    check = self.statement_checker.check(
                        problem=problem,
                        blueprint=updated,
                        node=checked_node,
                    )
                    if not check.success:
                        statement_error = (
                            check.error_message or check.stderr or check.stdout
                        )
            except Exception as error:
                statement_error = str(error)
        return updated, errors, statement_error

    def refine(
        self,
        *,
        problem: TheoremProblem,
        blueprint: RootFirstBlueprint,
        target_id: str,
        round_index: int,
    ) -> tuple[RootFirstBlueprint, list[RefinementAudit]]:
        target = blueprint.node_map()[target_id]
        if target.state not in {NodeState.FAILED, NodeState.DISPROVED}:
            raise ValueError("refinement target must be failed or disproved")
        expected_new_id = self._next_id(blueprint)
        feedback_history: list[str] = []
        audits: list[RefinementAudit] = []
        for generation_attempt in range(1, self.max_generation_attempts + 1):
            raw: dict[str, Any] | str | None = None
            try:
                raw = self.client.generate_text_or_json(
                    system_prompt=REFINEMENT_SYSTEM_PROMPT,
                    user_prompt=self._prompt(
                        problem=problem,
                        blueprint=blueprint,
                        target_id=target_id,
                        expected_new_id=expected_new_id,
                        feedback_history=feedback_history,
                    ),
                    empty_response_message="BlueprintRefinement模型空响应",
                    history=None,
                )
                if not isinstance(raw, dict):
                    raise ValueError(
                        "response is nonempty but is not one valid JSON object"
                    )
                patch = RefinementPatch.model_validate(raw)
            except (ValidationError, ValueError, TypeError) as error:
                feedback_history.append(f"schema error: {error}")
                audits.append(
                    RefinementAudit(
                        round_index=round_index,
                        generation_attempt=generation_attempt,
                        target_node_id=target_id,
                        input_state=target.state,
                        graph_errors=[feedback_history[-1]],
                        raw_output=raw,
                    )
                )
                continue
            candidate, errors, statement_error = self._apply(
                problem=problem,
                blueprint=blueprint,
                patch=patch,
                expected_new_id=expected_new_id,
            )
            audit = RefinementAudit(
                round_index=round_index,
                generation_attempt=generation_attempt,
                target_node_id=target_id,
                input_state=target.state,
                patch=patch,
                graph_errors=errors,
                statement_error=statement_error,
                accepted=not errors and not statement_error,
                raw_output=raw,
            )
            audits.append(audit)
            if audit.accepted:
                candidate.refinement_round = round_index
                return candidate, audits
            feedback_history.extend(errors)
            if statement_error:
                feedback_history.append("Pantograph statement error: " + statement_error)
        raise ValueError(
            "BlueprintRefinement exhausted generation attempts: "
            + " | ".join(feedback_history)
        )
