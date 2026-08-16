"""Retrieval-guided repair of Planner subproblem formalizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator

from lean_prover.Mathlib import (
    CompilerErrorRetriever,
    EnvironmentDeclarationIndex,
)
from lean_prover.Planner.client import LLMClient
from lean_prover.Planner.lean_checker import BlueprintLeanChecker
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintPlan,
    LeanEnvironmentIdentity,
    LeanPreamble,
    SemanticAlignment,
    TheoremProblem,
    ValidationIssue,
)


class FormalizationNodePatch(BaseModel):
    node_id: str
    lean_decl: str = Field(
        validation_alias=AliasChoices("lean_statement", "lean_decl"),
        serialization_alias="lean_statement",
    )
    preamble: LeanPreamble
    semantic_alignment: SemanticAlignment
    mathlib_hints: list[str] = Field(default_factory=list)

    @field_validator("lean_decl")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        # Reuse the exact BlueprintNode statement contract while allowing a
        # small patch schema instead of duplicating an entire Blueprint.
        from lean_prover.Planner.schemas import BlueprintNode

        return BlueprintNode.validate_lean_decl(value)


class FormalizationRepairCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    node_patches: list[FormalizationNodePatch] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class FormalizationRepairCandidates(BaseModel):
    candidates: list[FormalizationRepairCandidate] = Field(
        min_length=1,
        max_length=1,
    )


_CANDIDATE_FEW_SHOT = """
Output-format example only (the names below are illustrative, not API facts):
{
  "candidates": [
    {
      "candidate_id": "C1",
      "node_patches": [
        {
          "node_id": "L1",
          "lean_statement": "lemma L1 (n : Nat) : n + 0 = n",
          "preamble": {
            "imports": ["Mathlib.Data.Nat.Basic"],
            "raw_header": "",
            "namespaces": [],
            "open_namespaces": [],
            "open_scoped": [],
            "variable_declarations": [],
            "local_context": []
          },
          "semantic_alignment": {
            "objects": ["n is a natural number"],
            "hypotheses": [],
            "conclusion": "adding zero to n gives n",
            "alignment_notes": "Nat and addition match the informal statement exactly."
          },
          "mathlib_hints": ["Nat.add_zero"]
        }
      ],
      "rationale": "Use the indexed current-environment declaration."
    }
  ]
}
""".strip()

FORMALIZATION_REPAIR_SYSTEM_PROMPT = """
You repair Lean 4 declaration statements against one exact local Mathlib
environment. Return exactly one JSON object with a `candidates` array matching
the user-provided few-shot schema. Use only locally indexed declaration facts.
Do not return a complete Blueprint, markdown, prose outside JSON, or proof
tactics. Mathematical semantics, the dataset header, DAG dependencies,
informal_proof, and logical_ideas are immutable. Do not judge whether a
dependency is mathematically necessary, sufficient, or logically correct.
""".strip()


def _dependency_payload(blueprint: Blueprint, node_id: str) -> list[dict[str, str]]:
    nodes = {node.id: node for node in blueprint.nodes}
    node = nodes[node_id]
    return [
        {
            "node_id": dependency,
            "lean_statement": nodes[dependency].lean_decl,
            "informal_statement": nodes[dependency].informal_statement,
        }
        for dependency in node.depends_on
        if dependency in nodes
    ]


def _repair_prompt(
    *,
    problem: TheoremProblem,
    environment: LeanEnvironmentIdentity,
    previous: Blueprint,
    issues: list[ValidationIssue],
    failed_ids: list[str],
    retrieval: list[dict[str, object]],
) -> str:
    nodes = {node.id: node for node in previous.nodes}
    failed_nodes = [
        {
            "node_id": node_id,
            "original_natural_language_statement": nodes[node_id].informal_statement,
            "immutable_natural_language_proof": nodes[node_id].informal_proof,
            "immutable_logical_ideas": nodes[node_id].logical_ideas,
            "current_error_lean_statement": nodes[node_id].lean_decl,
            "current_preamble": nodes[node_id].preamble.model_dump(mode="json"),
            "current_dependency_propositions": _dependency_payload(previous, node_id),
        }
        for node_id in failed_ids
        if node_id in nodes
    ]
    return f"""
Repair only the failed Lean declaration statements by a strict
retrieval-generate-compile process. Generate exactly one small candidate in the
specified JSON format; every candidate must patch every failed node. The
caller will compile candidates one by one with Pantograph and will send the
best compiler result into the next repair round if none pass.

You may use only declaration names shown in `local_index_retrieval`, or names
already present in the immutable problem/Blueprint. Never invent a Mathlib
name, guessed import, field API, deprecated syntax, proof, tactic, `:=`, `by`,
`sorry`, `admit`, comment, or placeholder. Candidate imports must come from
the retrieved records or the existing preamble. Preserve all mathematical
objects, binder types, hypotheses and the exact conclusion; API adaptation
must not strengthen, weaken, reverse, or replace the proposition with a
supposedly equivalent statement. Preserve node IDs and graph. The dataset
header in raw_header is authoritative and must remain unchanged. Do not audit
or alter the mathematical/logical adequacy of dependencies; they are supplied
only as exact Lean context for compilation.

Exact local Lean/Mathlib identity (the cache key uses mathlib_commit and
environment_hash):
{environment.model_dump_json(indent=2)}

Original root natural-language statement:
{problem.natural_language_statement or "(not supplied)"}

Failed nodes, their current statements, preambles and exact dependencies:
{json.dumps(failed_nodes, ensure_ascii=False, indent=2)}

Full Pantograph/Verify/static errors:
{json.dumps([issue.model_dump(mode="json") for issue in issues], ensure_ascii=False, indent=2)}

Previous formalization repair and candidate-compilation history (do not repeat
an unchanged failed candidate):
{json.dumps(previous.metadata.get("formalization_repair", []), ensure_ascii=False, indent=2)}

Local index retrieval (every candidate is a real declaration from this exact
checkout, including signature, required import and source location):
{json.dumps(retrieval, ensure_ascii=False, indent=2)}

{_CANDIDATE_FEW_SHOT}
""".strip()


class PlannerSubproblemRepairer:
    """Generate indexed candidates and compile them before returning one."""

    def __init__(
        self,
        client: LLMClient,
        *,
        lean_checker: BlueprintLeanChecker | None = None,
        project_path: str | Path | None = None,
    ) -> None:
        self.client = client
        self.lean_checker = lean_checker
        backend = getattr(lean_checker, "backend", None)
        self.project_path = Path(
            project_path or getattr(backend, "project_path", "")
        ) if (project_path or getattr(backend, "project_path", None)) else None
        self._indices: dict[str, EnvironmentDeclarationIndex] = {}

    def _retriever(
        self,
        environment: LeanEnvironmentIdentity,
    ) -> CompilerErrorRetriever | None:
        if self.project_path is None:
            return None
        key = f"{environment.mathlib_commit}/{environment.environment_hash}"
        if key not in self._indices:
            self._indices[key] = EnvironmentDeclarationIndex.for_project(
                project_path=self.project_path,
                environment=environment,
            )
        return CompilerErrorRetriever(self._indices[key])

    def _retrieval_payload(
        self,
        *,
        problem: TheoremProblem,
        previous: Blueprint,
        issues: list[ValidationIssue],
        environment: LeanEnvironmentIdentity,
    ) -> list[dict[str, object]]:
        retriever = self._retriever(environment)
        if retriever is None:
            return []
        nodes = {node.id: node for node in previous.nodes}
        payload: list[dict[str, object]] = []
        grouped: dict[str | None, list[ValidationIssue]] = {}
        for issue in issues:
            grouped.setdefault(issue.node_id, []).append(issue)
        for node_id, node_issues in grouped.items():
            node = nodes.get(node_id or "")
            imports = list(problem.imports)
            if node is not None:
                imports.extend(node.preamble.imports)
            analyses: list[dict[str, object]] = []
            candidates: dict[tuple[str, str, int], dict[str, object]] = {}
            for issue in node_issues:
                context = retriever.retrieve(
                    issue.message,
                    current_imports=imports,
                    statement=node.lean_decl if node else "",
                    informal_statement=node.informal_statement if node else (
                        problem.natural_language_statement or ""
                    ),
                    limit=10,
                )
                analyses.append(context.analysis.prompt_record())
                for candidate in context.candidates:
                    declaration = candidate.declaration
                    key = (
                        declaration.full_name,
                        declaration.module,
                        declaration.source_line,
                    )
                    record = candidate.prompt_record()
                    previous_record = candidates.get(key)
                    if (
                        previous_record is None
                        or float(record["score"])
                        > float(previous_record["score"])
                    ):
                        candidates[key] = record
            ranked = sorted(
                candidates.values(),
                key=lambda item: float(item["score"]),
                reverse=True,
            )[:10]
            payload.append(
                {
                    "node_id": node_id,
                    "issues": [
                        issue.model_dump(mode="json") for issue in node_issues
                    ],
                    "compiler_error_analyses": analyses,
                    "candidate_declarations": ranked,
                }
            )
        return payload

    @staticmethod
    def _allowed_candidate_facts(
        previous: Blueprint,
        retrieval: list[dict[str, object]],
    ) -> tuple[set[str], set[str]]:
        imports = {
            module
            for node in previous.nodes
            for module in node.preamble.imports
        }
        names = {
            hint
            for node in previous.nodes
            for hint in node.mathlib_hints
        }
        for group in retrieval:
            for raw in group.get("candidate_declarations", []):
                if not isinstance(raw, dict):
                    continue
                required_import = raw.get("required_import")
                full_name = raw.get("full_name")
                if isinstance(required_import, str) and required_import:
                    imports.add(required_import)
                if isinstance(full_name, str):
                    names.add(full_name)
        return imports, names

    @staticmethod
    def _assert_unaffected_nodes_frozen(
        before: Blueprint,
        after: Blueprint,
        failed_node_ids: set[str],
    ) -> None:
        before_nodes = {node.id: node for node in before.nodes}
        after_nodes = {node.id: node for node in after.nodes}
        if set(before_nodes) != set(after_nodes):
            raise ValueError("Planner repair changed Blueprint node IDs")
        for node_id, before_node in before_nodes.items():
            if node_id in failed_node_ids:
                continue
            if before_node.model_dump(mode="json", by_alias=True) != after_nodes[
                node_id
            ].model_dump(mode="json", by_alias=True):
                raise ValueError(
                    f"Planner repair changed unaffected subproblem {node_id}"
                )

    @staticmethod
    def _apply_candidate(
        previous: Blueprint,
        candidate: FormalizationRepairCandidate,
        failed_ids: set[str],
        *,
        allowed_imports: set[str],
        allowed_names: set[str],
    ) -> Blueprint:
        patched = previous.model_copy(deep=True)
        nodes = {node.id: node for node in patched.nodes}
        patch_ids = {patch.node_id for patch in candidate.node_patches}
        if patch_ids != failed_ids:
            raise ValueError(
                "each formalization candidate must patch exactly the failed "
                f"nodes: expected={sorted(failed_ids)}, got={sorted(patch_ids)}"
            )
        for patch in candidate.node_patches:
            node = nodes.get(patch.node_id)
            if node is None:
                raise ValueError(f"unknown repaired node ID {patch.node_id}")
            original_header = node.preamble.raw_header
            unexpected_imports = set(patch.preamble.imports) - allowed_imports
            if unexpected_imports:
                raise ValueError(
                    "formalization repair invented imports not present in the "
                    f"local index: {sorted(unexpected_imports)}"
                )
            unexpected_hints = set(patch.mathlib_hints) - allowed_names
            if unexpected_hints:
                raise ValueError(
                    "formalization repair invented Mathlib hints not present in "
                    f"the local index: {sorted(unexpected_hints)}"
                )
            node.lean_decl = patch.lean_decl
            node.preamble = patch.preamble.model_copy(deep=True)
            node.preamble.raw_header = original_header
            node.semantic_alignment = patch.semantic_alignment.model_copy(deep=True)
            node.mathlib_hints = list(patch.mathlib_hints)
            node.dependency_statements_verified = None
            node.formal_statement_verified = None
            node.verification_issues = []
            node.verification_notes = ""
        return patched

    @staticmethod
    def _legacy_candidate(
        previous: Blueprint,
        raw: dict[str, Any],
        failed_ids: set[str],
    ) -> Blueprint:
        repaired = Blueprint.model_validate(raw)
        before_nodes = {node.id: node for node in previous.nodes}
        for node in repaired.nodes:
            before = before_nodes.get(node.id)
            if before is None:
                continue
            node.preamble.raw_header = before.preamble.raw_header
            if node.id in failed_ids:
                node.dependency_statements_verified = None
                node.formal_statement_verified = None
                node.verification_issues = []
                node.verification_notes = ""
        return repaired

    def repair(
        self,
        *,
        problem: TheoremProblem,
        previous_blueprint: Blueprint | dict,
        issues: list[ValidationIssue],
        environment: LeanEnvironmentIdentity,
        plan: BlueprintPlan | None = None,
    ) -> Blueprint:
        previous = Blueprint.model_validate(previous_blueprint)
        failed_ids = sorted(
            {issue.node_id for issue in issues if issue.node_id is not None}
        )
        if not failed_ids:
            failed_ids = [node.id for node in previous.nodes]
        failed_set = set(failed_ids)
        retrieval = self._retrieval_payload(
            problem=problem,
            previous=previous,
            issues=issues,
            environment=environment,
        )
        prompt = _repair_prompt(
            problem=problem,
            environment=environment,
            previous=previous,
            issues=issues,
            failed_ids=failed_ids,
            retrieval=retrieval,
        )
        raw = self.client.generate_json(
            system_prompt=FORMALIZATION_REPAIR_SYSTEM_PROMPT,
            user_prompt=prompt,
            empty_response_message=(
                (
                    "修复形式化节点"
                    + "、".join(failed_ids)
                    + "时模型空响应"
                )
                if failed_ids
                else "修复分解模型空响应"
            ),
            history=[
                {
                    "role": "assistant",
                    "content": previous.model_dump_json(by_alias=True),
                },
                {
                    "role": "user",
                    "content": "Pantograph/Verify feedback:\n"
                    + json.dumps(
                        [issue.model_dump(mode="json") for issue in issues],
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        )

        candidates: list[tuple[str, Blueprint]] = []
        if isinstance(raw, dict) and "candidates" in raw:
            generated = FormalizationRepairCandidates.model_validate(raw)
            allowed_imports, allowed_names = self._allowed_candidate_facts(
                previous,
                retrieval,
            )
            allowed_imports.update(problem.imports)
            for candidate in generated.candidates:
                candidates.append(
                    (
                        candidate.candidate_id,
                        self._apply_candidate(
                            previous,
                            candidate,
                            failed_set,
                            allowed_imports=allowed_imports,
                            allowed_names=allowed_names,
                        ),
                    )
                )
        else:
            candidates.append(
                ("legacy_full_blueprint", self._legacy_candidate(previous, raw, failed_set))
            )

        compile_audit: list[dict[str, object]] = []
        best: Blueprint | None = None
        best_success_count = -1
        for candidate_id, candidate_blueprint in candidates:
            self._assert_unaffected_nodes_frozen(
                previous,
                candidate_blueprint,
                failed_set,
            )
            if self.lean_checker is None:
                best = candidate_blueprint
                compile_audit.append(
                    {"candidate_id": candidate_id, "compiler_available": False}
                )
                break
            check = self.lean_checker.check_blueprint_detailed(
                problem,
                candidate_blueprint,
            )
            success_count = sum(row.success for row in check.node_results)
            compile_audit.append(
                {
                    "candidate_id": candidate_id,
                    "compiler_available": True,
                    "success": check.success,
                    "failed_node_ids": check.failed_node_ids,
                    "node_results": [
                        {
                            "node_id": row.node_id,
                            "success": row.success,
                            "error": row.result.error_message or row.result.stderr,
                        }
                        for row in check.node_results
                    ],
                }
            )
            if success_count > best_success_count:
                best = candidate_blueprint
                best_success_count = success_count
            if check.success:
                best = candidate_blueprint
                break
        if best is None:
            raise ValueError("formalization repair produced no usable candidate")
        best.metadata.setdefault("formalization_repair", []).append(
            {
                "environment": environment.model_dump(mode="json"),
                "failed_node_ids": failed_ids,
                "retrieval": retrieval,
                "candidate_compilation": compile_audit,
            }
        )
        return best


# Compatibility name used by older callers.
BlueprintRepairer = PlannerSubproblemRepairer

__all__ = [
    "BlueprintRepairer",
    "FormalizationNodePatch",
    "FormalizationRepairCandidate",
    "FormalizationRepairCandidates",
    "PlannerSubproblemRepairer",
]
