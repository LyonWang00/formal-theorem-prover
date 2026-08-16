"""Deterministic Blueprint audit plus a minimal semantic-model boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import threading
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from lean_prover.Planner.client import LLMClient, PlannerClientError
from lean_prover.model_api import MAX_EMPTY_RESPONSE_ATTEMPTS
from lean_prover.Planner.preamble import imports_from_header, ordered_imports
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    BlueprintNodeVerification,
    LeanEnvironmentIdentity,
    LeanPreamble,
    TheoremProblem,
    VerifyIssue,
)


VERIFY_SYSTEM_PROMPT = r"""
You perform exactly one task: compare one natural-language mathematical
statement with one proof-free Lean 4 declaration for strict semantic identity.
Do not audit JSON structure, headers, imports, dependencies, DAG structure, or
Lean syntax; deterministic code and Pantograph handle those tasks. In
particular, do not judge whether declared dependencies are mathematically
necessary, sufficient, or logically correct, and do not revise informal_proof
or logical_ideas.

Compare every mathematical object and its type, every binder and quantifier,
every condition/hypothesis, and the exact conclusion. Do not accept a stronger,
weaker, reversed, or supposedly equivalent reformulation. In particular, Nat
subtraction is truncated: never replace a requested order statement with a
subtraction equality or conversely merely because it looks equivalent.

Return one JSON object containing exactly five VALUES:
{
  "formal_statement_correct": true,
  "corrected_lean_statement": "",
  "issue_message": "",
  "repair_reference": "",
  "verification_notes": "The objects, hypotheses, and conclusion match exactly."
}

If the statements match, formal_statement_correct must be true and the next
three values must be empty strings. If they do not match, it must be false and
corrected_lean_statement must contain the complete corrected proof-free theorem
or lemma declaration. The natural-language statement is authoritative. Keep the
provided node declaration name exactly, do not output :=, by, tactics, proof,
sorry, admit, axiom, unsafe, markdown, comments, or surrounding prose.

Few-shot -- exact match:
NATURAL LANGUAGE: For a natural number n, prove n + 0 = n.
LEAN: lemma L1 (n : Nat) : n + 0 = n
JSON:
{
  "formal_statement_correct": true,
  "corrected_lean_statement": "",
  "issue_message": "",
  "repair_reference": "",
  "verification_notes": "Nat, the absence of extra assumptions, and the equality goal match exactly."
}

Few-shot -- semantic mismatch:
NATURAL LANGUAGE: For natural numbers a and b, assuming a - b = 0, prove b <= a.
LEAN: lemma L2 (a b : Nat) (h : a - b = 0) : a = b
JSON:
{
  "formal_statement_correct": false,
  "corrected_lean_statement": "lemma L2 (a b : Nat) (h : a - b = 0) : b <= a",
  "issue_message": "The Lean conclusion a = b is stronger than and different from the requested conclusion b <= a.",
  "repair_reference": "Preserve the Nat objects and hypothesis and replace only the conclusion with b <= a.",
  "verification_notes": "The natural-language order goal is authoritative; no unrequested equivalence is allowed."
}
""".strip()


class SemanticVerifyValues(BaseModel):
    """The only values an LLM is allowed to supply to Verify."""

    # This is an ingestion boundary, not the persisted schema. Extra model
    # decorations are ignored; the program constructs the strict final object.
    model_config = ConfigDict(extra="ignore")

    formal_statement_correct: bool
    corrected_lean_statement: str = ""
    issue_message: str = ""
    repair_reference: str = ""
    verification_notes: str = "Semantic correspondence checked."

    @model_validator(mode="after")
    def validate_consistency(self) -> "SemanticVerifyValues":
        if self.formal_statement_correct:
            # Ignore contradictory optional decorations when the authoritative
            # verdict is true. They never enter BlueprintNodeVerification.
            self.corrected_lean_statement = ""
            self.issue_message = ""
            self.repair_reference = ""
            return self
        return self


class VerifyStore(Protocol):
    def record(self, payload: dict[str, object]) -> None:
        ...


class JsonlVerifyStore:
    """Append every per-node result or failure after the node call finishes."""

    def __init__(self, path: str | Path = "outputs/verify/results.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, payload: dict[str, object]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                )


class BlueprintVerificationBatchError(RuntimeError):
    """All parallel calls finished, but at least one node call failed."""

    def __init__(
        self,
        *,
        failures: dict[str, Exception],
        partial_results: list[BlueprintNodeVerification],
    ) -> None:
        self.failures = failures
        self.partial_results = partial_results
        details = "; ".join(
            f"{node_id}: {type(error).__name__}: {error}"
            for node_id, error in sorted(failures.items())
        )
        super().__init__(
            "Verify collected all node calls and found failures: " + details
        )


class SemanticVerificationExhaustedError(RuntimeError):
    """One node remained inconsistent after two verify-repair rounds."""


@dataclass(frozen=True)
class _NodeVerifyOutcome:
    result: BlueprintNodeVerification
    corrected_lean_statement: str | None = None


def _normalized_line(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def deduplicate_raw_header(header: str) -> tuple[str, list[str]]:
    seen: set[str] = set()
    result: list[str] = []
    changes: list[str] = []
    for line in header.splitlines():
        normalized = _normalized_line(line)
        if not normalized:
            if result and result[-1] != "":
                result.append("")
            continue
        if normalized in seen:
            changes.append(f"removed duplicate header command: {normalized}")
            continue
        seen.add(normalized)
        result.append(line.rstrip())
    while result and not result[-1]:
        result.pop()
    return "\n".join(result).strip(), changes


def _deduplicate_values(
    values: list[str],
    *,
    rendered_prefix: str,
    raw_lines: set[str],
    changes: list[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        normalized = _normalized_line(stripped)
        rendered = _normalized_line(rendered_prefix + stripped)
        if normalized in seen:
            changes.append(f"removed duplicate preamble value: {stripped}")
            continue
        seen.add(normalized)
        if rendered in raw_lines:
            changes.append(
                f"removed preamble value already present in header: {rendered}"
            )
            continue
        result.append(stripped)
    return result


def _corrected_preamble(
    *,
    problem: TheoremProblem,
    node: BlueprintNode,
) -> tuple[LeanPreamble, list[str]]:
    """Normalize duplicates without delegating any header decision to the LLM."""

    raw_header, changes = deduplicate_raw_header(node.preamble.raw_header)
    raw_lines = {
        _normalized_line(line)
        for line in raw_header.splitlines()
        if line.strip()
    }
    imports = ordered_imports(
        problem.imports,
        imports_from_header(raw_header),
        node.preamble.imports,
    )
    if imports != node.preamble.imports:
        changes.append("deduplicated and completed imports deterministically")
    corrected = LeanPreamble(
        imports=imports,
        raw_header=raw_header,
        namespaces=_deduplicate_values(
            node.preamble.namespaces,
            rendered_prefix="namespace ",
            raw_lines=raw_lines,
            changes=changes,
        ),
        open_namespaces=_deduplicate_values(
            node.preamble.open_namespaces,
            rendered_prefix="open ",
            raw_lines=raw_lines,
            changes=changes,
        ),
        open_scoped=_deduplicate_values(
            node.preamble.open_scoped,
            rendered_prefix="open scoped ",
            raw_lines=raw_lines,
            changes=changes,
        ),
        variable_declarations=_deduplicate_values(
            node.preamble.variable_declarations,
            rendered_prefix="",
            raw_lines=raw_lines,
            changes=changes,
        ),
        local_context=_deduplicate_values(
            node.preamble.local_context,
            rendered_prefix="",
            raw_lines=raw_lines,
            changes=changes,
        ),
    )
    return corrected, changes


def _declaration_name(statement: str) -> str | None:
    match = re.match(r"\s*(?:lemma|theorem)\s+([A-Za-z_][A-Za-z0-9_'.]*)", statement)
    return match.group(1) if match else None


def _dependency_issues(
    *,
    blueprint: Blueprint,
    node: BlueprintNode,
) -> list[VerifyIssue]:
    """Audit existence, order, identity, and exact program materialization."""

    nodes = {item.id: item for item in blueprint.nodes}
    positions = {item.id: index for index, item in enumerate(blueprint.nodes)}
    issues: list[VerifyIssue] = []
    seen: set[str] = set()
    for dependency_id in node.depends_on:
        if dependency_id in seen:
            issues.append(
                VerifyIssue(
                    component="dependencies",
                    message=f"Dependency {dependency_id} is listed more than once.",
                    repair_reference="Keep each predecessor ID exactly once.",
                )
            )
            continue
        seen.add(dependency_id)
        dependency = nodes.get(dependency_id)
        if dependency is None:
            issues.append(
                VerifyIssue(
                    component="dependencies",
                    message=f"Dependency {dependency_id} does not exist in the Blueprint.",
                    repair_reference="Create the predecessor or remove the invalid dependency edge.",
                )
            )
            continue
        if positions[dependency_id] >= positions[node.id]:
            issues.append(
                VerifyIssue(
                    component="dependencies",
                    message=f"Dependency {dependency_id} is not placed before {node.id}.",
                    repair_reference="Place every predecessor before the node that cites it.",
                )
            )
        declaration_name = _declaration_name(dependency.lean_decl)
        if declaration_name != dependency_id:
            issues.append(
                VerifyIssue(
                    component="dependencies",
                    message=(
                        f"Dependency {dependency_id} declares the name "
                        f"{declaration_name!r}, so its exact proposition cannot "
                        "be inserted and cited under the dependency ID."
                    ),
                    repair_reference=(
                        f"Keep the exact predecessor proposition and declare it "
                        f"under the name {dependency_id}."
                    ),
                )
            )
    return issues


def _semantic_prompt(
    *,
    problem: TheoremProblem,
    node: BlueprintNode,
    environment: LeanEnvironmentIdentity,
) -> str:
    values_template = {
        "formal_statement_correct": True,
        "corrected_lean_statement": "",
        "issue_message": "",
        "repair_reference": "",
        "verification_notes": "Replace only this value with the semantic audit.",
    }
    payload = {
        "environment": environment.model_dump(mode="json"),
        "root_natural_language_reference": problem.natural_language_statement,
        "node_id_must_be_preserved": node.id,
        "node_natural_language_statement": node.informal_statement,
        "node_lean_statement": node.lean_decl,
    }
    return (
        "Fill only the five semantic values in SEMANTIC_VALUES_TEMPLATE_JSON. "
        "The program, not you, constructs BlueprintNodeVerification, its "
        "preamble, dependencies, node_id, and issue objects.\n\n"
        "SEMANTIC_VALUES_TEMPLATE_JSON:\n"
        + json.dumps(values_template, ensure_ascii=False, indent=2)
        + "\n\nSEMANTIC_VERIFY_INPUT_JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


class BlueprintVerifier:
    """Run semantic calls in parallel; construct every final result in code."""

    def __init__(
        self,
        client: LLMClient,
        *,
        store: VerifyStore | None = None,
        max_workers: int = 8,
        max_semantic_repairs: int = 2,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_semantic_repairs < 1:
            raise ValueError("max_semantic_repairs must be at least 1")
        self.client = client
        self.store = store
        self.max_workers = max_workers
        self.max_semantic_repairs = max_semantic_repairs

    def _semantic_values(
        self,
        *,
        problem: TheoremProblem,
        node: BlueprintNode,
        environment: LeanEnvironmentIdentity,
    ) -> SemanticVerifyValues:
        user_prompt = _semantic_prompt(
            problem=problem,
            node=node,
            environment=environment,
        )
        history: list[dict[str, str]] = []
        last_error: ValidationError | None = None
        for attempt in range(MAX_EMPTY_RESPONSE_ATTEMPTS):
            raw_values = self.client.generate_json(
                system_prompt=VERIFY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                empty_response_message="verify模型空响应",
                history=history,
            )
            try:
                return SemanticVerifyValues.model_validate(raw_values)
            except ValidationError as error:
                last_error = error
                if attempt + 1 >= MAX_EMPTY_RESPONSE_ATTEMPTS:
                    break
                history.extend(
                    [
                        {
                            "role": "assistant",
                            "content": json.dumps(raw_values, ensure_ascii=False),
                        },
                        {
                            "role": "user",
                            "content": (
                                "The program rejected those values. Return one "
                                "JSON object with the required boolean value "
                                "formal_statement_correct. All other requested "
                                "values may be empty strings. Validation error: "
                                + str(error)
                            ),
                        },
                    ]
                )
        raise PlannerClientError(
            "verify model did not return the required semantic verdict after "
            f"{MAX_EMPTY_RESPONSE_ATTEMPTS} attempts: {last_error}"
        )

    def _verify_one(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        node: BlueprintNode,
        environment: LeanEnvironmentIdentity,
    ) -> _NodeVerifyOutcome:
        corrected_preamble, header_changes = _corrected_preamble(
            problem=problem,
            node=node,
        )
        dependency_issues = _dependency_issues(
            blueprint=blueprint,
            node=node,
        )
        values = self._semantic_values(
            problem=problem,
            node=node,
            environment=environment,
        )
        corrected_statement: str | None = None
        formal_issues: list[VerifyIssue] = []
        if not values.formal_statement_correct:
            if values.corrected_lean_statement.strip():
                corrected_statement = BlueprintNode.validate_lean_decl(
                    values.corrected_lean_statement
                )
                if _declaration_name(corrected_statement) != node.id:
                    raise ValueError(
                        f"Verify semantic patch changed node name {node.id}"
                    )
                if corrected_statement == node.lean_decl:
                    # The semantic verdict is useful, but a no-op patch is not
                    # a repair. Planner will route the issue to formalization.
                    corrected_statement = None
            repair_count = int(
                node.metadata.get("verify_semantic_repair_count", 0)
            )
            if repair_count >= self.max_semantic_repairs:
                raise SemanticVerificationExhaustedError(
                    f"节点{node.id}经过{self.max_semantic_repairs}轮verify-repair后语义仍不一致"
                )
            formal_issues.append(
                VerifyIssue(
                    component="formal_statement",
                    message=(
                        values.issue_message.strip()
                        or "The Lean declaration does not strictly match the "
                        "node natural-language statement."
                    ),
                    repair_reference=(
                        values.repair_reference.strip()
                        or "Reformalize the declaration from the authoritative "
                        "natural-language statement without changing its objects, "
                        "hypotheses, or conclusion."
                    ),
                )
            )
        result = BlueprintNodeVerification(
            node_id=node.id,
            corrected_preamble=corrected_preamble,
            header_changes=header_changes,
            dependency_statements_correct=not dependency_issues,
            dependency_issues=dependency_issues,
            formal_statement_correct=values.formal_statement_correct,
            formal_statement_issues=formal_issues,
            verification_notes=(
                values.verification_notes.strip()
                or "Semantic correspondence checked."
            ),
        )
        return _NodeVerifyOutcome(
            result=result,
            corrected_lean_statement=corrected_statement,
        )

    def _record_result(
        self,
        *,
        problem: TheoremProblem,
        result: BlueprintNodeVerification,
    ) -> None:
        if self.store is None:
            return
        self.store.record(
            {
                "record_type": "verification_result",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "problem_id": problem.problem_id,
                "problem_hash": problem.problem_hash,
                **result.model_dump(mode="json"),
            }
        )

    def verify_blueprint(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        environment: LeanEnvironmentIdentity,
        node_ids: set[str] | None = None,
    ) -> list[BlueprintNodeVerification]:
        selected = [
            node
            for node in blueprint.nodes
            if node_ids is None or node.id in node_ids
        ]
        if not selected:
            return []
        outcomes: dict[str, _NodeVerifyOutcome] = {}
        failures: dict[str, Exception] = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(selected))
        ) as executor:
            futures = {
                executor.submit(
                    self._verify_one,
                    problem=problem,
                    blueprint=blueprint,
                    node=node,
                    environment=environment,
                ): node.id
                for node in selected
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    outcomes[node_id] = future.result()
                except Exception as error:
                    failures[node_id] = error

        ordered_outcomes = [
            outcomes[node.id] for node in selected if node.id in outcomes
        ]
        for outcome in ordered_outcomes:
            self._record_result(problem=problem, result=outcome.result)
        if failures:
            if self.store is not None:
                for node_id, error in sorted(failures.items()):
                    self.store.record(
                        {
                            "record_type": "verification_error",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "problem_id": problem.problem_id,
                            "problem_hash": problem.problem_hash,
                            "node_id": node_id,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
            raise BlueprintVerificationBatchError(
                failures=failures,
                partial_results=[item.result for item in ordered_outcomes],
            )

        nodes = {node.id: node for node in blueprint.nodes}
        for outcome in ordered_outcomes:
            node = nodes[outcome.result.node_id]
            if not outcome.result.formal_statement_correct:
                node.metadata["verify_semantic_repair_count"] = (
                    int(node.metadata.get("verify_semantic_repair_count", 0))
                    + 1
                )
            statement = outcome.corrected_lean_statement
            if statement is None:
                continue
            previous = node.lean_decl
            node.lean_decl = statement
            node.metadata["verify_semantic_patch_pending"] = True
            node.metadata.setdefault("verify_semantic_patches", []).append(
                {
                    "previous_lean_statement": previous,
                    "corrected_lean_statement": statement,
                    "issue": outcome.result.formal_statement_issues[0].message,
                }
            )
        return [item.result for item in ordered_outcomes]


__all__ = [
    "BlueprintVerificationBatchError",
    "BlueprintVerifier",
    "deduplicate_raw_header",
    "JsonlVerifyStore",
    "SemanticVerifyValues",
    "SemanticVerificationExhaustedError",
    "VERIFY_SYSTEM_PROMPT",
    "VerifyStore",
]
