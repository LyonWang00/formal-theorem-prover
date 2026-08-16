"""Retrieval-guided candidate generation for Pantograph-rejected proofs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import AliasChoices, BaseModel, Field, field_validator

from lean_prover.Mathlib import (
    CompilerErrorRetriever,
    EnvironmentDeclarationIndex,
)
from lean_prover.Planner.client import _extract_json_object
from lean_prover.Planner.schemas import LeanFeedback, LeanEnvironmentIdentity
from lean_prover.model_api import MAX_EMPTY_RESPONSE_ATTEMPTS
from lean_prover.Prover.service import (
    GeneratedProof,
    NodeProofRequest,
    ProofGenerationError,
    build_node_proof_prompt,
    extract_lean_proof_body,
)
from lean_prover.Prover.repair_policy import retrieval_grounding_required


class ProverRepairCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    proof: str = Field(
        min_length=1,
        validation_alias=AliasChoices("proof", "tactics", "proof_body"),
    )
    used_declarations: list[str] = Field(default_factory=list)
    rationale: str = "Compiler-feedback repair candidate."

    @field_validator("proof")
    @classmethod
    def proof_body_only(cls, value: str) -> str:
        return extract_lean_proof_body(value)


class ProverRepairCandidates(BaseModel):
    candidates: list[ProverRepairCandidate] = Field(
        min_length=1,
        max_length=1,
    )


_CANDIDATE_FEW_SHOT = r"""
Output-format example only:
{
  "candidates": [
    {
      "candidate_id": "C1",
      "proof": "exact h",
      "used_declarations": [],
      "rationale": "The local hypothesis h has exactly the target type."
    }
  ]
}
""".strip()


PROVER_REPAIR_SYSTEM_PROMPT = r"""
You repair one Lean 4 proof body in one exact pinned Lean/Mathlib environment.
Return exactly one JSON object with a `candidates` array in the supplied
few-shot format. Generate exactly one small proof-body candidate. The caller, not
you, appends `:= by` and compiles every candidate with Pantograph.

For an unknown/obsolete identifier or constant, or invalid field notation,
make the smallest possible reference edit. You MUST choose at least one exact
fully-qualified declaration from LOCAL_ENVIRONMENT_RETRIEVAL, use it in the
proof, and list that exact name in `used_declarations`; a repair without this
pinned-index evidence is rejected even if it compiles.

For unsolved goals, tactic execution failures, parser errors, synthesis errors,
or type/application mismatches, you may replace the tactic sequence and proof
idea while keeping the exact target and dependencies immutable. Use the full
Pantograph goal/error state and prior-round history. Do not change imports,
statements, hypotheses, or the dependency graph.

Use library declarations only when their exact full names occur in
LOCAL_ENVIRONMENT_RETRIEVAL or already occur in the immutable target,
dependencies, preamble, or available definitions. Built-in Lean tactics and
local hypotheses are allowed. Never guess an API, deprecated name, import, or
field notation. Do not alter the theorem statement or dependency statements.
The dependency graph and its mathematical/logical adequacy are immutable here:
use the supplied dependencies as available facts, but do not judge, add, delete,
or rewire them.
Do not use sorry, admit, axiom, unsafe, declarations, `:=`, a leading `by`,
Markdown, or prose outside JSON.
""".strip()


def build_prover_repair_prompt(
    *,
    request: NodeProofRequest,
    failed_proof: str,
    feedback: LeanFeedback,
    round_index: int = 1,
    retrieval: dict[str, object] | None = None,
    repair_history: Sequence[dict[str, object]] = (),
    include_blueprint_reasoning: bool = True,
) -> str:
    dependencies = [
        {
            "node_id": item.node_id,
            "lean_statement": item.declaration,
            "verified_proof_available": bool(item.verified_proof),
        }
        for item in request.dependencies
    ]
    blueprint_reasoning = ""
    if include_blueprint_reasoning:
        blueprint_reasoning = f"""
Concrete informal proof sketch:
{request.informal_proof}

Intended one-to-three logical ideas:
{json.dumps(list(request.logical_ideas), ensure_ascii=False, indent=2)}

"""
    return f"""
Run retrieval-generate-compile proof repair round {round_index}. Return exactly
one candidate value. Its `proof` contains only tactics after the
system-owned `:= by`. Candidates are compiled in their listed order. If all
fail, their complete Pantograph feedback is supplied in the next round.

If the failure detail is unknown_identifier, unknown_constant, or invalid_field,
the candidate must use and declare at least one exact full_name from
LOCAL_ENVIRONMENT_RETRIEVAL. Do not answer from model memory for those errors.

Exact local environment and cache identity:
{request.environment.model_dump_json(indent=2)}

Node: {request.node_id}
Original natural-language statement:
{request.informal_statement}

{blueprint_reasoning}Immutable Lean target:
{request.target_decl}

Current imports:
{json.dumps(list(request.imports), ensure_ascii=False, indent=2)}

Exact dependency propositions:
{json.dumps(dependencies, ensure_ascii=False, indent=2)}

Rejected proof body:
{failed_proof}

Pantograph failure category: {feedback.error_type}
Pantograph failure detail: {feedback.error_detail}
Complete Pantograph diagnostics:
{feedback.diagnostics or feedback.message}

LOCAL_ENVIRONMENT_RETRIEVAL (3--10 declarations when the diagnostic exposes
an actionable name/type; every entry comes from the pinned local index):
{json.dumps(retrieval or {}, ensure_ascii=False, indent=2)}

Previous repair rounds and compilation outcomes (do not repeat a failed
candidate unchanged):
{json.dumps(list(repair_history), ensure_ascii=False, indent=2)}

Complete verified-context request:
{build_node_proof_prompt(request, include_blueprint_reasoning=include_blueprint_reasoning)}

{_CANDIDATE_FEW_SHOT}
""".strip()


class ProverProofRepairer:
    """Retrieve pinned declarations and generate a small proof candidate set."""

    def __init__(
        self,
        *,
        client: object,
        model: str,
        project_path: str | Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        include_blueprint_reasoning: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.project_path = Path(project_path).resolve() if project_path else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_blueprint_reasoning = include_blueprint_reasoning
        self._indices: dict[str, EnvironmentDeclarationIndex] = {}

    def _retriever(
        self,
        environment: LeanEnvironmentIdentity,
    ) -> CompilerErrorRetriever | None:
        if self.project_path is None:
            return None
        key = (
            f"{environment.lean_commit}/{environment.mathlib_commit}/"
            f"{environment.environment_hash}"
        )
        if key not in self._indices:
            self._indices[key] = EnvironmentDeclarationIndex.for_project(
                project_path=self.project_path,
                environment=environment,
            )
        return CompilerErrorRetriever(self._indices[key])

    def _retrieval_payload(
        self,
        *,
        request: NodeProofRequest,
        feedback: LeanFeedback,
    ) -> dict[str, object]:
        retriever = self._retriever(request.environment)
        if retriever is None:
            return {
                "compiler_error_analysis": {},
                "candidate_declarations": [],
                "index_available": False,
            }
        context = retriever.retrieve(
            feedback.diagnostics or feedback.message,
            current_imports=request.imports,
            statement=request.target_decl,
            informal_statement=request.informal_statement,
            limit=10,
        )
        record = context.prompt_record()
        record["index_available"] = True
        return record

    @staticmethod
    def _allowed_declaration_names(
        request: NodeProofRequest,
        retrieval: dict[str, object],
    ) -> set[str]:
        names = set(request.mathlib_hints)
        names.update(item.node_id for item in request.dependencies)
        for raw in retrieval.get("candidate_declarations", []):
            if isinstance(raw, dict) and isinstance(raw.get("full_name"), str):
                names.add(str(raw["full_name"]))
        return names

    @staticmethod
    def _proof_mentions_declaration(proof: str, full_name: str) -> bool:
        """Require retrieved evidence to occur in the emitted proof itself."""

        identifier_char = r"A-Za-z0-9_'"
        return re.search(
            rf"(?<![{identifier_char}]){re.escape(full_name)}(?![{identifier_char}])",
            proof,
        ) is not None

    def generate_candidates(
        self,
        *,
        request: NodeProofRequest,
        failed_proof: str,
        feedback: LeanFeedback,
        round_index: int,
        repair_history: Sequence[dict[str, object]] = (),
    ) -> list[GeneratedProof]:
        retrieval = self._retrieval_payload(request=request, feedback=feedback)
        prompt = build_prover_repair_prompt(
            request=request,
            failed_proof=failed_proof,
            feedback=feedback,
            round_index=round_index,
            retrieval=retrieval,
            repair_history=repair_history,
            include_blueprint_reasoning=self.include_blueprint_reasoning,
        )
        correction_history: list[dict[str, str]] = []
        last_error = ""
        response: Any = None
        content: str | None = None
        generated: ProverRepairCandidates | None = None
        for attempt in range(MAX_EMPTY_RESPONSE_ATTEMPTS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": PROVER_REPAIR_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                    *correction_history,
                ],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                last_error = "empty response"
                continue
            raw = _extract_json_object(content)
            if raw is not None:
                try:
                    generated = ProverRepairCandidates.model_validate(raw)
                    break
                except Exception as error:
                    last_error = str(error)
            else:
                last_error = "response did not contain one JSON object"
            if attempt + 1 < MAX_EMPTY_RESPONSE_ATTEMPTS:
                correction_history.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "The program rejected that response. Return only "
                                "one JSON object matching the candidates few-shot. "
                                "Validation error: " + last_error
                            ),
                        },
                    ]
                )
        if generated is None:
            if last_error == "empty response":
                raise ProofGenerationError(
                    f"修复证明节点{request.node_id}时模型空响应"
                )
            raise ProofGenerationError(
                f"修复证明节点{request.node_id}时模型连续"
                f"{MAX_EMPTY_RESPONSE_ATTEMPTS}次未返回合法候选JSON: {last_error}"
            )

        allowed_names = self._allowed_declaration_names(request, retrieval)
        retrieved_names = {
            str(raw["full_name"])
            for raw in retrieval.get("candidate_declarations", [])
            if isinstance(raw, dict) and isinstance(raw.get("full_name"), str)
        }
        requires_index_grounding = retrieval_grounding_required(feedback)
        usage = getattr(response, "usage", None)
        results: list[GeneratedProof] = []
        for candidate in generated.candidates:
            ungrounded = sorted(
                set(candidate.used_declarations) - allowed_names
            )
            cited_retrieved_names = sorted(
                set(candidate.used_declarations) & retrieved_names
            )
            used_retrieved_names = sorted(
                name
                for name in cited_retrieved_names
                if self._proof_mentions_declaration(candidate.proof, name)
            )
            missing_required_index_evidence = requires_index_grounding and (
                not retrieval.get("index_available")
                or not retrieved_names
                or not used_retrieved_names
            )
            if requires_index_grounding and (
                ungrounded or missing_required_index_evidence
            ):
                # The Prover still compiles the candidate and retains the full
                # Pantograph result, but its deterministic grounding gate will
                # not accept a compiler success obtained with these names.
                grounding = {
                    "declarations_grounded": False,
                        "ungrounded_declared_names": ungrounded,
                        "advisory_ungrounded_declared_names": ungrounded,
                    "missing_required_index_evidence": (
                        missing_required_index_evidence
                    ),
                }
            else:
                grounding = {
                    "declarations_grounded": True,
                    "ungrounded_declared_names": [],
                    "advisory_ungrounded_declared_names": ungrounded,
                    "missing_required_index_evidence": False,
                }
            results.append(
                GeneratedProof(
                    proof=candidate.proof,
                    model=self.model,
                    prompt_type="api_proof_repair_retrieval",
                    metadata={
                        "prompt": prompt,
                        "raw_model_output": content,
                        "api_usage": {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(
                                usage, "completion_tokens", None
                            ),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        },
                        "repair_round": round_index,
                        "candidate_id": candidate.candidate_id,
                        "candidate_rationale": candidate.rationale,
                        "used_declarations": candidate.used_declarations,
                        "retrieval": retrieval,
                        "retrieval_grounding_required": (
                            requires_index_grounding
                        ),
                        "retrieved_declaration_names": sorted(retrieved_names),
                        "cited_retrieved_declaration_names": (
                            cited_retrieved_names
                        ),
                        "used_retrieved_declaration_names": (
                            used_retrieved_names
                        ),
                        "repaired_failure_type": feedback.error_type,
                        "repaired_failure_detail": feedback.error_detail,
                        **grounding,
                    },
                )
            )
        return results

    def repair(
        self,
        *,
        request: NodeProofRequest,
        failed_proof: str,
        feedback: LeanFeedback,
    ) -> GeneratedProof:
        """Compatibility wrapper for the single controlled candidate."""

        return self.generate_candidates(
            request=request,
            failed_proof=failed_proof,
            feedback=feedback,
            round_index=1,
        )[0]


__all__ = [
    "PROVER_REPAIR_SYSTEM_PROMPT",
    "ProverProofRepairer",
    "ProverRepairCandidate",
    "ProverRepairCandidates",
    "build_prover_repair_prompt",
]
