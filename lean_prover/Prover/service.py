"""API-only proof generation, verification, and result routing."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from lean_prover.Data import (
    LeanFailureDetail,
    LeanVerificationStatus,
    ProverProblemResult,
    classify_lean_diagnostics,
)
from lean_prover.model_api import MAX_EMPTY_RESPONSE_ATTEMPTS
from lean_prover.Planner.graph import topological_order
from lean_prover.Planner.preamble import (
    header_context_without_imports,
    ordered_imports,
    wrap_source_with_preamble,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    LeanEnvironmentIdentity,
    LeanFeedback,
    LeanPreamble,
    NodeStatus,
    ProofAttempt,
    ProofLengthEstimate,
    SemanticAlignment,
    TheoremProblem,
)

from .results import ProverResultStore, collect_prover_result
from .repair_policy import enforce_repair_grounding, proof_repair_eligible


class ProofGenerationError(RuntimeError):
    pass


class ProverSettingsError(RuntimeError):
    pass


_FENCED_CODE_RE = re.compile(
    r"```(?:lean4?|Lean4?)?\s*\n(?P<code>.*?)```",
    re.DOTALL,
)


def extract_lean_proof_body(model_output: str) -> str:
    """Extract tactics only; the verifier owns the single normalized `:= by`."""

    text = model_output.strip()
    if not text:
        raise ProofGenerationError("proof API returned empty content")
    fenced = [
        match.group("code").strip()
        for match in _FENCED_CODE_RE.finditer(text)
    ]
    for candidate in [*reversed(fenced), text]:
        declaration_match = re.search(
            r":=\s*by\b(?P<tactics>.*)\Z",
            candidate,
            flags=re.DOTALL,
        )
        if declaration_match:
            tactics = declaration_match.group("tactics").strip()
            if tactics:
                return tactics
        body_match = re.search(
            r"(?m)^[ \t]*by\b(?P<tactics>.*)\Z",
            candidate,
            flags=re.DOTALL,
        )
        if body_match:
            tactics = body_match.group("tactics").strip()
            if tactics:
                return tactics
        if candidate and not re.search(r"\b(?:theorem|lemma)\b|:=", candidate):
            return candidate.strip()
    raise ProofGenerationError(
        "proof API output did not contain Lean tactics"
    )


def normalized_proof_body(tactics: str) -> str:
    stripped = tactics.strip()
    if stripped.startswith("by") and re.match(r"^by(?:\s|$)", stripped):
        stripped = stripped[2:].strip()
    if not stripped:
        raise ProofGenerationError("proof tactics must not be empty")
    return "by\n" + "\n".join(
        "  " + line if line.strip() else line
        for line in stripped.splitlines()
    )


@dataclass(frozen=True)
class DependencyContext:
    node_id: str
    declaration: str
    preamble: LeanPreamble
    verified_proof: str | None = None

    def statement_source(self) -> str:
        return wrap_source_with_preamble(self.declaration, self.preamble)

    def verified_source(self) -> str | None:
        if not self.verified_proof:
            return None
        completed = (
            self.declaration.rstrip()
            + " := "
            + normalized_proof_body(self.verified_proof)
        )
        return wrap_source_with_preamble(completed, self.preamble)


@dataclass(frozen=True)
class NodeProofRequest:
    problem_id: str
    problem_hash: str
    node_id: str
    imports: tuple[str, ...]
    target_decl: str
    informal_statement: str
    informal_proof: str
    logical_ideas: tuple[str, ...]
    proof_strategy: str
    available_definitions: tuple[str, ...]
    dependencies: tuple[DependencyContext, ...]
    preamble: LeanPreamble
    semantic_alignment: SemanticAlignment
    estimated_proof_length: ProofLengthEstimate
    environment: LeanEnvironmentIdentity
    mathlib_hints: tuple[str, ...] = ()
    dependency_statements_verified: bool = False
    formal_statement_verified: bool = False


@dataclass(frozen=True)
class GeneratedProof:
    proof: str
    model: str
    prompt_type: str = "api_proof_generation"
    score: float | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ProverAttemptRecord:
    event_id: str
    created_at: str
    problem_id: str
    problem_hash: str
    node_id: str
    attempt_id: str
    model: str
    prompt_type: str
    prompt: str
    request: dict[str, Any]
    proof: str
    success: bool | None
    status: str
    verification_status: str = LeanVerificationStatus.SUCCESS.value
    failure_detail: str = LeanFailureDetail.NONE.value
    diagnostics: str = ""
    error_message: str = ""
    metadata: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "created_at": self.created_at,
            "problem_id": self.problem_id,
            "problem_hash": self.problem_hash,
            "node_id": self.node_id,
            "attempt_id": self.attempt_id,
            "model": self.model,
            "prompt_type": self.prompt_type,
            "prompt": self.prompt,
            "request": self.request,
            "proof": self.proof,
            "success": self.success,
            "status": self.status,
            "verification_status": self.verification_status,
            "failure_detail": self.failure_detail,
            "diagnostics": self.diagnostics,
            "error_message": self.error_message,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class ProverApiSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "ProverApiSettings":
        api_key = (
            os.environ.get("PROVER_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ProverSettingsError("PROVER_API_KEY is not set")
        base_url = os.environ.get("PROVER_BASE_URL")
        if not base_url:
            if os.environ.get("DEEPSEEK_API_KEY") == api_key:
                base_url = "https://api.deepseek.com"
            elif os.environ.get("OPENAI_API_KEY") == api_key:
                base_url = "https://api.openai.com/v1"
        if not base_url:
            raise ProverSettingsError("PROVER_BASE_URL is not set")
        model = (
            os.environ.get("PROVER_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
        )
        if not model:
            raise ProverSettingsError("PROVER_MODEL is not set")
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(
                os.environ.get("PROVER_TEMPERATURE", str(cls.temperature))
            ),
            max_tokens=int(
                os.environ.get("PROVER_MAX_TOKENS", str(cls.max_tokens))
            ),
        )


class ProofGenerator(Protocol):
    def generate(self, request: NodeProofRequest) -> GeneratedProof:
        ...


class ProofRepairer(Protocol):
    def repair(
        self,
        *,
        request: NodeProofRequest,
        failed_proof: str,
        feedback: LeanFeedback,
    ) -> GeneratedProof:
        ...


class CandidateProofRepairer(Protocol):
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


class ProverTelemetryStore(Protocol):
    def record_attempt(self, record: ProverAttemptRecord) -> None:
        ...

    def records_for_problem(self, problem_hash: str) -> list[dict[str, Any]]:
        ...


class ProofVerificationRuntime(Protocol):
    imports: tuple[str, ...]

    def check_source(self, source: str) -> Any:
        ...


class JsonlProverTelemetryStore:
    """Append-only storage preserving the remote Prover attempt contract."""

    def __init__(
        self,
        *,
        attempts_path: str | Path = "outputs/prover/attempts.jsonl",
    ) -> None:
        self.attempts_path = Path(attempts_path)

    def record_attempt(self, record: ProverAttemptRecord) -> None:
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record.to_json(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def records_for_problem(self, problem_hash: str) -> list[dict[str, Any]]:
        if not self.attempts_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.attempts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("problem_hash") == problem_hash:
                    records.append(record)
        return records


def _request_to_json(request: NodeProofRequest) -> dict[str, Any]:
    return {
        "problem_id": request.problem_id,
        "problem_hash": request.problem_hash,
        "node_id": request.node_id,
        "imports": list(request.imports),
        "target_decl": request.target_decl,
        "informal_statement": request.informal_statement,
        "informal_proof": request.informal_proof,
        "logical_ideas": list(request.logical_ideas),
        "proof_strategy": request.proof_strategy,
        "available_definitions": list(request.available_definitions),
        "dependencies": [
            {
                "node_id": dependency.node_id,
                "declaration": dependency.declaration,
                "preamble": dependency.preamble.model_dump(mode="json"),
                "verified": bool(dependency.verified_proof),
            }
            for dependency in request.dependencies
        ],
        "preamble": request.preamble.model_dump(mode="json"),
        "semantic_alignment": request.semantic_alignment.model_dump(mode="json"),
        "estimated_proof_length": (
            request.estimated_proof_length.model_dump(mode="json")
        ),
        "environment": request.environment.model_dump(mode="json"),
        "mathlib_hints": list(request.mathlib_hints),
        "dependency_statements_verified": request.dependency_statements_verified,
        "formal_statement_verified": request.formal_statement_verified,
    }


def build_node_proof_prompt(
    request: NodeProofRequest,
    *,
    include_blueprint_reasoning: bool = True,
) -> str:
    imports = "\n".join(f"import {item}" for item in request.imports)
    dependencies = "\n\n".join(
        dependency.statement_source()
        for dependency in request.dependencies
    ) or "None"
    definitions = "\n\n".join(request.available_definitions) or "None"
    preamble = json.dumps(
        request.preamble.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    materialized_target = wrap_source_with_preamble(
        request.target_decl,
        request.preamble,
    )
    alignment = json.dumps(
        request.semantic_alignment.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    estimate = request.estimated_proof_length
    hints = ", ".join(request.mathlib_hints) or "None"
    blueprint_reasoning = ""
    if include_blueprint_reasoning:
        blueprint_reasoning = f"""
Concrete informal proof sketch (declared dependencies use exact node IDs):
{request.informal_proof}

New logical ideas intended for this node (one to three):
{json.dumps(list(request.logical_ideas), ensure_ascii=False)}

"""
    return f"""You generate one Lean 4 proof body through an API.

Return only the tactics that follow the system-owned `:= by`. Do not output
`:=`, the leading `by`, a declaration, markdown, commentary, or proof plan. Do
not use sorry, admit, axiom, or unsafe.
The node has already passed strict header, dependency, formal/informal semantic,
and Pantograph statement validation. Use the dependency declarations under
their exact displayed names. Do not invent renamed, deprecated, or guessed
Mathlib declarations, and do not reinterpret the target as an equivalent claim.

Format-only example:
- Target declaration: `lemma example (p : Prop) (hp : p) : p`
- Output: `exact hp`

Solve the actual target below; do not copy the example proof unless it applies.

Exact Lean environment:
{request.environment.model_dump_json(indent=2)}

Problem hash: {request.problem_hash}
Node ID: {request.node_id}

Imports loaded by the verifier:
{imports}

Available local definitions:
{definitions}

Complete structured preamble for this node:
{preamble}

Materialized Lean context and target declaration:
{materialized_target}

System-owned normalized proof prefix (do not repeat it in your output):
{request.target_decl} := by

Available dependency declarations (their own preambles are materialized):
{dependencies}

Target declaration (without proof):
{request.target_decl}

Informal statement:
{request.informal_statement}

{blueprint_reasoning}Semantic-alignment audit:
{alignment}

Pre-compilation verification flags:
- dependencies verified: {request.dependency_statements_verified}
- formal statement verified: {request.formal_statement_verified}

Suggested proof strategy:
{request.proof_strategy or "None"}

Estimated proof-body budget:
- lines: {estimate.estimated_lines}
- tokens: {estimate.estimated_tokens}
- rationale: {estimate.rationale}

Mathlib hints:
{hints}
"""


class OpenAICompatibleProofGenerator:
    def __init__(
        self,
        *,
        client: object,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        include_blueprint_reasoning: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_blueprint_reasoning = include_blueprint_reasoning

    def generate(self, request: NodeProofRequest) -> GeneratedProof:
        prompt = build_node_proof_prompt(
            request,
            include_blueprint_reasoning=self.include_blueprint_reasoning,
        )
        content: str | None = None
        for _ in range(MAX_EMPTY_RESPONSE_ATTEMPTS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate verified Lean 4 proof bodies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                break
        else:
            raise ProofGenerationError(
                f"证明节点{request.node_id}时模型空响应"
            )
        usage = getattr(response, "usage", None)
        return GeneratedProof(
            proof=extract_lean_proof_body(content),
            model=self.model,
            prompt_type="api_proof_generation",
            metadata={
                "raw_model_output": content,
                "api_usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(
                        usage, "completion_tokens", None
                    ),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            },
        )


def create_openai_compatible_proof_generator_from_env(
    *,
    client: object | None = None,
) -> OpenAICompatibleProofGenerator:
    settings = ProverApiSettings.from_env()
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    return OpenAICompatibleProofGenerator(
        client=client,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


class NodeProofVerifier:
    _FORBIDDEN = re.compile(
        r"\b(?:sorry|admit|axiom|unsafe)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, verifier: ProofVerificationRuntime) -> None:
        self.verifier = verifier

    @staticmethod
    def _imports_are_covered(
        requested: tuple[str, ...],
        loaded: tuple[str, ...],
    ) -> bool:
        loaded_set = set(loaded)
        return all(
            module in loaded_set
            or (
                "Mathlib" in loaded_set
                and (module == "Mathlib" or module.startswith("Mathlib."))
            )
            for module in requested
        )

    def verify(
        self,
        *,
        request: NodeProofRequest,
        proof: str,
    ) -> LeanFeedback:
        if self._FORBIDDEN.search(proof):
            classification = classify_lean_diagnostics(
                "proof contains forbidden sorry/admit/axiom/unsafe",
                forbidden_token=True,
            )
            return LeanFeedback(
                stage="proof_verification",
                success=False,
                message="proof contains forbidden sorry/admit/axiom/unsafe",
                error_type=classification.status.value,
                error_detail=classification.detail.value,
            )
        if not self._imports_are_covered(
            request.imports,
            self.verifier.imports,
        ):
            diagnostics = (
                f"requested={request.imports}, "
                f"loaded={self.verifier.imports}"
            )
            classification = classify_lean_diagnostics(
                diagnostics,
                environment_error=True,
            )
            return LeanFeedback(
                stage="proof_verification",
                success=False,
                message="proof imports are not covered by Pantograph server",
                diagnostics=diagnostics,
                error_type=classification.status.value,
                error_detail=classification.detail.value,
            )
        dependency_sources = [
            source
            for dependency in request.dependencies
            if (source := dependency.verified_source()) is not None
        ]
        try:
            completed = (
                request.target_decl.rstrip()
                + " := "
                + normalized_proof_body(proof)
            )
        except ProofGenerationError as error:
            classification = classify_lean_diagnostics(
                str(error), extraction_error=True
            )
            return LeanFeedback(
                stage="proof_verification",
                success=False,
                message=str(error),
                diagnostics=str(error),
                error_type=classification.status.value,
                error_detail=classification.detail.value,
            )
        current_source = wrap_source_with_preamble(
            completed,
            request.preamble,
        )
        full_header = request.preamble.raw_header.strip()
        header = header_context_without_imports(full_header)
        dependency_sources = [
            source.replace(full_header, "", 1).lstrip()
            if full_header and source.startswith(full_header)
            else source
            for source in dependency_sources
        ]
        source = "\n\n".join(
            part
            for part in [
                header,
                *request.available_definitions,
                *dependency_sources,
                (
                    current_source.replace(full_header, "", 1).lstrip()
                    if full_header and current_source.startswith(full_header)
                    else current_source
                ),
            ]
            if part.strip()
        )
        result = self.verifier.check_source(source)
        diagnostics = str(result.diagnostics or "")
        classification = classify_lean_diagnostics(
            diagnostics,
            success=bool(result.success),
            timed_out=bool(getattr(result, "timed_out", False)),
        )
        return LeanFeedback(
            stage="proof_verification",
            success=bool(result.success),
            message=(
                "Lean accepted proof"
                if result.success
                else "Lean rejected proof"
            ),
            diagnostics=diagnostics,
            error_type=classification.status.value,
            error_detail=classification.detail.value,
        )


class BlueprintProver:
    """Connect a validated Planner Blueprint to the proof API and verifier."""

    def __init__(
        self,
        *,
        generator: ProofGenerator,
        verifier: NodeProofVerifier | None = None,
        telemetry_store: ProverTelemetryStore | None = None,
        result_store: ProverResultStore | None = None,
        proof_repairer: ProofRepairer | None = None,
        attempts_per_node: int = 4,
        max_proof_repair_rounds: int = 5,
        max_proof_repair_candidates_per_node: int | None = None,
    ) -> None:
        if attempts_per_node < 1:
            raise ValueError("attempts_per_node must be at least 1")
        if max_proof_repair_rounds < 1:
            raise ValueError("max_proof_repair_rounds must be at least 1")
        if (
            max_proof_repair_candidates_per_node is not None
            and max_proof_repair_candidates_per_node < 1
        ):
            raise ValueError(
                "max_proof_repair_candidates_per_node must be at least 1"
            )
        self.generator = generator
        self.verifier = verifier
        self.telemetry_store = telemetry_store
        self.result_store = result_store
        self.proof_repairer = proof_repairer
        self.attempts_per_node = attempts_per_node
        self.max_proof_repair_rounds = max_proof_repair_rounds
        self.max_proof_repair_candidates_per_node = (
            max_proof_repair_candidates_per_node
            if max_proof_repair_candidates_per_node is not None
            else max_proof_repair_rounds
        )

    @staticmethod
    def _nodes(blueprint: Blueprint) -> dict[str, BlueprintNode]:
        return {node.id: node for node in blueprint.nodes}

    def build_request(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        node: BlueprintNode,
    ) -> NodeProofRequest:
        nodes = self._nodes(blueprint)
        dependencies = tuple(
            DependencyContext(
                node_id=dependency,
                declaration=nodes[dependency].lean_decl,
                preamble=nodes[dependency].preamble,
                verified_proof=nodes[dependency].verified_proof,
            )
            for dependency in node.depends_on
            if dependency in nodes
        )
        return NodeProofRequest(
            problem_id=problem.problem_id,
            problem_hash=problem.problem_hash,
            node_id=node.id,
            imports=tuple(
                ordered_imports(
                    problem.imports,
                    node.preamble.imports,
                    *(dependency.preamble.imports for dependency in dependencies),
                )
            ),
            target_decl=node.lean_decl,
            informal_statement=node.informal_statement,
            informal_proof=node.informal_proof,
            logical_ideas=tuple(node.logical_ideas),
            proof_strategy=node.proof_strategy,
            available_definitions=tuple(problem.available_definitions),
            dependencies=dependencies,
            preamble=node.preamble,
            semantic_alignment=node.semantic_alignment,
            estimated_proof_length=node.estimated_proof_length,
            environment=blueprint.environment,
            mathlib_hints=tuple(node.mathlib_hints),
            dependency_statements_verified=bool(
                node.dependency_statements_verified
            ),
            formal_statement_verified=bool(node.formal_statement_verified),
        )

    @staticmethod
    def build_root_node(
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> BlueprintNode:
        """Materialize the virtual DAG ROOT as the immutable theorem target."""

        return BlueprintNode(
            id="ROOT",
            title="Final theorem target",
            informal_statement=(
                problem.natural_language_statement
                or "Prove the immutable Lean theorem target."
            ),
            informal_proof=(
                "Use "
                + ", ".join(f"`{node_id}`" for node_id in blueprint.root_dependencies)
                + ", then combine their stated conclusions to derive the "
                "immutable target."
                if blueprint.root_dependencies
                else "Prove the immutable target directly from its hypotheses."
            ),
            logical_ideas=["Combine the established prerequisites to close the target"],
            lean_statement=problem.target_lean_decl,
            preamble=LeanPreamble(
                imports=list(problem.imports),
                raw_header=problem.header,
            ),
            semantic_alignment=SemanticAlignment(
                objects=["All binders and objects in the immutable target"],
                hypotheses=["All hypotheses in the immutable target"],
                conclusion="The conclusion of the immutable target",
                alignment_notes=(
                    "ROOT is the original Planner-validated target without "
                    "strengthening or weakening."
                ),
            ),
            estimated_proof_length=ProofLengthEstimate(
                estimated_lines=20,
                estimated_tokens=256,
                rationale=(
                    "The final proof may combine every verified root dependency."
                ),
            ),
            depends_on=list(blueprint.root_dependencies),
            proof_strategy=(
                "Prove the original target using the verified root dependencies."
            ),
            difficulty=3,
            status=NodeStatus.PLANNED,
            metadata={"is_root_target": True},
            dependency_statements_verified=True,
            formal_statement_verified=True,
        )

    def _prove_root(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> Blueprint:
        """Prove ROOT with the same pass@k path used for ordinary nodes."""

        with_root = blueprint.model_copy(deep=True)
        with_root.nodes.append(
            self.build_root_node(problem=problem, blueprint=with_root)
        )
        proved = self.prove_node(
            problem=problem,
            blueprint=with_root,
            node_id="ROOT",
        )
        root = self._nodes(proved)["ROOT"]
        updated = proved.model_copy(deep=True)
        updated.nodes = [node for node in updated.nodes if node.id != "ROOT"]
        updated.metadata["root_node"] = root.model_dump(
            mode="json",
            by_alias=True,
        )
        return updated

    def _record_skipped_root(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        failed_dependencies: list[str],
    ) -> Blueprint:
        root = self.build_root_node(problem=problem, blueprint=blueprint)
        root.status = NodeStatus.FAILED
        root.metadata["skipped_reason"] = (
            "dependencies not proved: " + ", ".join(failed_dependencies)
        )
        root.metadata["attempt_budget"] = self.attempts_per_node
        root.metadata["attempts_executed"] = 0
        root.metadata["successful_attempt_ids"] = []
        root.lean_feedback.append(
            LeanFeedback(
                stage="dependency_gate",
                success=False,
                message=str(root.metadata["skipped_reason"]),
                error_type=LeanVerificationStatus.INTERNAL_ERROR.value,
                error_detail=LeanFailureDetail.INTERNAL_EXCEPTION.value,
            )
        )
        updated = blueprint.model_copy(deep=True)
        updated.metadata["root_node"] = root.model_dump(
            mode="json",
            by_alias=True,
        )
        return updated

    def _record_attempt(
        self,
        *,
        request: NodeProofRequest,
        prompt: str,
        attempt_id: str,
        model: str,
        prompt_type: str,
        proof: str,
        success: bool | None,
        status: str,
        verification_status: str = LeanVerificationStatus.SUCCESS.value,
        failure_detail: str = LeanFailureDetail.NONE.value,
        diagnostics: str = "",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = ProverAttemptRecord(
            event_id=f"T-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            problem_id=request.problem_id,
            problem_hash=request.problem_hash,
            node_id=request.node_id,
            attempt_id=attempt_id,
            model=model,
            prompt_type=prompt_type,
            prompt=prompt,
            request=_request_to_json(request),
            proof=proof,
            success=success,
            status=status,
            verification_status=verification_status,
            failure_detail=failure_detail,
            diagnostics=diagnostics,
            error_message=error_message,
            metadata=metadata,
        )
        if self.telemetry_store is not None:
            self.telemetry_store.record_attempt(record)
        return record.to_json()

    def _compile_repair_candidate(
        self,
        *,
        request: NodeProofRequest,
        node: BlueprintNode,
        repaired: GeneratedProof,
        base_attempt_id: str,
    ) -> tuple[LeanFeedback, dict[str, object]]:
        """Compile and persist one repair candidate as its own full attempt."""

        metadata = dict(repaired.metadata or {})
        metadata.setdefault("base_attempt_id", base_attempt_id)
        repair_attempt = ProofAttempt(
            attempt_id=f"A-{uuid.uuid4().hex[:8]}",
            model=repaired.model,
            prompt_type=repaired.prompt_type,
            proof=repaired.proof,
            score=repaired.score,
            metadata=metadata,
        )
        if self.verifier is None:
            repaired_feedback = LeanFeedback(
                stage="proof_repair_verification",
                success=False,
                message="Pantograph verifier is unavailable",
                error_type=LeanVerificationStatus.INTERNAL_ERROR.value,
                error_detail=LeanFailureDetail.INTERNAL_EXCEPTION.value,
            )
        else:
            repaired_feedback = self.verifier.verify(
                request=request,
                proof=repaired.proof,
            )
        repaired_feedback = enforce_repair_grounding(
            repaired_feedback,
            metadata,
        )
        repair_attempt.success = repaired_feedback.success
        repair_attempt.diagnostics = repaired_feedback.diagnostics
        repair_attempt.error_type = repaired_feedback.error_type
        repair_attempt.error_detail = repaired_feedback.error_detail
        repair_record = self._record_attempt(
            request=request,
            prompt=str(metadata.get("prompt") or ""),
            attempt_id=repair_attempt.attempt_id,
            model=repaired.model,
            prompt_type=repaired.prompt_type,
            proof=repaired.proof,
            success=repaired_feedback.success,
            status=(
                "repair_proved"
                if repaired_feedback.success
                else "repair_verification_failed"
            ),
            verification_status=repaired_feedback.error_type,
            failure_detail=repaired_feedback.error_detail,
            diagnostics=repaired_feedback.diagnostics,
            error_message=(
                "" if repaired_feedback.success else repaired_feedback.message
            ),
            metadata=metadata,
        )
        repair_attempt.metadata.setdefault(
            "telemetry_event_id",
            repair_record["event_id"],
        )
        node.proof_attempts.append(repair_attempt)
        node.lean_feedback.append(repaired_feedback)
        outcome = {
            "attempt_id": repair_attempt.attempt_id,
            "candidate_id": metadata.get("candidate_id"),
            "repair_round": metadata.get("repair_round"),
            "proof": repaired.proof,
            "success": repaired_feedback.success,
            "error_type": repaired_feedback.error_type,
            "error_detail": repaired_feedback.error_detail,
            "diagnostics": repaired_feedback.diagnostics,
        }
        if repaired_feedback.success:
            node.verified_proof = repaired.proof
            node.status = NodeStatus.PROVED
        return repaired_feedback, outcome

    def _prove_node_once(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        node_id: str,
        allow_repair: bool = True,
    ) -> Blueprint:
        if blueprint.problem_hash != problem.problem_hash:
            raise ValueError("Blueprint problem_hash differs from the problem")
        updated = blueprint.model_copy(deep=True)
        nodes = self._nodes(updated)
        if node_id not in nodes:
            raise KeyError(f"unknown Blueprint node: {node_id}")
        node = nodes[node_id]
        node.status = NodeStatus.PROVING
        request = self.build_request(
            problem=problem,
            blueprint=updated,
            node=node,
        )
        prompt = build_node_proof_prompt(request)
        try:
            generated = self.generator.generate(request)
        except Exception as error:
            classification = classify_lean_diagnostics(
                str(error),
                extraction_error=isinstance(error, ProofGenerationError),
                internal_error=not isinstance(error, ProofGenerationError),
            )
            attempt_id = f"A-{uuid.uuid4().hex[:8]}"
            attempt = ProofAttempt(
                attempt_id=attempt_id,
                model=str(getattr(self.generator, "model", "unknown-api")),
                prompt_type="proof_generation_error",
                proof="",
                success=False,
                diagnostics=str(error),
                error_type=classification.status.value,
                error_detail=classification.detail.value,
            )
            node.lean_feedback.append(
                LeanFeedback(
                    stage="proof_generation",
                    success=False,
                    message=str(error),
                    diagnostics=str(error),
                    error_type=classification.status.value,
                    error_detail=classification.detail.value,
                )
            )
            node.status = NodeStatus.FAILED
            record = self._record_attempt(
                request=request,
                prompt=prompt,
                attempt_id=attempt_id,
                model=str(getattr(self.generator, "model", "unknown-api")),
                prompt_type="proof_generation_error",
                proof="",
                success=False,
                status="generation_failed",
                verification_status=classification.status.value,
                failure_detail=classification.detail.value,
                error_message=str(error),
            )
            attempt.metadata.setdefault("telemetry_event_id", record["event_id"])
            node.proof_attempts.append(attempt)
            return updated

        attempt = ProofAttempt(
            attempt_id=f"A-{uuid.uuid4().hex[:8]}",
            model=generated.model,
            prompt_type=generated.prompt_type,
            proof=generated.proof,
            score=generated.score,
            metadata=generated.metadata or {},
        )
        attempt.metadata.setdefault("prompt", prompt)
        if self.verifier is None:
            node.proof_attempts.append(attempt)
            node.status = NodeStatus.PROVING
            record = self._record_attempt(
                request=request,
                prompt=prompt,
                attempt_id=attempt.attempt_id,
                model=generated.model,
                prompt_type=generated.prompt_type,
                proof=generated.proof,
                success=None,
                status="generated_unverified",
                verification_status=LeanVerificationStatus.INTERNAL_ERROR.value,
                failure_detail=LeanFailureDetail.INTERNAL_EXCEPTION.value,
                metadata=attempt.metadata,
            )
            attempt.error_type = LeanVerificationStatus.INTERNAL_ERROR.value
            attempt.error_detail = LeanFailureDetail.INTERNAL_EXCEPTION.value
            attempt.metadata.setdefault(
                "telemetry_event_id",
                record["event_id"],
            )
            return updated

        feedback = self.verifier.verify(request=request, proof=generated.proof)
        attempt.success = feedback.success
        attempt.diagnostics = feedback.diagnostics
        attempt.error_type = feedback.error_type
        attempt.error_detail = feedback.error_detail
        record = self._record_attempt(
            request=request,
            prompt=prompt,
            attempt_id=attempt.attempt_id,
            model=generated.model,
            prompt_type=generated.prompt_type,
            proof=generated.proof,
            success=feedback.success,
            status="proved" if feedback.success else "verification_failed",
            verification_status=feedback.error_type,
            failure_detail=feedback.error_detail,
            diagnostics=feedback.diagnostics,
            error_message="" if feedback.success else feedback.message,
            metadata=attempt.metadata,
        )
        attempt.metadata.setdefault("telemetry_event_id", record["event_id"])
        node.proof_attempts.append(attempt)
        node.lean_feedback.append(feedback)
        if feedback.success:
            node.verified_proof = generated.proof
            node.status = NodeStatus.PROVED
            return updated

        node.status = NodeStatus.FAILED
        if self.proof_repairer is None or not allow_repair:
            return updated

        generate_candidates = getattr(
            self.proof_repairer,
            "generate_candidates",
            None,
        )
        if callable(generate_candidates):
            failed_proof = generated.proof
            failed_feedback = feedback
            repair_history: list[dict[str, object]] = []
            for round_index in range(1, self.max_proof_repair_rounds + 1):
                try:
                    candidates = generate_candidates(
                        request=request,
                        failed_proof=failed_proof,
                        feedback=failed_feedback,
                        round_index=round_index,
                        repair_history=repair_history,
                    )
                except Exception as error:
                    classification = classify_lean_diagnostics(
                        str(error),
                        internal_error=True,
                    )
                    node.lean_feedback.append(
                        LeanFeedback(
                            stage="proof_repair_generation",
                            success=False,
                            message=str(error),
                            diagnostics=str(error),
                            error_type=classification.status.value,
                            error_detail=classification.detail.value,
                        )
                    )
                    break
                round_outcomes: list[dict[str, object]] = []
                if not candidates:
                    node.lean_feedback.append(
                        LeanFeedback(
                            stage="proof_repair_generation",
                            success=False,
                            message=(
                                f"repair round {round_index} returned no candidates"
                            ),
                            error_type=LeanVerificationStatus.INTERNAL_ERROR.value,
                            error_detail=LeanFailureDetail.INTERNAL_EXCEPTION.value,
                        )
                    )
                    break
                for repaired in candidates:
                    repaired_feedback, outcome = self._compile_repair_candidate(
                        request=request,
                        node=node,
                        repaired=repaired,
                        base_attempt_id=attempt.attempt_id,
                    )
                    round_outcomes.append(outcome)
                    if repaired_feedback.success:
                        node.metadata.setdefault("proof_repair_audit", []).append(
                            {
                                "base_attempt_id": attempt.attempt_id,
                                "round": round_index,
                                "candidate_outcomes": round_outcomes,
                                "success": True,
                            }
                        )
                        return updated
                    failed_proof = repaired.proof
                    failed_feedback = repaired_feedback
                repair_history.append(
                    {
                        "round": round_index,
                        "candidate_outcomes": round_outcomes,
                    }
                )
                node.metadata.setdefault("proof_repair_audit", []).append(
                    {
                        "base_attempt_id": attempt.attempt_id,
                        "round": round_index,
                        "candidate_outcomes": round_outcomes,
                        "success": False,
                    }
                )
            return updated

        # Compatibility path for an existing repairer that returns one proof.
        try:
            repaired = self.proof_repairer.repair(
                request=request,
                failed_proof=generated.proof,
                feedback=feedback,
            )
        except Exception as error:
            classification = classify_lean_diagnostics(
                str(error),
                internal_error=True,
            )
            node.lean_feedback.append(
                LeanFeedback(
                    stage="proof_repair",
                    success=False,
                    message=str(error),
                    diagnostics=str(error),
                    error_type=classification.status.value,
                    error_detail=classification.detail.value,
                )
            )
            return updated

        self._compile_repair_candidate(
            request=request,
            node=node,
            repaired=repaired,
            base_attempt_id=attempt.attempt_id,
        )
        return updated

    def prove_node(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
        node_id: str,
    ) -> Blueprint:
        """Run pass@k for one node and retain every generated attempt."""

        updated = blueprint
        candidate_repair = callable(
            getattr(self.proof_repairer, "generate_candidates", None)
        )
        starting_attempt_count = len(
            self._nodes(updated)[node_id].proof_attempts
        )
        for _ in range(self.attempts_per_node):
            updated = self._prove_node_once(
                problem=problem,
                blueprint=updated,
                node_id=node_id,
                # A structured multi-round repairer consumes one shared node
                # budget after all four independent pass@k generations.  This
                # avoids multiplying the configured repair rounds by every
                # independent base attempt.
                allow_repair=not candidate_repair,
            )

        node = self._nodes(updated)[node_id]
        base_attempts = node.proof_attempts[
            starting_attempt_count:
            starting_attempt_count + self.attempts_per_node
        ]
        if candidate_repair and not any(
            attempt.success for attempt in base_attempts
        ):
            failed = next(
                (
                    attempt
                    for attempt in reversed(base_attempts)
                    if attempt.proof and proof_repair_eligible(attempt)
                ),
                None,
            )
            if failed is not None:
                request = self.build_request(
                    problem=problem,
                    blueprint=updated,
                    node=node,
                )
                failed_feedback = LeanFeedback(
                    stage="proof_verification",
                    success=False,
                    message="Lean rejected proof",
                    diagnostics=failed.diagnostics,
                    error_type=failed.error_type,
                    error_detail=failed.error_detail,
                )
                # Reuse the candidate loop in _prove_node_once without making
                # another base API call: a tiny adapter supplies the selected
                # failed attempt to the loop below.
                generate_candidates = getattr(
                    self.proof_repairer,
                    "generate_candidates",
                )
                repair_history: list[dict[str, object]] = []
                failed_proof = failed.proof
                candidates_checked = 0
                for round_index in range(1, self.max_proof_repair_rounds + 1):
                    try:
                        candidates = generate_candidates(
                            request=request,
                            failed_proof=failed_proof,
                            feedback=failed_feedback,
                            round_index=round_index,
                            repair_history=repair_history,
                        )
                    except Exception as error:
                        classification = classify_lean_diagnostics(
                            str(error),
                            internal_error=True,
                        )
                        node.lean_feedback.append(
                            LeanFeedback(
                                stage="proof_repair_generation",
                                success=False,
                                message=str(error),
                                diagnostics=str(error),
                                error_type=classification.status.value,
                                error_detail=classification.detail.value,
                            )
                        )
                        break
                    if not candidates:
                        node.lean_feedback.append(
                            LeanFeedback(
                                stage="proof_repair_generation",
                                success=False,
                                message=(
                                    f"repair round {round_index} returned no candidates"
                                ),
                                error_type=(
                                    LeanVerificationStatus.INTERNAL_ERROR.value
                                ),
                                error_detail=(
                                    LeanFailureDetail.INTERNAL_EXCEPTION.value
                                ),
                            )
                        )
                        break
                    round_outcomes: list[dict[str, object]] = []
                    round_success = False
                    for repaired in candidates[:1]:
                        if (
                            candidates_checked
                            >= self.max_proof_repair_candidates_per_node
                        ):
                            break
                        candidates_checked += 1
                        repaired_feedback, outcome = (
                            self._compile_repair_candidate(
                                request=request,
                                node=node,
                                repaired=repaired,
                                base_attempt_id=failed.attempt_id,
                            )
                        )
                        round_outcomes.append(outcome)
                        if repaired_feedback.success:
                            round_success = True
                            break
                        failed_proof = repaired.proof
                        failed_feedback = repaired_feedback
                    repair_history.append(
                        {
                            "round": round_index,
                            "candidate_outcomes": round_outcomes,
                        }
                    )
                    node.metadata.setdefault("proof_repair_audit", []).append(
                        {
                            "base_attempt_id": failed.attempt_id,
                            "round": round_index,
                            "candidate_outcomes": round_outcomes,
                            "success": round_success,
                        }
                    )
                    if (
                        round_success
                        or candidates_checked
                        >= self.max_proof_repair_candidates_per_node
                        or not proof_repair_eligible(failed_feedback)
                    ):
                        break
                node.metadata["proof_repair_history"] = repair_history

        successful_attempts = [
            attempt for attempt in node.proof_attempts if attempt.success
        ]
        node.metadata["attempt_budget"] = self.attempts_per_node
        node.metadata["attempts_executed"] = len(node.proof_attempts)
        node.metadata["successful_attempt_ids"] = [
            attempt.attempt_id for attempt in successful_attempts
        ]
        if successful_attempts and node.verified_proof:
            node.status = NodeStatus.PROVED
        elif self.verifier is None and any(
            attempt.proof for attempt in node.proof_attempts
        ):
            node.status = NodeStatus.PROVING
        else:
            node.status = NodeStatus.FAILED
        return updated

    def prove_blueprint(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> Blueprint:
        updated = blueprint
        for node_id in topological_order(updated):
            if node_id == "ROOT":
                nodes = self._nodes(updated)
                failed_dependencies = [
                    dependency
                    for dependency in updated.root_dependencies
                    if (
                        dependency not in nodes
                        or nodes[dependency].status != NodeStatus.PROVED
                    )
                ]
                if failed_dependencies:
                    updated = self._record_skipped_root(
                        problem=problem,
                        blueprint=updated,
                        failed_dependencies=failed_dependencies,
                    )
                else:
                    updated = self._prove_root(
                        problem=problem,
                        blueprint=updated,
                    )
                continue
            nodes = self._nodes(updated)
            failed_dependencies = [
                dependency
                for dependency in nodes[node_id].depends_on
                if nodes[dependency].status != NodeStatus.PROVED
            ]
            if failed_dependencies:
                nodes[node_id].status = NodeStatus.FAILED
                nodes[node_id].metadata["skipped_reason"] = (
                    "dependencies not proved: "
                    + ", ".join(failed_dependencies)
                )
                nodes[node_id].lean_feedback.append(
                    LeanFeedback(
                        stage="dependency_gate",
                        success=False,
                        message=nodes[node_id].metadata["skipped_reason"],
                        error_type=LeanVerificationStatus.INTERNAL_ERROR.value,
                        error_detail=LeanFailureDetail.INTERNAL_EXCEPTION.value,
                    )
                )
                continue
            updated = self.prove_node(
                problem=problem,
                blueprint=updated,
                node_id=node_id,
            )
        return updated

    def prove(
        self,
        *,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> ProverProblemResult:
        """Prove all nodes, apply the all-node gate, and route one result."""

        proved = self.prove_blueprint(problem=problem, blueprint=blueprint)
        result = collect_prover_result(problem=problem, blueprint=proved)
        if self.result_store is not None:
            self.result_store.record_result(result)
        return result
