# Define the data structures used in the Planner.

from __future__ import annotations

from enum import Enum
import re

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from lean_prover.identity import compute_input_hash, compute_problem_hash


class NodeStatus(str, Enum):
    PLANNED = "planned"
    STATEMENT_VALID = "statement_valid"
    READY = "ready"
    PROVING = "proving"
    PROVED = "proved"
    FAILED = "failed"


class ProblemInputKind(str, Enum):
    NATURAL_LANGUAGE = "natural_language"
    LEAN = "lean"


class RawTheoremInput(BaseModel):
    """Unclassified theorem input accepted by the Planner API gate."""

    input_text: str = ""
    formal_statement: str | None = None
    informal_stmt: str | None = None
    header: str = ""
    problem_id: str | None = None
    imports: list[str] = Field(default_factory=lambda: ["Mathlib"])
    available_definitions: list[str] = Field(default_factory=list)
    input_hash: str = ""

    @model_validator(mode="after")
    def assign_input_hash(self) -> "RawTheoremInput":
        if not any(
            value and value.strip()
            for value in (
                self.input_text,
                self.formal_statement,
                self.informal_stmt,
            )
        ):
            raise ValueError("at least one theorem input field must be non-empty")
        if not self.input_text.strip():
            self.input_text = (
                self.formal_statement or self.informal_stmt or ""
            ).strip()
        expected = compute_input_hash(
            input_text=self.input_text,
            imports=self.imports,
            available_definitions=self.available_definitions,
            formal_statement=self.formal_statement,
            informal_statement=self.informal_stmt,
            header=self.header,
        )
        if self.input_hash and self.input_hash != expected:
            raise ValueError("input_hash does not match normalized input")
        self.input_hash = expected
        return self


class InputClassification(BaseModel):
    input_kind: ProblemInputKind
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class LeanToNaturalTranslation(BaseModel):
    natural_language_statement: str = Field(min_length=1)
    semantic_alignment_notes: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class TargetFormalization(BaseModel):
    target_lean_decl: str = Field(min_length=1)
    semantic_alignment_notes: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class LeanFeedback(BaseModel):
    """Lean/Pantograph feedback attached to one proof attempt or node."""

    stage: str
    success: bool
    message: str
    diagnostics: str = ""
    error_type: str = "success"
    error_detail: str = "none"


class ProofAttempt(BaseModel):
    """One model-generated proof body and its verifier result."""

    attempt_id: str
    model: str
    prompt_type: str = "proof_generation"
    proof: str
    success: bool = False
    diagnostics: str = ""
    error_type: str = "success"
    error_detail: str = "none"
    score: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ProofLengthEstimate(BaseModel):
    """Estimated size of the eventual Lean proof used to tune plan granularity."""

    estimated_lines: int = Field(ge=1, le=200)
    estimated_tokens: int = Field(ge=1, le=2048)
    rationale: str = Field(min_length=1)


class BlueprintPlanNode(BaseModel):
    """Natural-language-only node produced by the decomposition stage."""

    id: str = Field(description="Node ID, e.g., L1")
    title: str = Field(min_length=1)
    informal_statement: str = Field(min_length=1)
    informal_proof: str = Field(
        min_length=1,
        description=(
            "Concrete natural-language proof sketch. Every declared dependency "
            "must be cited by its exact node ID in backticks."
        ),
    )
    logical_ideas: list[str] = Field(
        min_length=1,
        max_length=3,
        description="The one to three new logical ideas introduced by this node.",
    )
    lean_statement: str = ""
    depends_on: list[str] = Field(default_factory=list)
    proof_strategy: str = Field(min_length=1)
    estimated_proof_length: ProofLengthEstimate
    difficulty: int = Field(default=3, ge=1, le=5)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if value != "ROOT" and not re.fullmatch(r"L[1-9][0-9]*", value):
            raise ValueError(
                "Node ID must be ROOT or L followed by a positive integer"
            )
        return value

    @field_validator("lean_statement")
    @classmethod
    def validate_empty_lean_statement(cls, value: str) -> str:
        if value.strip():
            raise ValueError(
                "decomposition-stage lean_statement must be the empty string"
            )
        return ""

    @field_validator("logical_ideas")
    @classmethod
    def validate_logical_ideas(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("logical_ideas entries must be non-empty")
        return normalized


class BlueprintPlan(BaseModel):
    """Stage-one Blueprint containing only mathematical decomposition."""

    blueprint_summary: str = Field(min_length=1)
    nodes: list[BlueprintPlanNode]
    root_dependencies: list[str]
    problem_hash: str = ""
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("problem_hash")
    @classmethod
    def validate_problem_hash(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("problem_hash must be a lowercase SHA-256 hex digest")
        return value


class LeanEnvironmentIdentity(BaseModel):
    """Exact local Lean/mathlib identity supplied to the formalizer."""

    lean_version: str = Field(min_length=1)
    lean_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    mathlib_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class LeanPreamble(BaseModel):
    """Structured, proof-free preamble needed to elaborate one node."""

    imports: list[str] = Field(min_length=1)
    raw_header: str = ""
    namespaces: list[str] = Field(default_factory=list)
    open_namespaces: list[str] = Field(default_factory=list)
    open_scoped: list[str] = Field(default_factory=list)
    variable_declarations: list[str] = Field(default_factory=list)
    local_context: list[str] = Field(default_factory=list)

    @field_validator("imports")
    @classmethod
    def validate_imports(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            module = value.strip()
            if module.startswith("import "):
                module = module.removeprefix("import ").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*(\.[A-Za-z_][A-Za-z0-9_']*)*", module):
                raise ValueError(f"invalid Lean import module: {value!r}")
            if module not in normalized:
                normalized.append(module)
        return normalized

    @field_validator("namespaces", "open_namespaces", "open_scoped")
    @classmethod
    def validate_namespace_names(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            name = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'.]*(\.[A-Za-z_][A-Za-z0-9_']*)*", name):
                raise ValueError(f"invalid Lean namespace/scoped name: {value!r}")
            if name not in normalized:
                normalized.append(name)
        return normalized

    @field_validator("variable_declarations")
    @classmethod
    def validate_variable_declarations(cls, values: list[str]) -> list[str]:
        return cls._validate_context_lines(values, required_prefix="variable ")

    @field_validator("local_context")
    @classmethod
    def validate_local_context(cls, values: list[str]) -> list[str]:
        return cls._validate_context_lines(values)

    @field_validator("raw_header")
    @classmethod
    def validate_raw_header(cls, value: str) -> str:
        header = value.strip()
        forbidden = re.compile(
            r"\b(sorry|admit|axiom|unsafe)\b|:=|\bby\b",
            flags=re.IGNORECASE,
        )
        match = forbidden.search(header)
        if match:
            raise ValueError(
                f"proof or unsafe content in raw header: {match.group(0)!r}"
            )
        return header

    @staticmethod
    def _validate_context_lines(
        values: list[str],
        *,
        required_prefix: str | None = None,
    ) -> list[str]:
        forbidden = re.compile(
            r"\b(sorry|admit|axiom|unsafe)\b|:=|\bby\b",
            flags=re.IGNORECASE,
        )
        normalized: list[str] = []
        for value in values:
            line = value.strip()
            if not line:
                raise ValueError("preamble lines must not be empty")
            if required_prefix and not line.startswith(required_prefix):
                raise ValueError(
                    f"preamble line must start with {required_prefix!r}: {line!r}"
                )
            if forbidden.search(line):
                raise ValueError(f"proof or unsafe content in preamble: {line!r}")
            if required_prefix is None and re.match(
                r"^(?:import|namespace|end|open|variable)\b",
                line,
            ):
                raise ValueError(
                    "imports, namespace scopes, open commands, variable "
                    "commands, and scope-closing commands must use their "
                    f"dedicated preamble fields: {line!r}"
                )
            if required_prefix is None and re.match(
                r"^[A-Za-z_][A-Za-z0-9_']*\s*:",
                line,
            ):
                raise ValueError(
                    "a bare hypothesis is not a top-level preamble command; "
                    f"put it in the theorem/lemma binders: {line!r}"
                )
            normalized.append(line)
        return normalized


class SemanticAlignment(BaseModel):
    """Audit showing that a formal node preserves its informal proposition."""

    objects: list[str] = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list)
    conclusion: str = Field(min_length=1)
    alignment_notes: str = Field(
        min_length=1,
        description=(
            "Explain representation, typing, or coercion choices and why the "
            "Lean statement neither strengthens nor weakens the informal node."
        ),
    )


class VerifyIssue(BaseModel):
    """One actionable semantic or dependency defect found before compilation."""

    component: str = Field(pattern=r"^(dependencies|formal_statement)$")
    message: str = Field(min_length=1)
    repair_reference: str = Field(min_length=1)


class BlueprintNodeVerification(BaseModel):
    """Structured output of one independent Verify API call."""

    node_id: str
    corrected_preamble: LeanPreamble
    header_changes: list[str] = Field(default_factory=list)
    dependency_statements_correct: bool
    dependency_issues: list[VerifyIssue] = Field(default_factory=list)
    formal_statement_correct: bool
    formal_statement_issues: list[VerifyIssue] = Field(default_factory=list)
    verification_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_issue_consistency(self) -> "BlueprintNodeVerification":
        if self.dependency_statements_correct and self.dependency_issues:
            raise ValueError(
                "dependency_issues must be empty when dependencies are correct"
            )
        if not self.dependency_statements_correct and not self.dependency_issues:
            raise ValueError(
                "dependency_issues are required when dependencies are incorrect"
            )
        if self.formal_statement_correct and self.formal_statement_issues:
            raise ValueError(
                "formal_statement_issues must be empty when statement is correct"
            )
        if not self.formal_statement_correct and not self.formal_statement_issues:
            raise ValueError(
                "formal_statement_issues are required when statement is incorrect"
            )
        if any(issue.component != "dependencies" for issue in self.dependency_issues):
            raise ValueError("dependency_issues must use component=dependencies")
        if any(
            issue.component != "formal_statement"
            for issue in self.formal_statement_issues
        ):
            raise ValueError(
                "formal_statement_issues must use component=formal_statement"
            )
        return self


class BlueprintNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Node ID, e.g., L1")
    title: str
    informal_statement: str
    informal_proof: str = Field(
        min_length=1,
        description=(
            "Concrete natural-language proof sketch. Every declared dependency "
            "must be cited by its exact node ID in backticks."
        ),
    )
    logical_ideas: list[str] = Field(
        min_length=1,
        max_length=3,
        description="The one to three new logical ideas introduced by this node.",
    )
    lean_decl: str = Field(
        validation_alias=AliasChoices("lean_statement", "lean_decl"),
        serialization_alias="lean_statement",
    )
    preamble: LeanPreamble
    semantic_alignment: SemanticAlignment
    estimated_proof_length: ProofLengthEstimate

    depends_on: list[str] = Field(default_factory=list)
    proof_strategy: str = ""
    mathlib_hints: list[str] = Field(default_factory=list)

    difficulty: int = Field(default=3, ge=1, le=5)
    status: NodeStatus = NodeStatus.PLANNED
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    local_context: list[str] = Field(default_factory=list)
    proof_attempts: list[ProofAttempt] = Field(default_factory=list)
    verified_proof: str | None = None
    lean_feedback: list[LeanFeedback] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    dependency_statements_verified: bool | None = None
    formal_statement_verified: bool | None = None
    verification_issues: list[VerifyIssue] = Field(default_factory=list)
    verification_notes: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    # Planner nodes use L1, L2, ...; Prover additionally materializes ROOT.
    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if value != "ROOT" and not re.fullmatch(r"L[1-9][0-9]*", value):
            raise ValueError(
                "Node ID must be ROOT or L followed by a positive integer"
            )
        return value

    # check if lean_decl contains forbidden tokens and starts with "theorem" or "lemma"
    @field_validator("lean_decl")
    @classmethod
    def validate_lean_decl(cls, value: str) -> str:
        stripped = value.strip()
        forbidden = re.compile(
            r"\b(sorry|admit|axiom|unsafe)\b|:=|\bby\b",
            flags=re.IGNORECASE,
        )
        match = forbidden.search(stripped)
        if match:
            raise ValueError(
                f"lean_statement contains forbidden proof token: {match.group(0)}"
            )
        if not stripped.startswith(("theorem ", "lemma ")):
            raise ValueError(
                "lean_statement must begin with theorem or lemma"
            )
        return stripped

    @field_validator("logical_ideas")
    @classmethod
    def validate_logical_ideas(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("logical_ideas entries must be non-empty")
        return normalized


class Blueprint(BaseModel):
    blueprint_summary: str
    nodes: list[BlueprintNode]
    root_dependencies: list[str]
    environment: LeanEnvironmentIdentity
    problem_hash: str = ""
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("problem_hash")
    @classmethod
    def validate_problem_hash(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("problem_hash must be a lowercase SHA-256 hex digest")
        return value


class BlueprintRepairState(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class BlueprintRepairOutput(Blueprint):
    """One BluePrintRepair round: a complete Blueprint plus audit state."""

    state: BlueprintRepairState


class TheoremProblem(BaseModel):
    problem_id: str
    imports: list[str] = Field(default_factory=lambda: ["Mathlib"]) # imports should always include Mathlib
    natural_language_statement: str | None = None
    target_lean_decl: str
    header: str = ""
    available_definitions: list[str] = Field(default_factory=list)
    input_hash: str = ""
    problem_hash: str = ""

    @field_validator("input_hash")
    @classmethod
    def validate_input_hash(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("input_hash must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def assign_problem_hash(self) -> "TheoremProblem":
        expected = compute_problem_hash(
            input_hash=self.input_hash,
            natural_language_statement=self.natural_language_statement,
            target_lean_decl=self.target_lean_decl,
            imports=self.imports,
            available_definitions=self.available_definitions,
            header=self.header,
        )
        if self.problem_hash and self.problem_hash != expected:
            raise ValueError("problem_hash does not match normalized problem")
        self.problem_hash = expected
        return self


class LeanCheckResult(BaseModel):
    success: bool
    declaration: str
    stdout: str = ""
    stderr: str = ""
    error_message: str | None = None


class NodeLeanCheckResult(BaseModel):
    node_id: str
    success: bool
    lean_statement: str
    preamble: LeanPreamble
    result: LeanCheckResult


class ValidationIssue(BaseModel):
    stage: str
    code: str
    message: str
    node_id: str | None = None


class BlueprintRepairRound(BaseModel):
    """Auditable result of one stateless BluePrintRepair inspection."""

    round_id: int = Field(ge=1, le=3)
    state: BlueprintRepairState | None = None
    input_was_valid_blueprint: bool
    deterministic_issues_before: list[ValidationIssue] = Field(
        default_factory=list
    )
    changed: bool
    changed_fields: list[str] = Field(default_factory=list)
    input_snapshot: dict[str, object] | str | None = None
    blueprint: Blueprint | None = None
    raw_output: dict[str, object] | str | None = None
    output_error: str | None = None


class BlueprintLeanCheckResult(BaseModel):
    success: bool
    node_results: list[NodeLeanCheckResult] = Field(default_factory=list)
    failed_node_ids: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)


class BlueprintValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class PlannerResult(BaseModel):
    success: bool
    stage: str = "unknown"
    problem: TheoremProblem | None = None
    input_classification: InputClassification | None = None
    planner_mode: ProblemInputKind | None = None
    input_hash: str = ""
    plan: BlueprintPlan | None = None
    blueprint: Blueprint | None = None
    attempts: int = 0
    decomposition_attempts: int = 0
    formalization_attempts: int = 0
    classification_attempts: int = 0
    translation_attempts: int = 0
    target_formalization_attempts: int = 0
    verify_attempts: int = 0
    blueprint_repair_attempts: int = 0
    environment: LeanEnvironmentIdentity | None = None
    target_check: LeanCheckResult | None = None
    node_checks: list[NodeLeanCheckResult] = Field(default_factory=list)
    failed_node_ids: list[str] = Field(default_factory=list)
    verify_results: list[BlueprintNodeVerification] = Field(default_factory=list)
    blueprint_repair_rounds: list[BlueprintRepairRound] = Field(
        default_factory=list
    )
    issues: list[ValidationIssue] = Field(default_factory=list)
