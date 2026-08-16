"""Data contracts for the isolated root-first experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lean_prover.Planner.schemas import (
    LeanEnvironmentIdentity,
    LeanFeedback,
    LeanPreamble,
    ProofLengthEstimate,
    SemanticAlignment,
)


class NodeState(str, Enum):
    PENDING = "pending"
    FAILED = "failed"
    SUCCESS = "success"
    DISPROVED = "disproved"


class RootFirstAttempt(BaseModel):
    attempt_id: str
    proof_invocation: int = Field(default=0, ge=0)
    attempt_index: int = Field(ge=1)
    kind: Literal[
        "proof",
        "proof_repair",
        "disproof",
        "disproof_repair",
        "generation_error",
    ]
    repair_round: int | None = None
    candidate_id: str | None = None
    proof: str = ""
    success: bool = False
    pantograph: LeanFeedback
    model: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RootFirstNode(BaseModel):
    id: str
    title: str
    informal_statement: str
    informal_proof: str
    logical_ideas: list[str] = Field(min_length=1, max_length=3)
    lean_statement: str
    father_nodes: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    preamble: LeanPreamble
    semantic_alignment: SemanticAlignment
    estimated_proof_length: ProofLengthEstimate
    proof_strategy: str = ""
    mathlib_hints: list[str] = Field(default_factory=list)
    state: NodeState = NodeState.PENDING
    frozen: bool = False
    verified_proof: str | None = None
    formal_disproof: str | None = None
    attempts: list[RootFirstAttempt] = Field(default_factory=list)
    last_diagnostics: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not re.fullmatch(r"L(?:0|[1-9][0-9]*)", value):
            raise ValueError("RootFirst node IDs must be L0, L1, L2, ...")
        return value

    @field_validator("lean_statement")
    @classmethod
    def proof_free_statement(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith(("theorem ", "lemma ")):
            raise ValueError("lean_statement must begin with theorem or lemma")
        if re.search(
            r"\b(?:sorry|admit|axiom|unsafe)\b|:=|\bby\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            raise ValueError("lean_statement must contain no proof body")
        return stripped

    @field_validator("logical_ideas")
    @classmethod
    def nonempty_ideas(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not item for item in normalized):
            raise ValueError("logical_ideas entries must be non-empty")
        return normalized

    @field_validator("father_nodes", "children")
    @classmethod
    def unique_edges(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("edge lists must not contain duplicates")
        return values

    @model_validator(mode="after")
    def state_invariants(self) -> "RootFirstNode":
        if self.state == NodeState.SUCCESS:
            if not self.verified_proof:
                raise ValueError("success node requires verified_proof")
            if not self.frozen:
                raise ValueError("success node must be frozen")
        if self.state == NodeState.DISPROVED and not self.formal_disproof:
            raise ValueError("disproved node requires a Pantograph-verified disproof")
        return self

    def frozen_content(self) -> dict[str, object]:
        """Content that may not change once a node is proved.

        Relationship fields are deliberately excluded: a later refinement may
        insert a node between this node and one of its children.
        """

        return {
            "id": self.id,
            "title": self.title,
            "informal_statement": self.informal_statement,
            "informal_proof": self.informal_proof,
            "logical_ideas": self.logical_ideas,
            "lean_statement": self.lean_statement,
            "preamble": self.preamble.model_dump(mode="json"),
            "semantic_alignment": self.semantic_alignment.model_dump(mode="json"),
            "estimated_proof_length": self.estimated_proof_length.model_dump(
                mode="json"
            ),
            "proof_strategy": self.proof_strategy,
            "mathlib_hints": self.mathlib_hints,
            "verified_proof": self.verified_proof,
        }

    def frozen_fingerprint(self) -> str:
        payload = json.dumps(
            self.frozen_content(), ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RootFirstBlueprint(BaseModel):
    problem_id: str
    problem_hash: str
    root_id: Literal["L0"] = "L0"
    original_target: str
    nodes: list[RootFirstNode]
    environment: LeanEnvironmentIdentity
    refinement_round: int = Field(default=0, ge=0, le=3)
    nodes_added: int = Field(default=0, ge=0, le=3)
    metadata: dict[str, object] = Field(default_factory=dict)

    def node_map(self) -> dict[str, RootFirstNode]:
        return {node.id: node for node in self.nodes}


class RefinementNodeValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(max_length=16)
    informal_statement: str = Field(max_length=1200)
    lean_statement: str = Field(max_length=2400)
    father_nodes: list[str]
    children: list[str]


class RevisedNodeValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    informal_statement: str = Field(max_length=1200)
    lean_statement: str = Field(max_length=2400)


class RefinementPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["failed", "disproved"]
    action: Literal["add_node", "revise_disproved_node"]
    target_node_id: str
    new_node: RefinementNodeValues | None = None
    revised_node: RevisedNodeValues | None = None

    @model_validator(mode="after")
    def action_payload(self) -> "RefinementPatch":
        if self.action == "add_node":
            if self.state != "failed" or self.new_node is None:
                raise ValueError("add_node requires state=failed and new_node")
            if self.revised_node is not None:
                raise ValueError("add_node must not return revised_node")
        else:
            if self.state != "disproved" or self.revised_node is None:
                raise ValueError(
                    "revise_disproved_node requires state=disproved and revised_node"
                )
            if self.new_node is not None:
                raise ValueError("revision round must not add a node")
        return self


class RefinementAudit(BaseModel):
    round_index: int
    generation_attempt: int
    target_node_id: str
    input_state: NodeState
    patch: RefinementPatch | None = None
    graph_errors: list[str] = Field(default_factory=list)
    statement_error: str = ""
    accepted: bool = False
    raw_output: dict[str, object] | str | None = None


class RootFirstRunResult(BaseModel):
    success: bool
    stage: str
    problem_id: str
    problem_hash: str
    blueprint: RootFirstBlueprint
    refinements: list[RefinementAudit] = Field(default_factory=list)
    total_attempts: int = 0
    warnings: list[str] = Field(default_factory=list)
