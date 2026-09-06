"""Strict data contracts used by the SFT, GRPO, and evaluation pipelines."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictDataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DatasetStage(StrEnum):
    SFT = "sft"
    GRPO = "grpo"
    EVALUATION = "evaluation"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    DISCOVERY = "discovery"
    MONITOR = "monitor"
    BENCHMARK = "benchmark"


class NormalizationFamily(StrEnum):
    LEAN_WORKBOOK = "lean_workbook"
    LEANDOJO_SFT = "leandojo_sft"
    NUMINAMATH_SFT = "numinamath_sft"
    NUMINAMATH_GRPO = "numinamath_grpo"
    KIMINA_GRPO = "kimina_grpo"
    MINIF2F = "minif2f"
    GENERIC = "generic"


class SFTGeneralData(StrictDataModel):
    """Verified SFT manifest row before trainer projection."""

    schema_version: Literal["sft_manifest"] = "sft_manifest"
    data_stage: Literal[DatasetStage.SFT] = DatasetStage.SFT
    split: DatasetSplit = DatasetSplit.TRAIN
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    proof: str = Field(min_length=1, pattern=r"^by(?:\s|$)")
    imports: list[str] = Field(default_factory=list)
    context_lines: list[str] = Field(default_factory=list)
    unknown_preamble_lines: list[str] = Field(default_factory=list)
    source: str = Field(min_length=1)
    statement_hash: str = Field(min_length=64, max_length=64)
    proof_hash: str = Field(min_length=64, max_length=64)
    pantograph_verified: Literal[True] = True
    verification_scope: Literal["full_proof"] = "full_proof"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SFTData(StrictDataModel):
    """Strict trainer-facing SFT row."""

    prompt: str = Field(min_length=1)
    completion: str = Field(min_length=1, pattern=r"^by(?:\s|$)")


class GRPOData(StrictDataModel):
    """Trainer-facing GRPO row: a statement without a proof target."""

    schema_version: Literal["grpo_data_v1"] = "grpo_data_v1"
    data_stage: Literal[DatasetStage.GRPO] = DatasetStage.GRPO
    split: DatasetSplit = DatasetSplit.TRAIN
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    source: str = Field(min_length=1)
    pantograph_verified: Literal["success"] = "success"
    verification_scope: Literal["statement_only"] = "statement_only"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GRPOGeneralData(StrictDataModel):
    """Model-independent statement-only GRPO manifest row."""

    schema_version: Literal["grpo_manifest"] = "grpo_manifest"
    data_stage: Literal[DatasetStage.GRPO] = DatasetStage.GRPO
    split: DatasetSplit = DatasetSplit.TRAIN
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    imports: list[str] = Field(default_factory=list)
    context_lines: list[str] = Field(default_factory=list)
    unknown_preamble_lines: list[str] = Field(default_factory=list)
    source: str = Field(min_length=1)
    statement_hash: str = Field(min_length=64, max_length=64)
    pantograph_verified: Literal[True] = True
    verification_scope: Literal["statement_only"] = "statement_only"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationData(StrictDataModel):
    """One evaluation problem and its optional reference proof."""

    schema_version: Literal["evaluation_data_v1"] = "evaluation_data_v1"
    data_stage: Literal[DatasetStage.EVALUATION] = DatasetStage.EVALUATION
    split: DatasetSplit = DatasetSplit.EVAL
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    reference_proof: str | None = None
    source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
