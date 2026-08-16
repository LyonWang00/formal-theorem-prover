"""Unified project data contracts."""

from .compile_errors import (
    LeanErrorClassification,
    LeanFailureDetail,
    LeanVerificationStatus,
    classify_lean_diagnostics,
)
from .schemas import (
    BlueprintData,
    DatasetSplit,
    DatasetStage,
    EIData,
    EvaluationData,
    GRPOData,
    NormalizationFamily,
    PlannerDataStatus,
    ProverDataStatus,
    ProverNodeResult,
    ProverProblemResult,
    ProverRootResult,
    RawInferenceData,
    SFTData,
)


def __getattr__(name: str):
    if name == "normalize_project_record":
        from .normalization import normalize_project_record

        return normalize_project_record
    raise AttributeError(name)

__all__ = [
    "BlueprintData",
    "DatasetSplit",
    "DatasetStage",
    "EIData",
    "EvaluationData",
    "GRPOData",
    "LeanErrorClassification",
    "LeanFailureDetail",
    "LeanVerificationStatus",
    "NormalizationFamily",
    "PlannerDataStatus",
    "ProverDataStatus",
    "ProverNodeResult",
    "ProverProblemResult",
    "ProverRootResult",
    "RawInferenceData",
    "SFTData",
    "classify_lean_diagnostics",
]
