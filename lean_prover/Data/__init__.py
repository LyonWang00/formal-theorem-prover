"""SFT, GRPO, evaluation, and Lean verification data contracts."""

from .compile_errors import (
    LeanErrorClassification,
    LeanFailureDetail,
    LeanVerificationStatus,
    classify_lean_diagnostics,
)
from .schemas import (
    DatasetSplit,
    DatasetStage,
    EvaluationData,
    GRPOData,
    GRPOGeneralData,
    NormalizationFamily,
    SFTData,
    SFTGeneralData,
)


def __getattr__(name: str):
    if name == "normalize_project_record":
        from .normalization import normalize_project_record

        return normalize_project_record
    raise AttributeError(name)


__all__ = [
    "DatasetSplit",
    "DatasetStage",
    "EvaluationData",
    "GRPOData",
    "GRPOGeneralData",
    "LeanErrorClassification",
    "LeanFailureDetail",
    "LeanVerificationStatus",
    "NormalizationFamily",
    "SFTData",
    "SFTGeneralData",
    "classify_lean_diagnostics",
]
