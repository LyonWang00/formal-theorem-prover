from __future__ import annotations

import pytest
from pydantic import ValidationError

from lean_prover.Data import (
    DatasetSplit,
    EIData,
    EvaluationData,
    GRPOData,
    NormalizationFamily,
    RawInferenceData,
    SFTData,
    LeanFailureDetail,
    LeanVerificationStatus,
    classify_lean_diagnostics,
)
from lean_prover.Data.normalization import normalize_project_record
from lean_prover.Planner.schemas import RawTheoremInput


def test_raw_inference_is_exactly_one_routed_problem() -> None:
    row = RawInferenceData(
        input=RawTheoremInput(input_text="theorem target : True")
    )

    assert row.input.input_hash
    assert row.detected_input_kind is None


def test_sft_ei_grpo_and_evaluation_have_distinct_proof_contracts() -> None:
    sft = SFTData(
        record_id="sft-1",
        lean_statement="theorem target : True",
        verified_proof="by trivial",
        source="unit",
    )
    ei = EIData(
        split=DatasetSplit.DISCOVERY,
        record_id="ei-1",
        lean_statement="theorem target : True",
        proof="by trivial",
        pantograph_verified=True,
        source="unit",
        iteration=0,
    )
    grpo = GRPOData(
        record_id="grpo-1",
        lean_statement="theorem target : True",
        source="unit",
    )
    evaluation = EvaluationData(
        record_id="eval-1",
        lean_statement="theorem target : True",
        source="unit",
    )

    assert sft.verified_proof == "by trivial"
    assert ei.pantograph_verified
    assert not hasattr(grpo, "proof")
    assert evaluation.reference_proof is None
    with pytest.raises(ValidationError):
        GRPOData.model_validate(
            {
                **grpo.model_dump(mode="json"),
                "proof": "by trivial",
            }
        )


@pytest.mark.parametrize(
    ("family", "record", "expected_source"),
    [
        (
            NormalizationFamily.LEAN_WORKBOOK,
            {
                "id": "wb-1",
                "formal_statement": "theorem wb : True := by sorry",
                "tactic": "by trivial",
            },
            "lean-workbook",
        ),
        (
            NormalizationFamily.MINIF2F,
            {"id": "mini-1", "formal_statement": "theorem mini : True"},
            "minif2f",
        ),
        (
            NormalizationFamily.GENERIC,
            {"id": "generic-1", "statement": "theorem generic : True"},
            "generic-unit",
        ),
    ],
)
def test_existing_three_normalizers_are_preserved(
    family,
    record,
    expected_source,
) -> None:
    normalized = normalize_project_record(
        record,
        family=family,
        index=0,
        source_name="generic-unit" if family == NormalizationFamily.GENERIC else "unit",
    )

    assert normalized.lean_statement.startswith("theorem ")
    assert normalized.source == expected_source


@pytest.mark.parametrize(
    ("diagnostics", "expected_status", "expected_detail"),
    [
        (
            "unsolved goals\n⊢ False",
            LeanVerificationStatus.UNSOLVED_GOALS,
            LeanFailureDetail.UNSOLVED_GOALS,
        ),
        (
            "syntax error: unexpected token ')'",
            LeanVerificationStatus.SYNTAX_ERROR,
            LeanFailureDetail.UNEXPECTED_TOKEN,
        ),
        (
            "unknown identifier `foo`",
            LeanVerificationStatus.ELABORATION_ERROR,
            LeanFailureDetail.UNKNOWN_IDENTIFIER,
        ),
        (
            "application type mismatch",
            LeanVerificationStatus.ELABORATION_ERROR,
            LeanFailureDetail.APPLICATION_TYPE_MISMATCH,
        ),
        (
            "tactic `omega` failed",
            LeanVerificationStatus.TACTIC_ERROR,
            LeanFailureDetail.TACTIC_EXECUTION_FAILED,
        ),
        (
            "failed to import module Missing.Module",
            LeanVerificationStatus.ENVIRONMENT_ERROR,
            LeanFailureDetail.MISSING_IMPORT,
        ),
    ],
)
def test_shared_compiler_taxonomy_keeps_training_status_and_fine_detail(
    diagnostics,
    expected_status,
    expected_detail,
) -> None:
    result = classify_lean_diagnostics(diagnostics)

    assert result.status == expected_status
    assert result.detail == expected_detail
