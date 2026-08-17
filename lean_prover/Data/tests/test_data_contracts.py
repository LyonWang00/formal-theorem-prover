from __future__ import annotations

import pytest
from pydantic import ValidationError

from lean_prover.Data import (
    DatasetSplit,
    EIData,
    EvaluationData,
    GRPOData,
    GRPOGeneralData,
    NormalizationFamily,
    RawInferenceData,
    SFTData,
    SFTGeneralData,
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
    sft = SFTGeneralData(
        record_id="sft-1",
        lean_statement="theorem target : True",
        proof="by trivial",
        source="unit",
        statement_hash="a" * 64,
        proof_hash="b" * 64,
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

    assert sft.proof == "by trivial"
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


def test_manifest_contracts_preserve_verification_context_without_role_leakage() -> None:
    sft = SFTGeneralData(
        record_id="sft-manifest-1",
        lean_statement="theorem target : True",
        proof="by trivial",
        imports=["Mathlib"],
        source="unit",
        statement_hash="a" * 64,
        proof_hash="b" * 64,
    )
    grpo = GRPOGeneralData(
        record_id="grpo-manifest-1",
        lean_statement="theorem target : True",
        context_lines=["lemma support : True := by sorry"],
        source="unit",
        statement_hash="a" * 64,
    )

    assert sft.verification_scope == "full_proof"
    assert grpo.verification_scope == "statement_only"
    with pytest.raises(ValidationError):
        GRPOGeneralData.model_validate(
            {**grpo.model_dump(mode="json"), "proof": "by trivial"}
        )


def test_sft_training_contract_is_exactly_prompt_and_completion() -> None:
    row = SFTData(prompt="prove this", completion="by trivial")

    assert row.model_dump() == {
        "prompt": "prove this",
        "completion": "by trivial",
    }
    with pytest.raises(ValidationError):
        SFTData.model_validate(
            {
                "prompt": "prove this",
                "completion": "by trivial",
                "lean_statement": "theorem t : True",
            }
        )


def test_grpo_adapter_allows_sorry_support_but_rejects_main_proof() -> None:
    accepted = normalize_project_record(
        {
            "record_id": "numina-grpo-support",
            "lean_statement": (
                "import Mathlib\n\n"
                "lemma support : True := by sorry\n\n"
                "theorem target : True"
            ),
            "pantograph_verified": "success",
            "verification_scope": "statement_only",
        },
        family=NormalizationFamily.NUMINAMATH_GRPO,
        index=0,
        source_name="unit",
    )
    assert accepted.lean_statement == "theorem target : True"
    assert accepted.context_lines == ("lemma support : True := by sorry",)
    assert accepted.proof == ""

    with pytest.raises(ValueError, match="main theorem must not contain a proof"):
        normalize_project_record(
            {
                "record_id": "numina-grpo-invalid",
                "lean_statement": "theorem target : True := by trivial",
                "pantograph_verified": "success",
                "verification_scope": "statement_only",
            },
            family=NormalizationFamily.NUMINAMATH_GRPO,
            index=0,
            source_name="unit",
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
            {
                "id": "mini-1",
                "split": "test",
                "formal_statement": "theorem mini : True",
                "header": "import Mathlib\n\nopen Nat",
                "pantograph_verified": "success",
            },
            "minif2f",
        ),
        (
            NormalizationFamily.NUMINAMATH_SFT,
            {
                "record_id": "numina-sft-1",
                "lean_statement": "import Mathlib\n\ntheorem numina_sft : True",
                "proof": "by trivial",
                "problem": "Prove True.",
                "pantograph_verified": "success",
                "verification_scope": "full_proof",
            },
            "numinamath-sft",
        ),
        (
            NormalizationFamily.NUMINAMATH_GRPO,
            {
                "record_id": "numina-grpo-1",
                "lean_statement": "import Mathlib\n\ntheorem numina_grpo : True",
                "metadata": {"original_problem": "Prove True."},
                "pantograph_verified": "success",
                "verification_scope": "statement_only",
            },
            "numinamath-grpo",
        ),
        (
            NormalizationFamily.KIMINA_GRPO,
            {
                "record_id": "kimina-grpo-1",
                "formal_statement": "import Mathlib\n\ntheorem kimina : True := by sorry",
                "lean_statement": "theorem kimina : True",
                "natural_language": "Prove True.",
                "pantograph_verified": "success",
                "verification_scope": "statement_only",
            },
            "kimina-grpo",
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
