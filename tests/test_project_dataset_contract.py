from __future__ import annotations

import json

import pytest

from lean_prover.Dataset.build_verified_datasets import (
    FAIL,
    SUCCESS,
    _proof_from_record,
    add_dataset_contract,
    dataset_record_hash,
    validate_contract_row,
)


def test_success_contract_uses_string_status_source_and_hashes() -> None:
    row = {
        "id": "wb-1",
        "lean_statement": "theorem wb_1 : True",
        "proof": "by trivial",
        "pantograph_verified": True,
    }
    normalized = add_dataset_contract(
        row,
        source="InternLM/Lean-Workbook",
        status=SUCCESS,
    )

    assert normalized["pantograph_verified"] == "success"
    assert normalized["source"] == "InternLM/Lean-Workbook"
    assert "sourcce" not in normalized
    assert len(normalized["record_hash"]) == 64
    assert normalized["record_hash"] == dataset_record_hash(
        normalized,
        source=normalized["source"],
    )
    validate_contract_row(normalized, expected_status=SUCCESS)


def test_fail_contract_always_has_error_message() -> None:
    normalized = add_dataset_contract(
        {
            "record_id": "wb-2",
            "statement": "theorem wb_2 : False",
            "proof": "by trivial",
        },
        source="InternLM/Lean-Workbook",
        status=FAIL,
    )

    assert normalized["pantograph_verified"] == "fail"
    assert normalized["error_message"]
    validate_contract_row(normalized, expected_status=FAIL)


def test_contract_rejects_boolean_pantograph_status() -> None:
    with pytest.raises(ValueError, match="success.*fail"):
        validate_contract_row(
            {
                "pantograph_verified": True,
                "source": "x",
                "record_hash": "0" * 64,
                "statement_sha256": "0" * 64,
                "proof_sha256": "0" * 64,
            }
        )


def test_disproved_trajectory_keeps_attempt_for_actual_compilation() -> None:
    raw_context = json.dumps(
        [
            {
                "tactic": "refine not_forall.2 ⟨0, by norm_num⟩",
                "state_before": "⊢ ¬ ∀ x : ℝ, x = 1",
                "state_after": "no goals",
            }
        ]
    )
    proof = _proof_from_record(
        {
            "proof": None,
            "raw_source_context": raw_context,
        }
    )

    assert proof.startswith("by\n")
    assert "not_forall" in proof
