from __future__ import annotations

import numpy as np

from lean_prover.lean_training.data.audit import proof_distribution, tactic_signature
from lean_prover.lean_training.data.contracts import (
    DataState,
    LeanDataRecord,
    make_attestation_id,
)
from lean_prover.lean_training.data.adapters.lean_workbook import (
    reconstruct_lean_workbook_records,
)
from lean_prover.lean_training.data.verified_builder import (
    _filter_sft_length,
    _select_balanced_train,
)
from scripts.verify_statement_datasets import (
    migrate_minif2f_context,
    statement_elaborated,
)
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
)
from lean_prover.lean_training.sft_pipeline.trainer import (
    completion_token_accuracy,
    completion_token_predictions,
)


def raw_row(**overrides):
    row = {
        "id": "workbook_1",
        "status": "proved",
        "tactic": "intro h",
        "state_before": "⊢ True → True",
        "state_after": "h : True\n⊢ True",
        "natural_language_statement": "True implies true.",
        "answer": "",
        "formal_statement": "theorem workbook_1 : True → True := by sorry",
    }
    row.update(overrides)
    return row


def test_workbook_tactic_rows_are_reconstructed_as_one_complete_proof(tmp_path):
    rows = [
        raw_row(),
        raw_row(
            tactic="exact h",
            state_before="h : True\n⊢ True",
            state_after="no goals",
        ),
    ]
    records, report = reconstruct_lean_workbook_records(
        rows,
        source_file=tmp_path / "raw.parquet",
        source_commit="snapshot",
    )
    assert len(records) == 1
    assert records[0].data_state is DataState.RAW
    assert records[0].proof == "by\n  intro h\n  exact h"
    assert records[0].metadata["trajectory_steps"] == 2
    assert report["transition_mismatches"] == 0


def test_disproved_workbook_trajectory_is_quarantined():
    records, _ = reconstruct_lean_workbook_records(
        [raw_row(status="disproved", state_after="no goals")],
        source_file=None,
        source_commit="snapshot",
    )
    assert records[0].data_state is DataState.QUARANTINED
    assert records[0].proof is None
    assert records[0].verification_error_type == "source_status_disproved"


def verified_record(record_id="r1", proof="by\n  simp"):
    attestation_id = make_attestation_id(
        record_id=record_id,
        environment_hash="env-v1",
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash=f"source-{record_id}",
    )
    return LeanDataRecord(
        record_id=record_id,
        statement_id=f"stmt_{record_id}",
        data_state=DataState.VERIFIED,
        source_dataset="fixture",
        imports=["Mathlib"],
        statement=f"theorem {record_id} : True",
        proof=proof,
        proof_format=ProofFormat.FULL_PROOF,
        statement_verified=True,
        proof_verified=True,
        pantograph_verified=True,
        verification_status="verified",
        lean_version="Lean fixture",
        mathlib_commit="mathlib fixture",
        environment_hash="env-v1",
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        attestation_id=attestation_id,
        attested_at="2026-01-01T00:00:00+00:00",
        assembled_source_hash=f"source-{record_id}",
    )


def test_attestation_is_bound_to_environment_and_assembler_versions():
    record = verified_record()
    assert record.can_enter_sft()
    assert record.attestation_is_current(
        environment_hash="env-v1",
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash="source-r1",
    )
    assert not record.attestation_is_current(
        environment_hash="env-v2",
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
    )
    assert not record.attestation_is_current(
        environment_hash="env-v1",
        assembler_version="new-assembler",
        normalization_version=NORMALIZATION_VERSION,
    )


def test_context_contract_preserves_source_backed_context_without_guessing():
    record = verified_record().model_copy(
        update={
            "namespace": "Workbook",
            "open_declarations": ["Real"],
            "open_scoped_declarations": ["BigOperators"],
            "section_context": "section LocalSection",
            "variable_context": "variable (α : Type)",
            "local_context": "variable [LinearOrder α]",
            "local_notations": ["local notation:65 x \" ≺ \" y => x < y"],
            "local_attributes": ["local attribute [simp] Nat.add_zero"],
        }
    )
    rendered = "\n".join(record.deterministic_context_lines())
    assert "namespace Workbook" in rendered
    assert "open Real" in rendered
    assert "open scoped BigOperators" in rendered
    assert "section LocalSection" in rendered
    assert "variable (α : Type)" in rendered
    assert "local notation" in rendered
    assert "local attribute" in rendered


def test_proof_distribution_and_tactic_signatures_are_deterministic():
    records = [
        verified_record("r1", "by\n  simp"),
        verified_record("r2", "by\n  simp"),
        verified_record("r3", "by\n  intro h\n  exact h"),
    ]
    report = proof_distribution(records)
    assert report["records"] == 3
    assert report["unique_normalized_proofs"] == 2
    assert report["exact_duplicate_records"] == 1
    assert tactic_signature(records[2].proof or "") == ("intro", "exact")


def test_balanced_train_selection_is_reproducible_and_bounded():
    single = [verified_record(f"s{i}", f"by\n  exact h{i}") for i in range(10)]
    multi = [
        verified_record(f"m{i}", f"by\n  intro h{i}\n  exact h{i}")
        for i in range(10)
    ]
    first, manifest = _select_balanced_train(
        single + multi,
        10,
        seed=42,
        max_identical_proof_occurrences=2,
        max_single_tactic_fraction=0.4,
    )
    second, _ = _select_balanced_train(
        single + multi,
        10,
        seed=42,
        max_identical_proof_occurrences=2,
        max_single_tactic_fraction=0.4,
    )
    assert [row.record_id for row in first] == [row.record_id for row in second]
    assert manifest["selected_single_tactic_records"] == 4
    assert manifest["selected_multi_step_records"] == 6


def test_overlength_verified_rows_stay_out_of_training_split(monkeypatch):
    class FakeTokenizer:
        eos_token = "<eos>"

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text)))}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    short = verified_record("short", "by simp")
    long = verified_record("long", "x" * 200)
    eligible, report = _filter_sft_length(
        [short, long],
        tokenizer_name_or_path="fixture-tokenizer",
        max_seq_length=150,
    )
    assert [record.record_id for record in eligible] == ["short"]
    assert report["excluded_overlength_records"] == 1
    assert report["excluded"][0]["record_id"] == "long"


def test_statement_only_migrations_are_explicit_and_warning_safe():
    statement, context, rules = migrate_minif2f_context(
        {
            "id": "amc12a_2020_p15",
            "lean_statement": (
                "theorem t (a b : ℂ) : Complex.abs (a - b) ≤ "
                "∑ k in Finset.range 1, ∏ j in Finset.range 1, 1"
            ),
            "context_lines": ["open BigOperators"],
        }
    )
    assert "‖a - b‖" in statement
    assert "∑ k ∈ Finset.range 1" in statement
    assert "∏ j ∈ Finset.range 1" in statement
    assert "open scoped BigOperators" in context
    assert len(rules) == 3
    assert statement_elaborated(
        {"success": False, "timed_out": False, "diagnostics": "warning: declaration uses `sorry`"}
    ) == (True, "sorry_warning_only")
    assert statement_elaborated(
        {"success": False, "timed_out": False, "diagnostics": "error: unknown identifier"}
    ) == (False, "elaboration_error")


def test_completion_token_accuracy_supports_flat_transformers_predictions():
    predictions = np.array([1, 2, 3, 9])
    labels = np.array([-100, 1, 2, 3])
    assert completion_token_accuracy((predictions, labels)) == {
        "token_accuracy": 1.0
    }


def test_completion_token_accuracy_shifts_batched_predictions():
    predictions = np.array([[1, 2, 9], [4, 9, 9]])
    labels = np.array([[-100, 1, 2], [-100, 4, 0]])
    assert completion_token_accuracy((predictions, labels)) == {
        "token_accuracy": 0.75
    }


def test_completion_token_predictions_preserves_precomputed_token_ids():
    import torch

    token_ids = torch.tensor([[1, 2, 3]])
    assert completion_token_predictions(token_ids, None).equal(token_ids)
    logits = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    assert completion_token_predictions(logits, None).tolist() == [[1, 0]]
