import json
from pathlib import Path

import pytest

from lean_prover.lean_training.data.preparation import (
    ProofFormat,
    compose_lean_theorem,
    contains_forbidden_proof_token,
    normalize_records,
    normalize_lean_workbook_example,
    should_validate_dataset_with_pantograph,
    split_lean_statement_and_proof,
)
from lean_prover.lean_training.data.training import (
    build_grpo_evaluation_record,
    build_grpo_training_record,
    build_sft_evaluation_record,
    build_sft_training_record,
    exclude_statement_overlaps,
    proof_length_metrics,
)


@pytest.mark.parametrize(
    ("code", "expected_statement", "expected_rhs_prefix"),
    [
        ("theorem t1 : True := by\n  trivial", "theorem t1 : True", "by"),
        (
            "theorem t2 : let x := 1; x = 1 := by\n  rfl",
            "theorem t2 : let x := 1; x = 1",
            "by",
        ),
        ("theorem t3 : True := by\n  let x := 1\n  trivial", "theorem t3 : True", "by"),
        ("theorem t4 : Prop -> Prop := fun x => x", "theorem t4 : Prop -> Prop", "fun"),
        (
            'theorem t5 : ("a := b" = "a := b") := by\n  rfl',
            'theorem t5 : ("a := b" = "a := b")',
            "by",
        ),
        ("theorem t6 : True := by\n  -- comment :=\n  trivial", "theorem t6 : True", "by"),
        (
            "theorem t7 : True := by\n  /- outer := /- nested := -/ -/\n  trivial",
            "theorem t7 : True",
            "by",
        ),
        (
            "theorem t8 (x : Nat := 1) :\n  True := by\n  trivial",
            "theorem t8 (x : Nat := 1) :\n  True",
            "by",
        ),
    ],
)
def test_split_lean_statement_and_proof(code, expected_statement, expected_rhs_prefix):
    statement, rhs = split_lean_statement_and_proof(code)
    assert statement == expected_statement
    assert rhs.startswith(expected_rhs_prefix)


def test_proof_rhs_preserves_by_and_term_proofs():
    record = normalize_lean_workbook_example(
        {
            "id": "a",
            "formal_statement": "theorem a : True := by sorry",
            "tactic": "by\n  trivial",
        },
        0,
    )
    assert record.proof == "by\n  trivial"
    assert compose_lean_theorem("theorem id : Prop -> Prop", "fun x => x") == (
        "theorem id : Prop -> Prop := fun x => x"
    )


def test_tactic_body_is_wrapped_in_exactly_one_by():
    assert compose_lean_theorem("theorem t : True", "trivial") == (
        "theorem t : True := by\n  trivial"
    )
    assert compose_lean_theorem("theorem t : True", "by\n  trivial") == (
        "theorem t : True := by\n  trivial"
    )


def test_explicit_proof_format_controls_assembly():
    assert compose_lean_theorem(
        "theorem t : True",
        "exact True.intro",
        proof_format=ProofFormat.TACTIC_BODY,
    ) == "theorem t : True := by\n  exact True.intro"


def test_existing_statement_assignment_is_not_duplicated():
    assert compose_lean_theorem(
        "theorem t : True := by\n  trivial",
        "trivial",
    ) == "theorem t : True := by\n  trivial"


@pytest.mark.parametrize("dataset_kind", ["generic", "lean-workbook", "minif2f"])
def test_requested_pantograph_validation_applies_to_every_source(dataset_kind):
    assert should_validate_dataset_with_pantograph(dataset_kind)


def test_statement_proof_mismatch_rejected():
    with pytest.raises(ValueError):
        compose_lean_theorem("theorem target : False", "theorem easy : True := by trivial")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("by sorry", True),
        ("by admit", True),
        ("sorryAx foo", True),
        ("-- sorry in a comment", False),
        ('"sorry in a string"', False),
    ],
)
def test_forbidden_token_scanner(source, expected):
    assert contains_forbidden_proof_token(source) is expected


def test_duplicate_id_and_statement_hash_are_rejected():
    rows = [
        {"id": "1", "formal_statement": "theorem a : True := by sorry", "tactic": "by trivial"},
        {"id": "1", "formal_statement": "theorem b : True := by sorry", "tactic": "by trivial"},
        {"id": "2", "formal_statement": "theorem a : True := by sorry", "tactic": "by trivial"},
    ]
    records = normalize_records(
        rows,
        dataset_kind="lean-workbook",
        source_name="test",
        require_proof=True,
    )
    assert [record.id for record in records] == ["1"]


def test_tuple_preamble_is_preserved_and_unknown_not_imported():
    rows = [
        {
            "id": "p",
            "formal_statement": "theorem p : True := by sorry",
            "tactic": "by trivial",
            "imports": ("Mathlib", "open Nat", "mystery command"),
        }
    ]
    record = normalize_records(
        rows,
        dataset_kind="lean-workbook",
        source_name="test",
        require_proof=True,
    )[0]
    assert record.imports == ("Mathlib",)
    assert "open Nat" in record.context_lines
    assert "mystery command" in record.unknown_preamble_lines


def test_workflow_specific_records_keep_targets_out_of_evaluation_and_grpo():
    example = normalize_lean_workbook_example(
        {
            "id": "formats",
            "formal_statement": "theorem formats : True := by sorry",
            "tactic": "by\n  trivial",
        },
        0,
    )

    sft_train = build_sft_training_record(example)
    assert sft_train["lean_statement"] == "theorem formats : True"
    assert sft_train["proof"] == "by\n  trivial"
    assert sft_train["completion"] == sft_train["proof"]

    sft_eval = build_sft_evaluation_record(example)
    grpo_train = build_grpo_training_record(example)
    grpo_eval = build_grpo_evaluation_record(example)
    for record in (sft_eval, grpo_train, grpo_eval):
        assert record["lean_statement"] == "theorem formats : True"
        assert "prompt" in record
        assert "proof" not in record
        assert "completion" not in record
        assert "text" not in record

    expected_lengths = proof_length_metrics(example.proof)
    assert grpo_train["reference_proof_length_tokens"] == expected_lengths["tokens"]
    assert "reference_proof_hash" in grpo_train


def test_grpo_overlap_filter_uses_statement_hash_not_record_id():
    records = normalize_records(
        [
            {
                "id": "first",
                "formal_statement": "theorem first : True := by sorry",
                "tactic": "by trivial",
            },
            {
                "id": "second",
                "formal_statement": "theorem second : True := by sorry",
                "tactic": "by trivial",
            },
        ],
        dataset_kind="lean-workbook",
        source_name="test",
        require_proof=True,
    )
    excluded_hash = build_sft_training_record(records[0])["statement_hash"]
    kept, excluded = exclude_statement_overlaps(records, {excluded_hash})
    assert [record.id for record in kept] == ["second"]
    assert [record.id for record in excluded] == ["first"]
