import json
from dataclasses import replace
from pathlib import Path

import pytest

from lean_prover.lean_training.data.preparation import (
    NormalizedExample,
    ProofFormat,
    compose_lean_theorem,
    contains_forbidden_proof_token,
    normalize_records,
    normalize_lean_workbook_example,
    should_validate_dataset_with_pantograph,
    split_lean_statement_and_proof,
    statement_hash,
)
from lean_prover.lean_training.data.training import (
    build_grpo_evaluation_record,
    build_grpo_general_data,
    build_grpo_training_record,
    build_sft_evaluation_record,
    build_sft_general_data,
    build_sft_training_record,
    deduplicate_training_records,
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


def test_sft_dedup_requires_both_statement_and_proof_to_match():
    rows = [
        {"id": "1", "formal_statement": "theorem a : True := by sorry", "tactic": "by trivial"},
        {"id": "1", "formal_statement": "theorem b : True := by sorry", "tactic": "by trivial"},
        {"id": "2", "formal_statement": "theorem a : True := by sorry", "tactic": "by trivial"},
        {"id": "3", "formal_statement": "theorem a : True := by sorry", "tactic": "by exact True.intro"},
    ]
    records = normalize_records(
        rows,
        dataset_kind="lean-workbook",
        source_name="test",
        require_proof=True,
    )
    assert len(records) == 3
    assert records[0].id == "1"
    assert records[1].id.startswith("1::")
    assert records[2].id == "3"
    assert records[0].lean_statement == records[2].lean_statement
    assert records[0].proof != records[2].proof


def test_cross_file_training_dedup_is_role_specific():
    sft_rows = [
        {"lean_statement": "theorem a : True", "proof": "by trivial", "id": "1"},
        {"lean_statement": "theorem a : True", "proof": "by exact True.intro", "id": "2"},
        {"lean_statement": "theorem a : True", "proof": "by trivial", "id": "3"},
    ]
    kept_sft, duplicate_sft = deduplicate_training_records(sft_rows, workflow="sft")
    assert [row["id"] for row in kept_sft] == ["1", "2"]
    assert [row["id"] for row in duplicate_sft] == ["3"]

    grpo_rows = [
        {"lean_statement": "theorem a : True", "id": "1"},
        {"lean_statement": "theorem a : True", "id": "2"},
    ]
    kept_grpo, duplicate_grpo = deduplicate_training_records(
        grpo_rows,
        workflow="grpo",
    )
    assert [row["id"] for row in kept_grpo] == ["1"]
    assert [row["id"] for row in duplicate_grpo] == ["2"]

    with pytest.raises(ValueError, match="proof/completion mismatch"):
        deduplicate_training_records(
            [
                {
                    "lean_statement": "theorem a : True",
                    "proof": "by trivial",
                    "completion": "by exact True.intro",
                }
            ],
            workflow="sft",
        )


def test_statement_hash_preserves_proposition_level_let_assignments():
    left = "theorem t : let x := 1; x = 1"
    right = "theorem t : let x := 2; x = 2"

    assert statement_hash(left) != statement_hash(right)


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
    example = replace(example, pantograph_verified=True)

    sft_train = build_sft_training_record(example)
    assert set(sft_train) == {"prompt", "completion"}
    assert "theorem formats : True" in sft_train["prompt"]
    assert sft_train["prompt"].endswith(":= by sorry")
    assert sft_train["completion"] == "by\n  trivial"
    with pytest.raises(ValueError, match="not Pantograph verified"):
        build_sft_training_record(replace(example, pantograph_verified=False))

    sft_eval = build_sft_evaluation_record(example)
    grpo_example = NormalizedExample(
        id="grpo-formats",
        source="unit-grpo",
        informal_statement="True is provable.",
        lean_statement="theorem formats_grpo : True",
        pantograph_verified=True,
    )
    grpo_train = build_grpo_training_record(grpo_example)
    grpo_eval = build_grpo_evaluation_record(grpo_example)
    assert sft_eval["lean_statement"] == "theorem formats : True"
    for record in (sft_eval, grpo_train, grpo_eval):
        assert "prompt" in record
        assert "proof" not in record
        assert "completion" not in record
        assert "text" not in record

    for record in (grpo_train, grpo_eval):
        assert record["lean_statement"] == "theorem formats_grpo : True"
        assert record["schema_version"] == "grpo_data"
        assert record["data_stage"] == "grpo"
        assert record["split"] == "train"
        assert record["verification_scope"] == "statement_only"
        assert not any(key.startswith("reference_proof") for key in record)
        assert "has_reference_proof" not in record

    # The GRPO projection never inspects or exports an in-memory proof value.
    proof_bearing_projection = build_grpo_training_record(example)
    assert proof_bearing_projection["verification_scope"] == "statement_only"
    assert "proof" not in proof_bearing_projection
    assert "completion" not in proof_bearing_projection
    assert not any(key.startswith("reference_proof") for key in proof_bearing_projection)


def test_manifest_projection_is_verified_and_model_independent():
    sft_example = NormalizedExample(
        id="sft-manifest",
        source="unit-sft",
        informal_statement="True is provable.",
        lean_statement="theorem manifest_sft : True",
        proof="by trivial",
        imports=("Mathlib",),
        pantograph_verified=True,
    )
    sft = build_sft_general_data(sft_example)
    assert sft["schema_version"] == "sft_manifest"
    assert sft["lean_statement"] == "theorem manifest_sft : True"
    assert sft["proof"] == "by trivial"
    assert "prompt" not in sft and "completion" not in sft and "text" not in sft

    grpo_example = NormalizedExample(
        id="grpo-manifest",
        source="unit-grpo",
        informal_statement="True is provable.",
        lean_statement="theorem manifest_grpo : True",
        imports=("Mathlib",),
        context_lines=("lemma support : True := by sorry",),
        pantograph_verified=True,
    )
    grpo = build_grpo_general_data(grpo_example)
    assert grpo["schema_version"] == "grpo_manifest"
    assert grpo["context_lines"] == ["lemma support : True := by sorry"]
    assert "proof" not in grpo and "completion" not in grpo

    with pytest.raises(ValueError, match="contains a reference proof"):
        build_grpo_general_data(sft_example)


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
    excluded_hash = statement_hash(records[0].lean_statement)
    kept, excluded = exclude_statement_overlaps(records, {excluded_hash})
    assert [record.id for record in kept] == ["second"]
    assert [record.id for record in excluded] == ["first"]
