from __future__ import annotations

import json

from lean_prover.lean_training.data.contracts import DataState
from lean_prover.lean_training.data.leandojo import (
    CorpusDeclaration,
    corpus_key,
    leandojo_prompt_statement,
    reconstruct_leandojo_records,
    render_tactic_trace,
    reservoir_sample,
    theorem_from_proof_state,
)


def test_theorem_from_proof_state_recovers_context() -> None:
    state = """case h
α : Type u_1
inst✝² : Semiring α
a b : α
h : a = b
⊢ a + 0 = b"""
    statement, universes = theorem_from_proof_state(
        state, record_id="leandojo_0123456789abcdef"
    )
    assert statement.startswith("theorem ld_0123456789abcdef")
    assert "(α : Type u_1)" in statement
    assert "[inst_ld_0 : Semiring α]" in statement
    assert "(a b : α)" in statement
    assert "(h : a = b)" in statement
    assert statement.endswith(": a + 0 = b")
    assert universes == ("u_1",)


def test_universe_recovery_does_not_treat_type_as_a_universe() -> None:
    state = """F : Type u → Type v
fᴴ : F α
α : Type u
⊢ fᴴ = fᴴ"""
    statement, universes = theorem_from_proof_state(
        state, record_id="leandojo_feedface"
    )
    assert universes == ("u", "v")
    assert "universe Type" not in statement
    assert "ld_local_0" in statement
    assert "fᴴ" not in statement


def test_reconstruct_record_preserves_source_provenance() -> None:
    row = {
        "url": "https://github.com/leanprover-community/mathlib4",
        "commit": "abc",
        "file_path": "Mathlib/Test.lean",
        "full_name": "Demo.add_zero",
        "start": [10, 1],
        "end": [12, 10],
        "traced_tactics": [
            {
                "tactic": "simpa using add_zero a",
                "annotated_tactic": "simpa using add_zero a",
                "state_before": "α : Type u\na : α\ninst : AddZeroClass α\n⊢ a + 0 = a",
                "state_after": "no goals",
            }
        ],
    }
    declaration = CorpusDeclaration(
        path="Mathlib/Test.lean",
        full_name="Demo.add_zero",
        code="theorem add_zero : a + 0 = a",
        imports=("Mathlib.Algebra.Group.Defs",),
    )
    records, report = reconstruct_leandojo_records(
        [row],
        corpus_declarations={corpus_key(row["file_path"], row["full_name"]): declaration},
        split_name="train",
    )
    record = records[0]
    assert record.data_state is DataState.RAW
    assert record.namespace == "Demo"
    assert record.proof == "by\n  simpa using add_zero a"
    assert record.metadata["source_imports"] == ["Mathlib.Algebra.Group.Defs"]
    assert "namespace Demo" in leandojo_prompt_statement(record)
    assert report["clean_records"] == 1


def test_reconstruct_record_applies_local_name_mapping_to_proof() -> None:
    row = {
        "commit": "abc",
        "file_path": "Mathlib/Test.lean",
        "full_name": "Demo.modifier",
        "traced_tactics": [
            {
                "tactic": "exact fᴴ",
                "state_before": "P : Prop\nfᴴ : P\n⊢ P",
                "state_after": "no goals",
            }
        ],
    }
    declaration = CorpusDeclaration(
        path="Mathlib/Test.lean",
        full_name="Demo.modifier",
        code="theorem modifier : P",
        imports=(),
    )
    records, _ = reconstruct_leandojo_records(
        [row],
        corpus_declarations={corpus_key(row["file_path"], row["full_name"]): declaration},
        split_name="train",
    )
    assert "fᴴ" not in records[0].statement
    assert records[0].proof == "by\n  exact ld_local_0"


def test_trace_corruption_is_quarantined() -> None:
    row = {
        "commit": "abc",
        "file_path": "Mathlib/Test.lean",
        "full_name": "Demo.bad",
        "traced_tactics": [
            {
                "tactic": "simp",
                "state_before": "⊢ True",
                "state_after": "⊢ False",
            },
            {
                "tactic": "trivial",
                "state_before": "⊢ True",
                "state_after": "no goals",
            },
        ],
    }
    declaration = CorpusDeclaration(
        path="Mathlib/Test.lean",
        full_name="Demo.bad",
        code="theorem bad : True",
        imports=(),
    )
    records, _ = reconstruct_leandojo_records(
        [row],
        corpus_declarations={corpus_key(row["file_path"], row["full_name"]): declaration},
        split_name="train",
    )
    assert records[0].data_state is DataState.QUARANTINED
    assert records[0].verification_error_type == "corrupt_tactic_trace"
    assert records[0].proof is None


def test_reservoir_sample_is_deterministic_and_without_replacement(tmp_path) -> None:
    rows = [{"id": value} for value in range(100)]
    first, total = reservoir_sample(rows, size=10, seed=20260801)
    second, _ = reservoir_sample(rows, size=10, seed=20260801)
    assert total == 100
    assert first == second
    assert len({json.dumps(row, sort_keys=True) for row in first}) == 10


def test_render_tactic_trace_preserves_multiline_tactics() -> None:
    proof = render_tactic_trace(
        [{"tactic": "constructor\n· trivial\n· assumption"}]
    )
    assert proof == "by\n  constructor\n  · trivial\n  · assumption"
