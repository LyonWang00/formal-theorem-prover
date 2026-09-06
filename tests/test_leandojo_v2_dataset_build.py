from __future__ import annotations

import copy

from lean_prover.lean_training.data.leandojo_v2_dataset.dedup import (
    audit_cross_source,
    audit_internal_dedup,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.fingerprints import (
    normalized_hash,
    normalize_lexical,
    statement_hash,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.leakage import (
    audit_leakage,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.manifest_builder import (
    sample_rows,
    sample_token_matched,
    summarize_manifest,
)
from lean_prover.lean_training.data.leandojo_v2_dataset.normalization import (
    normalize_leandojo_record,
    normalize_workbook_record,
)


def candidate(
    row_id: str,
    *,
    statement: str = "theorem T (n : Nat) : n = n :=",
    proof: str = "by\n  rfl",
) -> dict:
    declaration = f"{statement} {proof}"
    return {
        "id": row_id,
        "source": "leandojo_v2_current_mathlib",
        "repository_commit": "commit",
        "source_file": "Mathlib/Test.lean",
        "qualified_name": f"Test.{row_id}",
        "source_span": {"start_line": 1, "end_line": 2},
        "statement": statement,
        "proof": proof,
        "declaration_source": declaration,
        "assembled_source_hash": "assembled",
        "premises": [{"qualified_name": "Nat.add_zero"}, "Nat.zero_add"],
        "tactic_trace": [
            {
                "step_index": 7,
                "state_before": "n : Nat\r\n⊢ n = n ",
                "tactic": "rfl ",
                "state_after": "no goals",
            }
        ],
        "proof_style": "tactic",
        "metadata": {"assembled_source": declaration},
    }


def verified() -> dict:
    return {
        "compile_success": True,
        "timed_out": False,
        "error_category": None,
        "stderr": "",
    }


def test_normalization_preserves_raw_and_separates_executable_layer() -> None:
    raw = candidate("a")
    before = copy.deepcopy(raw)
    row = normalize_leandojo_record(raw, verified())
    assert raw == before
    assert row["raw_statement"] == before["statement"]
    assert row["raw_proof"] == before["proof"]
    assert row["metadata"]["assembled_source"] == before["metadata"]["assembled_source"]
    assert row["training_statement"] == "theorem T (n : Nat) : n = n"
    assert row["training_declaration"] == "theorem T (n : Nat) : n = n := by\n  rfl"
    assert ":= :=" not in row["training_declaration"]
    assert "by by" not in row["training_declaration"]


def test_unicode_line_endings_hash_and_trace_order_are_stable() -> None:
    assert normalize_lexical("e\u0301 \r\nx\t ") == normalize_lexical("é\nx")
    assert normalized_hash("e\u0301 \r\nx\t ") == normalized_hash("é\nx")
    assert statement_hash("theorem GeneratedA : True :=") == statement_hash(
        "theorem OriginalName : True"
    )
    first = normalize_leandojo_record(candidate("a"), verified())
    second = normalize_leandojo_record(candidate("a"), verified())
    assert first["statement_hash_normalized"] == second["statement_hash_normalized"]
    assert first["proof_hash_normalized"] == second["proof_hash_normalized"]
    assert first["raw_tactic_trace"][0]["step_index"] == 7
    assert first["normalized_tactic_trace"][0]["normalized_step_index"] == 0
    assert first["tactic_trace_validation"] == {
        "step_indices_continuous": False,
        "adjacent_states_aligned": True,
        "final_goal_closed": True,
        "step_count": 1,
    }
    assert first["raw_premises"] == [
        {"qualified_name": "Nat.add_zero"},
        "Nat.zero_add",
    ]
    assert first["premise_set_canonical"] == ["Nat.add_zero", "Nat.zero_add"]


def test_term_tactic_and_mixed_proof_formats() -> None:
    tactic = normalize_leandojo_record(candidate("t", proof="by\n  rfl"), verified())
    term = normalize_leandojo_record(candidate("u", proof="Eq.refl n"), verified())
    mixed_raw = candidate("m", proof="by\n  exact Eq.refl n")
    mixed_raw["proof_style"] = "mixed"
    mixed = normalize_leandojo_record(mixed_raw, verified())
    assert tactic["proof_style"] == "tactic_by"
    assert term["proof_style"] == "term"
    assert mixed["proof_style"] == "mixed"


def test_dedup_preserves_same_statement_different_proof() -> None:
    exact_a = normalize_leandojo_record(candidate("a"), verified())
    exact_b = normalize_leandojo_record(candidate("b"), verified())
    variant = normalize_leandojo_record(
        candidate("c", proof="by\n  exact Eq.refl _"), verified()
    )
    different_statement_same_proof = normalize_leandojo_record(
        candidate("d", statement="theorem U (n : Nat) : n + 0 = n :="), verified()
    )
    result = audit_internal_dedup(
        [exact_a, exact_b, variant, different_statement_same_proof]
    )
    assert len(result["canonical_records"]) == 3
    assert len(result["same_statement_multiple_proofs"]) == 1
    assert result["same_statement_multiple_proofs"][0]["proof_variant_count"] == 2
    assert any(row["id"] == "d" for row in result["canonical_records"])


def test_cross_source_duplicate_and_proof_variant_are_distinct() -> None:
    ld = normalize_leandojo_record(candidate("ld"), verified())
    wb_exact = normalize_workbook_record(
        {"id": "wb1", "statement": ld["raw_statement"], "proof": ld["raw_proof"]}
    )
    wb_variant = normalize_workbook_record(
        {
            "id": "wb2",
            "statement": ld["raw_statement"],
            "proof": "by\n  exact Eq.refl _",
        }
    )
    result = audit_cross_source([ld], [wb_exact, wb_variant])
    assert result["overlaps"][0]["exact_statement_exact_proof_alias_ids"] == ["wb1"]
    assert result["overlaps"][0]["same_statement_different_proof_ids"] == ["wb2"]


def test_leakage_levels_zero_to_two_remove_entire_theorem_group_only() -> None:
    first = normalize_leandojo_record(candidate("a"), verified())
    variant = normalize_leandojo_record(
        candidate("b", proof="by\n  exact Eq.refl _"), verified()
    )
    safe = normalize_leandojo_record(
        candidate("safe", statement="theorem Safe : True :=", proof="by trivial"),
        verified(),
    )
    protected = [
        normalize_workbook_record(
            {
                "id": "eval",
                "statement": first["raw_statement"].replace("\n", "\r\n"),
                "proof": "by\n  omega",
            }
        )
    ]
    before = copy.deepcopy(protected)
    result = audit_leakage([first, variant, safe], protected)
    assert {row["id"] for row in result["removed"]} == {"a", "b"}
    assert [row["id"] for row in result["retained"]] == ["safe"]
    assert protected == before


def test_near_duplicate_is_review_only() -> None:
    train = normalize_leandojo_record(candidate("a"), verified())
    protected = [
        normalize_workbook_record(
            {
                "id": "eval",
                "statement": "lemma Other (n : Nat) : n = n :=",
                "proof": "by rfl",
            }
        )
    ]
    result = audit_leakage([train], protected)
    assert result["removed"] == []
    assert [row["id"] for row in result["retained"]] == ["a"]
    assert len(result["structural_near_duplicates"]) == 1


def test_row_manifest_is_deterministic_without_replacement_or_group_overlap() -> None:
    wb = [
        normalize_workbook_record(
            {
                "id": f"wb{i}",
                "statement": f"theorem W{i} : {i} = {i}",
                "proof": "by rfl",
            }
        )
        for i in range(4)
    ]
    ld = [
        normalize_leandojo_record(
                candidate(
                    f"ld{i}",
                    statement=f"theorem L{i} : {i + 10} = {i + 10} :=",
                    proof="by rfl",
                ),
            verified(),
        )
        for i in range(4)
    ]
    first = sample_rows(wb, ld, workbook_rows=2, leandojo_rows=2, seed=42)
    second = sample_rows(wb, ld, workbook_rows=2, leandojo_rows=2, seed=42)
    assert [row.get("id") for row in first] == [row.get("id") for row in second]
    summary = summarize_manifest("test", first, seed=42)
    assert summary["rows"] == 4
    assert summary["theorem_group_duplicates"] == 0
    assert summary["max_repeat"] == 1


def test_token_manifest_matches_budget_without_replacement() -> None:
    wb = [
        {
            **normalize_workbook_record(
                {
                    "id": f"wb{i}",
                    "statement": f"theorem W{i} : {i} = {i}",
                    "proof": "by rfl",
                }
            ),
            "statement_tokens": 5,
            "label_tokens": 10,
            "total_tokens": 15,
        }
        for i in range(20)
    ]
    ld = [
        {
            **normalize_leandojo_record(
                candidate(
                    f"ld{i}",
                    statement=f"theorem L{i} : {i + 20} = {i + 20} :=",
                    proof="by rfl",
                ),
                verified(),
            ),
            "statement_tokens": 5,
            "label_tokens": 10,
            "total_tokens": 15,
        }
        for i in range(20)
    ]
    rows = sample_token_matched(
        wb,
        ld,
        workbook_share=0.5,
        label_token_budget=200,
        seed=42,
    )
    summary = summarize_manifest("token", rows, seed=42, budget=200)
    assert summary["label_token_budget_error_ratio"] <= 0.02
    assert summary["theorem_group_duplicates"] == 0
    assert summary["max_repeat"] == 1
