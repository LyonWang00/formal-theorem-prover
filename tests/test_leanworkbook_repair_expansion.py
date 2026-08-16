from __future__ import annotations

from pathlib import Path

from lean_prover.Dataset.expand_leanworkbook import (
    applicable_high_quality_methods,
    expanded_row,
    reordered_proof,
    split_statement,
    split_top_level_relation,
    transform,
)
from lean_prover.Dataset.repair_leanworkbook_failures import (
    _classification,
    _manual_result_requires_retry,
    _manual_source,
    _statement_from_initial_state,
    _targeted_proof_variants,
    _proof_portfolio,
    eligible_rows,
    manual_compatibility_repair,
    normalize_missing_goal_colon,
    restore_disproved_statement,
)


def test_manual_source_uses_preloaded_pantograph_imports() -> None:
    source = _manual_source(
        {
            "candidate_id": "demo:full_repair",
            "repaired_statement": "theorem demo : True",
            "repaired_proof": "by trivial",
        }
    )
    assert source == "theorem demo : True := by trivial\n"
    assert not source.startswith("import ")


def test_manual_runtime_failures_are_not_reused_as_proof_failures() -> None:
    assert _manual_result_requires_retry(
        {
            "error_type": "pantograph_error",
            "timed_out": True,
            "diagnostics": "Server reached timeout limit",
        }
    )
    assert _manual_result_requires_retry(
        {
            "error_type": "pantograph_error",
            "timed_out": False,
            "diagnostics": "Server not running.",
        }
    )
    assert not _manual_result_requires_retry(
        {
            "error_type": "lean_compilation",
            "timed_out": False,
            "diagnostics": "error: unsolved goals",
        }
    )


def test_split_statement_ignores_binder_type_colons() -> None:
    name, header, proposition = split_statement(
        "theorem demo (n : Nat) (h : n > 0) : n = n"
    )
    assert name == "demo"
    assert header.endswith("(h : n > 0)")
    assert proposition == "n = n"


def test_identity_transforms_are_complete_proofs() -> None:
    statement = "theorem demo (n : Nat) : n = n"
    proof = "by\n  rfl"
    true_statement, true_proof = transform(statement, proof, "true_and", "demo_true")
    or_statement, or_proof = transform(statement, proof, "or_false", "demo_or")
    assert true_statement == "theorem demo_true (n : Nat) : True ∧ (n = n)"
    assert "constructor" in true_proof and "exact (by" in true_proof
    assert or_statement == "theorem demo_or (n : Nat) : (n = n) ∨ False"
    assert "Or.inl" in or_proof


def test_top_level_relation_ignores_relations_inside_binders() -> None:
    proposition = "(h : x < y) -> x + 1 <= y + 1"
    # An implication is not a direct numeric-relation target and must not be
    # analogized by rewriting a nested hypothesis or conclusion in isolation.
    assert split_top_level_relation(proposition) is None


def test_numeric_equality_high_quality_transforms_are_complete() -> None:
    statement = "theorem demo (x y : Real) (h : x = y) : x + 2 = y + 2"
    proof = "by\n  rw [h]"
    assert applicable_high_quality_methods(statement) == [
        "equality_symmetry",
        "relation_translate_one",
        "relation_scale_two",
        "equality_sub_zero",
    ]
    translated_statement, translated_proof = transform(
        statement, proof, "relation_translate_one", "demo_translate"
    )
    assert translated_statement.endswith(": 1 + (x + 2) = 1 + (y + 2)")
    assert "congrArg (fun z => 1 + z) h_parent" in translated_proof
    sub_statement, sub_proof = transform(
        statement, proof, "equality_sub_zero", "demo_sub"
    )
    assert sub_statement.endswith(": (x + 2) - (y + 2) = 0")
    assert "sub_eq_zero.mpr h_parent" in sub_proof


def test_nat_equality_does_not_offer_subtraction_equivalence() -> None:
    methods = applicable_high_quality_methods(
        "theorem demo (n m : Nat) (h : n = m) : n + 1 = m + 1"
    )
    assert "equality_sub_zero" not in methods
    assert "relation_scale_two" in methods


def test_numeric_inequality_scaling_uses_positive_multiplier_proof() -> None:
    statement, proof = transform(
        "theorem demo (x y : Real) (h : x < y) : x < y",
        "by\n  exact h",
        "relation_scale_two",
        "demo_scale",
    )
    assert statement.endswith(": 2 * (x) < 2 * (y)")
    assert "mul_lt_mul_of_pos_left h_parent" in proof


def test_reorder_changes_only_one_tactic_argument_list() -> None:
    assert reordered_proof("by\n  nlinarith [h1, h2, h3]") == (
        "by\n  nlinarith [h3, h2, h1]"
    )


def test_raw_expansion_preserves_parent_top_level_field_set() -> None:
    parent = {
        "record_id": "demo",
        "statement_id": "stmt_demo",
        "data_state": "verified",
        "source_file": "source.jsonl",
        "source_declaration": "demo",
        "source_span": {"start_row": 1, "end_row": 1},
        "informal_statement": "Prove reflexivity.",
        "statement": "theorem demo (n : Nat) : n = n",
        "proof": "by\n  rfl",
        "raw_declaration": "theorem demo (n : Nat) : n = n := by sorry",
        "raw_source_context": "[]",
        "statement_verified": True,
        "proof_verified": True,
        "reference_proof_verified": True,
        "pantograph_verified": "success",
        "verification_status": "success",
        "verification_error_type": None,
        "verification_error_message": None,
        "assembled_source_hash": "a" * 64,
        "recovered_source_hash": "b" * 64,
        "statement_sha256": "c" * 64,
        "proof_sha256": "d" * 64,
        "record_hash": "e" * 64,
        "metadata": {},
    }
    result = expanded_row(
        parent,
        method="extra_true_assumption",
        ordinal=0,
        raw_path=Path("leanworkbook_expand_raw.jsonl"),
    )
    assert set(result) == set(parent)
    assert result["pantograph_verified"] == "pending"
    assert "h_extra" in result["statement"]


def test_repair_selection_prioritizes_actionable_unknown_names() -> None:
    rows = [
        {
            "record_id": "unsolved",
            "statement": "theorem u : True",
            "proof": "by aesop",
            "error_message": "unsolved goals",
            "metadata": {"raw_status": "proved", "trajectory_steps": 1},
        },
        {
            "record_id": "unknown",
            "statement": "theorem k : True",
            "proof": "by exact old_name",
            "error_message": "Unknown identifier `old_name`\nunsolved goals",
            "metadata": {"raw_status": "proved", "trajectory_steps": 2},
        },
    ]
    selected, excluded = eligible_rows(rows, count=1, seed=7)
    assert not excluded
    assert selected[0]["record_id"] == "unknown"
    assert _classification(selected[0])[1] == "unknown_identifier"


def test_manual_repair_qualifies_real_and_current_mathlib_names() -> None:
    statement, proof, changes = manual_compatibility_repair(
        "theorem demo (x : ℝ) (hx : 0 < x) : sin x / x ≤ cos x",
        "by\n  rw [div_le_iff hx, sin_add]\n  exact add_one_le_exp x",
    )
    assert "Real.sin x" in statement
    assert "Real.cos x" in statement
    assert "div_le_iff₀" in proof
    assert "Real.sin_add" in proof
    assert "Real.add_one_le_exp" in proof
    assert changes


def test_manual_repair_does_not_double_qualify_names() -> None:
    statement, proof, changes = manual_compatibility_repair(
        "theorem demo (x : ℝ) : Real.sin x = Real.sin x",
        "by\n  rw [Real.sin_add]",
    )
    assert statement.count("Real.sin") == 2
    assert proof.count("Real.sin_add") == 1
    assert not changes


def test_manual_repair_recovers_legacy_implicit_real_variables() -> None:
    statement, _, changes = manual_compatibility_repair(
        "theorem demo : sin x ^ 2 + cos x ^ 2 = 1",
        "by\n  exact sin_sq_add_cos_sq x",
    )
    assert "Real.sin x" in statement
    assert "Real.cos x" in statement
    assert any(change["to"] == "Real.sin_sq_add_cos_sq" for change in changes)


def test_manual_repair_replaces_legacy_complex_abs() -> None:
    statement, _, changes = manual_compatibility_repair(
        "theorem demo (z : ℂ) : Complex.abs z = Complex.abs z",
        "by\n  rfl",
    )
    assert statement == "theorem demo (z : ℂ) : norm z = norm z"
    assert any(change["from"] == "Complex.abs" for change in changes)


def test_manual_repair_uses_locally_probed_current_mathlib_names() -> None:
    _, proof, changes = manual_compatibility_repair(
        "theorem demo (x y : Real) : |x + y| <= |x| + |y|",
        "by\n  simpa [Int.mod_self, Function.funext_iff] using abs_add x y",
    )
    assert "Int.emod_self" in proof
    assert "funext_iff" in proof
    assert "abs_add_le" in proof
    assert any(change["reason"] == "current_mathlib_name" for change in changes)


def test_manual_repair_can_select_reviewed_clean_proof() -> None:
    _, proof, changes = manual_compatibility_repair(
        "theorem lean_workbook_plus_1072 : Real.cos 10 = Real.cos 10",
        "by\n  simp [cos_10]",
        record_id="lean_workbook_plus_1072",
    )
    assert proof == "by\n  rfl"
    assert any(change["to"] == "individually_reviewed_clean_proof" for change in changes)


def test_manual_repair_updates_legacy_big_operator_and_factorial_syntax() -> None:
    statement, proof, changes = manual_compatibility_repair(
        "theorem demo (n : Nat) : \u2211 i in Finset.range n, (i + 1)! > 0",
        "by\n  simp [\u2211 i in Finset.range n, i]",
    )
    assert "\u2211 i \u2208 Finset.range n" in statement
    assert "Nat.factorial (i + 1)" in statement
    assert "\u2211 i \u2208 Finset.range n" in proof
    assert any(change["reason"] == "current_mathlib_syntax" for change in changes)


def test_manual_repair_does_not_rewrite_tactic_exclamation() -> None:
    _, proof, _ = manual_compatibility_repair(
        "theorem demo (p : Prop) : p -> p",
        "by\n  intro h\n  simpa! using h",
    )
    assert "simpa!" in proof


def test_manual_repair_does_not_rewrite_contrapose_exclamation() -> None:
    _, proof, _ = manual_compatibility_repair(
        "theorem demo (p : Prop) : p -> p",
        "by\n  contrapose! h",
    )
    assert "contrapose!" in proof


def test_statement_can_be_rebuilt_from_frozen_initial_state() -> None:
    rebuilt = _statement_from_initial_state(
        "demo",
        "a b : Real\nh : a = b\n\u22a2 b = a",
    )
    assert rebuilt == "theorem demo (a b : Real) (h : a = b) : b = a"


def test_targeted_variants_remove_redundant_rewrite_after_field_simp() -> None:
    variants = _targeted_proof_variants(
        "by\n"
        "  field_simp [ha.ne']\n"
        "  rw [div_le_iff₀ (by positivity)]\n"
        "  nlinarith [sq_nonneg (a - b)]"
    )
    proofs = [row["proof"] for row in variants]
    assert any("field_simp" in proof and "nlinarith" in proof and "rw [" not in proof for proof in proofs)


def test_targeted_variants_preserve_post_rewrite_tactic_chain() -> None:
    variants = _targeted_proof_variants(
        "by\n  constructor <;> field_simp <;> rw [div_le_div_iff₀] <;> nlinarith"
    )
    assert any(
        "constructor <;> field_simp <;> nlinarith" in row["proof"]
        for row in variants
    )


def test_targeted_variants_can_drop_redundant_trailing_tactic() -> None:
    variants = _targeted_proof_variants(
        "by\n  induction n <;> simp [*]\n  simp [Nat.succ_eq_add_one]"
    )
    assert any(row["proof"] == "by\n  induction n <;> simp [*]" for row in variants)


def test_missing_goal_colon_before_quantifier_uses_declaration_boundary() -> None:
    repaired, changed = normalize_missing_goal_colon(
        "theorem demo \u2200 x : Nat, x = x"
    )
    assert changed
    assert repaired == "theorem demo : \u2200 x : Nat, x = x"


def test_disproved_statement_restores_top_level_negation() -> None:
    repaired, changed = restore_disproved_statement(
        "theorem demo (x : ℝ) (hx : x > 0) : ∀ y : ℝ, y = y"
    )
    assert changed
    assert repaired == (
        "theorem demo (x : ℝ) (hx : x > 0) : ¬ (∀ y : ℝ, y = y)"
    )


def test_full_proof_portfolio_keeps_primary_first_and_unique() -> None:
    variants = _proof_portfolio(
        "by\n  push_neg\n  norm_num",
        "theorem demo : ¬ (∀ x : ℝ, x = x)",
        "disproved",
    )
    assert variants[0]["strategy"] == "manual_compatibility_or_reviewed"
    assert variants[1]["strategy"] == "replace_deprecated_push_neg"
    assert len({row["proof"] for row in variants}) == len(variants)


def test_closed_combinatorial_portfolio_uses_native_decide_before_heavy_tactics() -> None:
    variants = _proof_portfolio(
        "by\n  norm_num [Finset.sum_range_id]",
        "theorem demo : Not (\u2211 k \u2208 Finset.range 2019, k = 2039190)",
        "disproved",
    )
    strategies = [row["strategy"] for row in variants]
    assert strategies[0] == "clean_native_decide"
    assert "clean_ring" not in strategies
    assert "clean_ring_nf" not in strategies


def test_missing_goal_colon_is_restored_before_set_builder_proposition() -> None:
    repaired, changed = normalize_missing_goal_colon(
        "theorem demo {x : ℕ | x > 0} = {1}"
    )
    assert changed
    assert repaired == "theorem demo : {x : ℕ | x > 0} = {1}"
