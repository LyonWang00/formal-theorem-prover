from __future__ import annotations

from lean_prover.Dataset.repair_numinamath_failures import (
    annotate_first_decimal_as_rational,
    annotate_pure_closed_arithmetic,
    classify_row,
    complete_source,
    eligible_missing_proof,
    eligible_closed_decimal_rational,
    eligible_closed_integer_norm_num,
    eligible_pure_closed_arithmetic,
    eligible_trim_trailing_no_goals,
    eligible_replace_complex_abs,
    eligible_short_placeholder,
    eligible_simple_empty_proof,
    eligible_api_compatibility_renames,
    eligible_big_operator_in_syntax,
    eligible_guarded_complex_placeholder,
    eligible_targeted_placeholder,
    eligible_hazardous_power_grind_placeholder,
    error_family,
    only_no_goals_errors,
    proof_portfolio,
    trailing_tactic_variants,
    short_completion_portfolio,
    has_hazardous_power,
    is_simple_automation_source,
)


def missing_row(source: str) -> dict[str, object]:
    return {
        "record_id": "demo::row_1",
        "formal_ground_truth": "",
        "formal_proof": "",
        "formal_statement": source,
        "proof": "",
        "error_message": "unsolved goals",
    }


def test_missing_proof_classification_distinguishes_forbidden_support() -> None:
    clean = missing_row("import Mathlib\n\ntheorem demo : True := by")
    forbidden = missing_row(
        "import Mathlib\n\nlemma helper : True := by sorry\n\ntheorem demo : True := by"
    )
    assert classify_row(clean) == "missing_proof_clean"
    assert classify_row(forbidden) == "missing_proof_with_forbidden_support"


def test_first_batch_gate_requires_short_single_clean_declaration() -> None:
    row = missing_row("import Mathlib\n\ntheorem demo : True := by")
    assert eligible_missing_proof(row, max_source_chars=200, max_declarations=1) == (
        True,
        "eligible",
    )
    two = missing_row(
        "import Mathlib\n\nlemma helper : True := by trivial\n\ntheorem demo : True := by"
    )
    assert eligible_missing_proof(two, max_source_chars=500, max_declarations=1)[0] is False


def test_complete_source_only_fills_the_final_empty_proof() -> None:
    source = "open Nat\n\ntheorem demo (n : Nat) : n = n := by"
    assert complete_source(source, "by\n  rfl") == (
        "open Nat\n\ntheorem demo (n : Nat) : n = n := by\n  rfl"
    )


def test_complete_source_preserves_backslashes_in_proof_text() -> None:
    source = "theorem demo : True := by"
    proof = "by\n  -- derived from \\sqrt{x}\n  trivial"
    assert complete_source(source, proof) == (
        "theorem demo : True := by\n  -- derived from \\sqrt{x}\n  trivial"
    )


def test_portfolio_is_short_unique_and_goal_sensitive() -> None:
    variants = proof_portfolio(
        "theorem demo (n : Nat) : n + 0 = n := by", max_variants=5
    )
    assert len(variants) == 5
    assert len({row["proof"] for row in variants}) == 5
    assert any(row["strategy"] == "clean_omega" for row in variants)


def test_closed_decimal_goal_prioritizes_arithmetic_tactics() -> None:
    variants = proof_portfolio(
        "theorem demo : 2.4 / 6 = 0.4 := by", max_variants=5
    )
    assert variants[0]["strategy"] == "clean_omega"
    assert variants[1]["strategy"] == "clean_norm_num"


def test_closed_decimal_statement_gets_one_minimal_rational_annotation() -> None:
    source = "/- Compute. -/ theorem demo : 2.4 / 6 = 0.4 := by"
    repaired = annotate_first_decimal_as_rational(source)
    assert repaired is not None
    statement, change = repaired
    assert statement == "/- Compute. -/ theorem demo : (2.4 : ℚ) / 6 = 0.4 := by"
    assert change["kind"] == "disambiguate_decimal_type"


def test_decimal_statement_repair_rejects_bindered_theorem() -> None:
    assert annotate_first_decimal_as_rational(
        "theorem demo (x : ℝ) : x + 2.4 = x + 2.4 := by"
    ) is None


def test_decimal_lane_does_not_repeat_the_same_failed_repair() -> None:
    row = missing_row("theorem demo : 2.4 / 6 = 0.4 := by")
    row["repair"] = {"lane": "closed_decimal_rational"}
    allowed, reason, _, _ = eligible_closed_decimal_rational(
        row, max_source_chars=500, max_declarations=1
    )
    assert allowed is False
    assert reason == "already_attempted_decimal_lane"


def test_closed_integer_lane_accepts_only_closed_non_decimal_goals() -> None:
    closed = missing_row("theorem demo : 17 + 25 = 42 := by")
    decimal = missing_row("theorem demo : 1.5 + 2 = 3.5 := by")
    binder = missing_row("theorem demo (n : Nat) : n + 0 = n := by")
    assert eligible_closed_integer_norm_num(
        closed, max_source_chars=500, max_declarations=1
    )[0] is True
    assert eligible_closed_integer_norm_num(
        decimal, max_source_chars=500, max_declarations=1
    )[0] is False
    assert eligible_closed_integer_norm_num(
        binder, max_source_chars=500, max_declarations=1
    )[0] is False


def test_pure_arithmetic_annotation_uses_signed_type_for_subtraction() -> None:
    repaired = annotate_pure_closed_arithmetic(
        "theorem demo : 8 - 3 = 5 := by"
    )
    assert repaired is not None
    statement, change = repaired
    assert statement == "theorem demo : (8 : ℤ) - 3 = 5 := by"
    assert change["target_type"] == "ℤ"


def test_pure_arithmetic_annotation_rejects_fractional_power_and_identifier() -> None:
    assert annotate_pure_closed_arithmetic(
        "theorem demo : 81 ^ (3 / 4) = 27 := by"
    ) is None
    assert annotate_pure_closed_arithmetic(
        "theorem demo : Nat.choose 5 2 = 10 := by"
    ) is None
    assert annotate_pure_closed_arithmetic(
        "theorem demo : 10 ^ 0.5 = 3 := by"
    ) is None
    assert annotate_pure_closed_arithmetic(
        "theorem demo : 1997 ^ 1999 > 1999 ^ 1997 := by"
    ) is None
    assert annotate_pure_closed_arithmetic(
        "theorem demo : 2 ^ (5 ^ (4 ^ 3)) > 3 ^ 2 := by"
    ) is None


def test_pure_arithmetic_annotation_uses_rational_type_for_decimals() -> None:
    repaired = annotate_pure_closed_arithmetic(
        "theorem demo : 1.5 + 2.5 = 4 := by"
    )
    assert repaired is not None
    statement, change = repaired
    assert statement == "theorem demo : (1.5 : ℚ) + 2.5 = 4 := by"
    assert change["target_type"] == "ℚ"


def test_pure_arithmetic_lane_accepts_final_placeholder_proof() -> None:
    row = missing_row("theorem demo : 17 + 25 = 42 := by sorry")
    allowed, reason, repaired, change = eligible_pure_closed_arithmetic(
        row, max_source_chars=500, max_declarations=1
    )
    assert allowed is True
    assert reason == "eligible"
    assert repaired == "theorem demo : (17 : ℕ) + 25 = 42 := by sorry"
    assert change is not None and change["target_type"] == "ℕ"


def test_no_goals_gate_rejects_mixed_compile_errors() -> None:
    assert only_no_goals_errors("14:2-14:29: error: No goals to be solved\nfailed")
    assert not only_no_goals_errors(
        "10:2-10:9: error: Unknown identifier `x`\n"
        "14:2-14:29: error: No goals to be solved"
    )


def test_trailing_tactic_variants_remove_only_flat_suffixes() -> None:
    variants = trailing_tactic_variants(
        "by\n  simp [foo]\n  unfold bar\n  simp\n  norm_num", max_variants=3
    )
    assert [item["proof"] for item in variants] == [
        "by\n  simp [foo]\n  unfold bar\n  simp",
        "by\n  simp [foo]\n  unfold bar",
        "by\n  simp [foo]",
    ]
    assert trailing_tactic_variants(
        "by\n  cases h with\n  | intro x hx => simp", max_variants=3
    ) == []


def test_trim_lane_uses_statement_and_existing_proof() -> None:
    row = missing_row("unused")
    row.update({
        "lean_statement": "import Mathlib\n\ntheorem demo : True",
        "proof": "by\n  trivial\n  simp",
        "error_message": "4:2-4:6: error: No goals to be solved\nfailed",
    })
    allowed, reason, body, variants = eligible_trim_trailing_no_goals(
        row, max_source_chars=500, max_variants=2
    )
    assert allowed is True and reason == "eligible"
    assert body == "theorem demo : True := by"
    assert variants[0]["proof"] == "by\n  trivial"


def test_complex_abs_lane_replaces_only_the_removed_exact_identifier() -> None:
    row = missing_row("unused")
    row.update({
        "lean_statement": "import Mathlib\n\ntheorem demo (z : ℂ) : Complex.abs z = ‖z‖",
        "proof": "by\n  rw [Complex.abs_re_le_norm]\n  change Complex.abs z = _\n  rfl",
        "error_message": "3:31-3:42: error: Unknown constant `Complex.abs`\nfailed",
    })
    allowed, reason, body, variants, changes, statement_changed = eligible_replace_complex_abs(
        row, max_source_chars=1000
    )
    assert allowed is True and reason == "eligible"
    assert body == "theorem demo (z : ℂ) : norm z = ‖z‖ := by"
    assert "Complex.abs_re_le_norm" in variants[0]["proof"]
    assert "change norm z = _" in variants[0]["proof"]
    assert changes[0]["statement_replacements"] == "1"
    assert statement_changed is True


def test_short_placeholder_gate_ignores_only_the_final_placeholder() -> None:
    row = missing_row("theorem demo (h : True) : True := by sorry")
    assert eligible_short_placeholder(
        row, max_source_chars=500, max_declarations=1
    ) == (True, "eligible")
    row = missing_row(
        "lemma helper : True := by sorry\n\ntheorem demo : True := by sorry"
    )
    assert eligible_short_placeholder(
        row, max_source_chars=500, max_declarations=2
    )[0] is False


def test_short_completion_portfolio_contains_computation_and_reasoning_tactics() -> None:
    strategies = [
        item["strategy"] for item in short_completion_portfolio(max_variants=10)
    ]
    assert strategies[:4] == [
        "short_native_decide",
        "short_linarith",
        "short_norm_num",
        "short_grind",
    ]
    assert "short_simp" not in strategies
    assert strategies[-1] == "short_aesop"


def test_hazardous_power_gate_blocks_nested_large_and_variable_exponents() -> None:
    assert has_hazardous_power("theorem t : 3 ^ 3 ^ 3 % 10 = 7 := by sorry")
    assert has_hazardous_power("theorem t : (x : ℝ) ^ 2004 > 0 := by sorry")
    assert has_hazardous_power("theorem t (a : ℕ → ℕ) : 2 ^ a 3 > 0 := by sorry")
    assert not has_hazardous_power("theorem t (x : ℝ) : x ^ 2 ≥ 0 := by sorry")


def test_simple_automation_gate_separates_high_risk_structures() -> None:
    assert is_simple_automation_source(
        "theorem t (x : ℝ) (h : x = 2) : x = 2 := by sorry"
    )
    assert not is_simple_automation_source(
        "theorem t : IsLeast {n : ℕ | n > 2} 3 := by sorry"
    )
    assert not is_simple_automation_source(
        "theorem t : ∀ n : ℕ, n = n := by sorry"
    )
    assert not is_simple_automation_source(
        "theorem t (x : ℝ) : sqrt x ≥ 0 := by sorry"
    )
    assert not is_simple_automation_source(
        "theorem t (x : ℝ) : 1 / (x + 1) ≤ 1 := by sorry"
    )


def test_simple_empty_lane_requires_safe_unattempted_single_declaration() -> None:
    row = missing_row("theorem t (x : ℝ) (h : x = 2) : x = 2 := by")
    assert eligible_simple_empty_proof(
        row, max_source_chars=500, max_declarations=1
    ) == (True, "eligible")
    row["repair_error"] = "previous attempt failed"
    assert eligible_simple_empty_proof(
        row, max_source_chars=500, max_declarations=1
    ) == (False, "already_attempted")


def test_api_compatibility_lane_changes_only_diagnosed_identifier() -> None:
    row = missing_row(
        "theorem t (s : Finset ℕ) : (s : Set ℕ).ncard = s.card"
    )
    row["proof"] = "by\n  exact Set.ncard_coe_Finset s"
    row["error_message"] = "Unknown constant `Set.ncard_coe_Finset`"
    allowed, reason, body, variants, changes, statement_changed = (
        eligible_api_compatibility_renames(
            row,
            max_source_chars=1000,
            old_name="Set.ncard_coe_Finset",
        )
    )
    assert allowed is True and reason == "eligible"
    assert body is not None and "Set.ncard_coe_Finset" not in body
    assert variants[0]["proof"] == "by\n  exact Set.ncard_coe_finset s"
    assert changes[0]["to"] == "Set.ncard_coe_finset"
    assert statement_changed is False
    row["repair"] = {"changes": changes}
    assert eligible_api_compatibility_renames(
        row,
        max_source_chars=1000,
        old_name="Set.ncard_coe_Finset",
    )[1] == "already_attempted_selected_api"


def test_big_operator_in_lane_updates_statement_and_existing_proof() -> None:
    row = missing_row(
        "theorem t (n : ℕ) : ∑ i in Finset.range n, i = 0"
    )
    row["proof"] = "by\n  change (∑ i in Finset.range n, i) = 0\n  simp"
    row["error_message"] = "3:10-3:12: error: unexpected token 'in'; expected ','"
    allowed, reason, body, variants, changes, statement_changed = (
        eligible_big_operator_in_syntax(row, max_source_chars=1000)
    )
    assert allowed is True and reason == "eligible"
    assert body is not None and "∑ i ∈ Finset.range n" in body
    assert "∑ i ∈ Finset.range n" in variants[0]["proof"]
    assert changes[0]["statement_replacements"] == "1"
    assert changes[0]["proof_replacements"] == "1"
    assert statement_changed is True


def test_guarded_complex_lane_adds_resource_limits_without_changing_statement() -> None:
    row = missing_row("theorem t : IsLeast {n : ℕ | n > 2} 3 := by sorry")
    allowed, reason, body = eligible_guarded_complex_placeholder(
        row, max_source_chars=500, max_declarations=1
    )
    assert allowed is True and reason == "eligible"
    assert body is not None and body.startswith("set_option maxRecDepth 256")
    assert body.endswith("theorem t : IsLeast {n : ℕ | n > 2} 3 := by sorry")


def test_targeted_placeholder_requires_unattempted_row_and_adds_limits() -> None:
    row = missing_row("theorem t (h : True) : True := by sorry")
    allowed, reason, body = eligible_targeted_placeholder(
        row, max_source_chars=500, max_declarations=1
    )
    assert allowed is True and reason == "eligible"
    assert body is not None and body.startswith("set_option maxRecDepth 256")
    row["repair"] = {"lane": "some_previous_lane"}
    assert eligible_targeted_placeholder(
        row, max_source_chars=500, max_declarations=1
    )[1] == "already_attempted_other_repair"


def test_hazardous_power_lane_requires_power_and_uses_tighter_limits() -> None:
    row = missing_row("theorem t (n : ℕ) : n ^ n = n ^ n := by sorry")
    allowed, reason, body = eligible_hazardous_power_grind_placeholder(
        row, max_source_chars=500, max_declarations=1
    )
    assert allowed is True and reason == "eligible"
    assert body is not None and "maxHeartbeats 50000" in body
    safe = missing_row("theorem t (n : ℕ) : n + 0 = n := by sorry")
    assert eligible_hazardous_power_grind_placeholder(
        safe, max_source_chars=500, max_declarations=1
    )[1] == "not_hazardous_power"


def test_error_family_prefers_actionable_compile_type() -> None:
    assert error_family("Unknown identifier `foo`\nunsolved goals") == "unknown_identifier"
    assert error_family("Server reached timeout limit") == "timeout"
