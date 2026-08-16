from pathlib import Path

import pytest

from lean_prover.Dataset.minif2f_compat import (
    MINIF2F_IMPORT_COMPATIBILITY,
    mapped_imports,
    rewrite_header,
    rewrite_statement,
    validate_mapped_modules_exist,
)
from scripts.verify_minif2f_compat import _failure_category


def test_rewrite_header_is_deterministic_and_preserves_non_import_lines() -> None:
    header = (
        "import Mathlib.Algebra.BigOperators.Basic\n"
        "import Mathlib.Data.Complex.Exponential\n"
        "import Mathlib.Data.Nat.Digits\n\n"
        "open BigOperators\nopen Nat"
    )
    assert rewrite_header(header) == (
        "import Mathlib.Algebra.BigOperators.Group.Finset.Basic\n"
        "import Mathlib.Analysis.Complex.Exponential\n"
        "import Mathlib.Analysis.SpecialFunctions.Log.Base\n"
        "import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n"
        "import Mathlib.NumberTheory.Real.Irrational\n"
        "import Mathlib.Data.Nat.Digits.Lemmas\n\n"
        "open BigOperators\nopen Nat"
    )
    assert mapped_imports(header) == [
        "Mathlib.Algebra.BigOperators.Group.Finset.Basic",
        "Mathlib.Analysis.Complex.Exponential",
        "Mathlib.Analysis.SpecialFunctions.Log.Base",
        "Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
        "Mathlib.NumberTheory.Real.Irrational",
        "Mathlib.Data.Nat.Digits.Lemmas",
    ]


def test_unmapped_import_is_rejected() -> None:
    with pytest.raises(ValueError, match="unmapped miniF2F import"):
        rewrite_header("import Mathlib.Unregistered.Module")


def test_every_mapping_target_exists_in_current_mathlib() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "lean_project/.lake/packages/mathlib"
    )
    validate_mapped_modules_exist(root)
    assert len(MINIF2F_IMPORT_COMPATIBILITY) == 10


@pytest.mark.parametrize(
    ("diagnostic", "category"),
    [
        ("unexpected token 'in'; expected ','", "deleted_big_operator_in_syntax"),
        ("unexpected end of input", "commented_out_no_declaration"),
        ("Unknown constant `Complex.abs`", "removed_complex_abs"),
        ("Ambiguous term lcm", "ambiguous_lcm"),
    ],
)
def test_failure_categories_are_stable(
    diagnostic: str,
    category: str,
) -> None:
    assert _failure_category(diagnostic) == category


def test_rewrite_statement_updates_all_legacy_big_operator_binders() -> None:
    statement = (
        "theorem demo :\n"
        "  (∑ x in Finset.range 10, x) + "
        "(∏ y in Finset.Icc 1 3, y) = 0 := sorry"
    )
    rewritten, repairs = rewrite_statement(statement)
    assert "∑ x ∈ Finset.range 10, x" in rewritten
    assert "∏ y ∈ Finset.Icc 1 3, y" in rewritten
    assert repairs == ["rewrite_big_operator_in_to_membership"]


def test_rewrite_statement_restores_commented_theorem() -> None:
    statement = (
        "-- Error: Real.log\n"
        "-- theorem demo\n"
        "--   (x : ℝ) :\n"
        "--   Real.log x = 0 := sorry"
    )
    rewritten, repairs = rewrite_statement(statement)
    assert rewritten == (
        "theorem demo\n  (x : ℝ) :\n  Real.log x = 0 := sorry"
    )
    assert repairs == ["uncomment_formal_declaration"]


def test_rewrite_statement_updates_removed_and_ambiguous_names() -> None:
    statement = (
        "theorem demo (a b : ℂ) (m n : ℕ) : "
        "Complex.abs (a - b) ≤ Nat.lcm m n + lcm m n := sorry"
    )
    rewritten, repairs = rewrite_statement(statement)
    assert "‖a - b‖" in rewritten
    assert rewritten.count("Nat.lcm m n") == 2
    assert repairs == [
        "rewrite_complex_abs_to_norm",
        "qualify_nat_lcm",
    ]


def test_rewrite_statement_repairs_exposed_type_and_declaration_issues() -> None:
    statement = (
        "imosl_2007_algebra_p6\n"
        "  (n : ℕ)\n"
        "  (h : ∏ k in Finset.Icc 1 n, (1 + 1 / k^3) = 1) :\n"
        "  irrational (2 : ℝ) := sorry"
    )
    rewritten, repairs = rewrite_statement(statement)
    assert rewritten.startswith("theorem imosl_2007_algebra_p6")
    assert "(1 + (1 : ℝ) / (k : ℝ)^3)" in rewritten
    assert "Irrational (2 : ℝ)" in rewritten
    assert repairs == [
        "rewrite_big_operator_in_to_membership",
        "disambiguate_real_product_term",
        "capitalize_irrational_predicate",
        "restore_missing_theorem_keyword",
    ]
