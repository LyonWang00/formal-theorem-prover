from __future__ import annotations

from lean_prover.Dataset.verify_external_datasets import (
    SPECS,
    effective_success,
    extract_numina_proof,
    normalize_kimina_target,
    select_numina_source,
    split_imports,
    strip_placeholder_statement,
)


def test_split_imports_keeps_declaration_body() -> None:
    imports, body = split_imports(
        "import Mathlib\nimport Mathlib.Data.Nat.Basic\n\ntheorem t : True := by trivial"
    )
    assert imports == ("Mathlib", "Mathlib.Data.Nat.Basic")
    assert body == "theorem t : True := by trivial"


def test_numina_proof_extraction() -> None:
    source = "import Mathlib\n\ntheorem t : True := by\n  trivial"
    assert extract_numina_proof(source) == "by\n  trivial"
    assert extract_numina_proof("theorem t : True := by") == ""


def test_kimina_placeholder_is_removed_from_statement() -> None:
    source = "import Mathlib\n\ntheorem t (n : Nat) : n = n := by sorry"
    assert strip_placeholder_statement(source) == "theorem t (n : Nat) : n = n"


def test_numina_falls_back_to_complete_formal_proof_source() -> None:
    row = {
        "formal_statement": "theorem t : True := by",
        "formal_ground_truth": "",
        "formal_proof": "import Mathlib\n\ntheorem t : True := by trivial",
    }
    field, source = select_numina_source(row)
    assert field == "formal_proof"
    assert source.endswith("by trivial")


def test_kimina_target_normalization_preserves_supporting_prefix() -> None:
    source = """import Mathlib
open Nat

theorem unrelated : True := by trivial

theorem target (n : Nat) : n = n :=
"""
    imports, statement, normalized, metadata = normalize_kimina_target(
        source, name="target"
    )
    assert imports == ("Mathlib",)
    assert statement == "theorem target (n : Nat) : n = n"
    assert "unrelated" in normalized
    assert normalized.endswith(":= by sorry")
    assert metadata["supporting_prefix_declarations"] == 1


def test_kimina_warning_only_result_is_statement_success() -> None:
    result = {
        "success": False,
        "status": "failed",
        "compile_errors": ["4:8-4:34: warning: declaration uses `sorry`"],
        "timed_out": False,
    }
    assert effective_success(result, spec=SPECS["kimina"]) is True
    assert effective_success(result, spec=SPECS["numinamath"]) is False
