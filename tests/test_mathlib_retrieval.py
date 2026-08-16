from pathlib import Path

from lean_prover.Mathlib import (
    CompilerErrorCategory,
    CompilerErrorRetriever,
    LocalMathlibDeclarationIndex,
    classify_compiler_error,
)
from lean_prover.Mathlib.index import extract_declarations
from lean_prover.Planner.schemas import LeanEnvironmentIdentity


def _environment() -> LeanEnvironmentIdentity:
    return LeanEnvironmentIdentity(
        lean_version="Lean (version 4.19.0, commit " + "1" * 40 + ")",
        lean_commit="1" * 40,
        mathlib_commit="2" * 40,
        environment_hash="3" * 64,
    )


def _fixture_index(tmp_path: Path) -> LocalMathlibDeclarationIndex:
    mathlib_root = tmp_path / "checkout" / "Mathlib"
    source = mathlib_root / "Data" / "Demo.lean"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
namespace Nat
section Local
def factorization (n : Nat) : Nat := n
end Local
protected theorem Prime.factorization {p : Nat} (hp : p.Prime) : True := by
  trivial
end Nat
namespace Other
lemma factorization_aux (n : Nat) : n = n := by rfl
end Other
""".strip(),
        encoding="utf-8",
    )
    index = LocalMathlibDeclarationIndex(tmp_path / "index.sqlite3")
    index.ensure_built(mathlib_root=mathlib_root, environment=_environment())
    return index


def test_source_extraction_tracks_sections_without_corrupting_namespace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout" / "Mathlib"
    source = root / "Data" / "Demo.lean"
    source.parent.mkdir(parents=True)
    source.write_text(
        "namespace Nat\nsection X\ndef f (n : Nat) : Nat := n\nend X\n"
        "protected theorem Prime.g (p : Nat) : True := by trivial\nend Nat\n",
        encoding="utf-8",
    )
    declarations = list(extract_declarations(source, root))
    assert [item.full_name for item in declarations] == ["Nat.f", "Nat.Prime.g"]
    assert declarations[1].namespace == "Nat.Prime"
    assert declarations[0].first_explicit_parameter_type == "Nat"


def test_equation_body_is_not_part_of_signature(tmp_path: Path) -> None:
    root = tmp_path / "checkout" / "Mathlib"
    source = root / "Data" / "Equation.lean"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def branch : Nat → Nat\n  | 0 => 0\n  | n + 1 => n\n",
        encoding="utf-8",
    )
    declaration = list(extract_declarations(source, root))[0]
    assert declaration.type_signature == "branch : Nat → Nat"
    assert declaration.first_explicit_parameter_type == "Nat"


def test_anonymous_instance_header_is_not_misread_as_a_name(tmp_path: Path) -> None:
    root = tmp_path / "checkout" / "Mathlib"
    source = root / "Data" / "Anonymous.lean"
    source.parent.mkdir(parents=True)
    source.write_text(
        "instance [Fact (1 < n)] : Nonempty Nat := inferInstance\n",
        encoding="utf-8",
    )
    assert list(extract_declarations(source, root)) == []


def test_index_search_orders_exact_before_fuzzy_and_caches_identity(
    tmp_path: Path,
) -> None:
    index = _fixture_index(tmp_path)
    results = index.search(
        "factorization", parameter_type="Nat", domain="number_theory", limit=5
    )
    assert results
    assert results[0].declaration.full_name == "Nat.factorization"
    assert results[0].match_stage == "exact"
    assert index.metadata()["mathlib_commit"] == "2" * 40
    assert index.metadata()["environment_hash"] == "3" * 64
    qualified_old = index.search("Nat.factorization_old", limit=5)
    assert len(qualified_old) >= 3
    assert any(
        item.declaration.full_name == "Nat.factorization"
        for item in qualified_old
    )


def test_compiler_error_classification_extracts_names_types_and_imports() -> None:
    invalid_field = classify_compiler_error(
        "Invalid field notation: Type of\n  n\nis Nat; cannot resolve field `factorization`",
        current_imports=["Mathlib"],
    )
    assert invalid_field.category == CompilerErrorCategory.INVALID_FIELD_NOTATION
    assert invalid_field.missing_name == "factorization"
    assert invalid_field.receiver_type == "Nat"
    assert invalid_field.current_imports == ("Mathlib",)

    mismatch = classify_compiler_error(
        "application type mismatch\n  f x\nargument x has type\n  Int\n"
        "but is expected to have type\n  Nat"
    )
    assert mismatch.category == CompilerErrorCategory.APPLICATION_TYPE_MISMATCH
    assert mismatch.expected_type == "Nat"

    parser = classify_compiler_error("unexpected token 'begin'; expected term")
    assert parser.category == CompilerErrorCategory.PARSER_OR_OLD_SYNTAX

    missing_import = classify_compiler_error(
        "failed to import module Mathlib.Old.Module"
    )
    assert missing_import.category == CompilerErrorCategory.MISSING_IMPORT
    assert missing_import.missing_import == "Mathlib.Old.Module"

    ambiguous = classify_compiler_error("ambiguous identifier `map`")
    assert ambiguous.category == CompilerErrorCategory.AMBIGUOUS_NAMESPACE

    unknown_constant = classify_compiler_error("unknown constant `Nat.oldApi`")
    assert unknown_constant.category == CompilerErrorCategory.UNKNOWN_CONSTANT


def test_error_retriever_returns_only_indexed_local_declarations(
    tmp_path: Path,
) -> None:
    retriever = CompilerErrorRetriever(_fixture_index(tmp_path))
    context = retriever.retrieve(
        "unknown identifier 'factorization'",
        current_imports=["Mathlib"],
        statement="lemma L1 (n : Nat) : Nat.factorization n = n",
        informal_statement="The factorization of n has the stated property.",
    )
    assert context.analysis.category == CompilerErrorCategory.UNKNOWN_IDENTIFIER
    assert context.candidates
    names = [item.declaration.full_name for item in context.candidates]
    assert "Nat.factorization" in names
    assert names.index("Nat.factorization") < names.index("Other.factorization_aux")
    assert all(
        item.declaration.required_import == "Mathlib.Data.Demo"
        for item in context.candidates
    )
    assert all(
        item.declaration.declaration_origin == "mathlib"
        for item in context.candidates
    )
