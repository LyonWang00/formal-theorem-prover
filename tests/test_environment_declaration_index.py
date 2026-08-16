from pathlib import Path

from lean_prover.Mathlib.environment_index import (
    EnvironmentDeclarationIndex,
    LeanCoreStdDeclarationIndex,
)
from lean_prover.Mathlib.error_retrieval import CompilerErrorRetriever
from lean_prover.Mathlib.index import LocalMathlibDeclarationIndex
from lean_prover.Planner.tests.helpers import environment


def _write_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "toolchain" / "src" / "lean"
    init_root = source_root / "Init"
    std_root = source_root / "Std"
    init_root.mkdir(parents=True)
    std_root.mkdir(parents=True)
    (init_root / "Core.lean").write_text(
        "@[simp] protected theorem Nat.add_zero (n : Nat) : n + 0 = n := rfl\n",
        encoding="utf-8",
    )
    (std_root / "Demo.lean").write_text(
        "namespace Std\nlemma demo (n : Nat) : n = n := by rfl\nend Std\n",
        encoding="utf-8",
    )
    return source_root, init_root, std_root


def test_core_std_companion_tracks_origin_and_import_contract(
    tmp_path: Path,
) -> None:
    source_root, init_root, std_root = _write_tree(tmp_path)
    index = LeanCoreStdDeclarationIndex(tmp_path / "core.sqlite3")
    index.ensure_core_std_built(
        source_root=source_root,
        init_root=init_root,
        std_root=std_root,
        environment=environment(),
    )
    add_zero = index.search("Nat.add_zero", parameter_type="Nat", limit=5)[0]
    assert add_zero.declaration.full_name == "Nat.add_zero"
    assert add_zero.declaration.declaration_origin == "lean_core"
    assert add_zero.declaration.required_import == ""
    assert add_zero.declaration.module == "Init.Core"
    assert add_zero.declaration.source_line == 1

    demo = index.search("Std.demo", limit=5)[0]
    assert demo.declaration.declaration_origin == "lean_std"
    assert demo.declaration.required_import == "Std.Demo"


def test_environment_index_merges_and_reranks_old_name_suffix(
    tmp_path: Path,
) -> None:
    source_root, init_root, std_root = _write_tree(tmp_path)
    core = LeanCoreStdDeclarationIndex(tmp_path / "core.sqlite3")
    core.ensure_core_std_built(
        source_root=source_root,
        init_root=init_root,
        std_root=std_root,
        environment=environment(),
    )
    mathlib_root = tmp_path / "checkout" / "Mathlib"
    mathlib_file = mathlib_root / "Data" / "Nat" / "Demo.lean"
    mathlib_file.parent.mkdir(parents=True)
    mathlib_file.write_text(
        "namespace Nat\ntheorem add_comm_demo (n m : Nat) : n + m = m + n := by omega\nend Nat\n",
        encoding="utf-8",
    )
    mathlib = LocalMathlibDeclarationIndex(tmp_path / "mathlib.sqlite3")
    mathlib.ensure_built(
        mathlib_root=mathlib_root,
        environment=environment(),
    )
    merged = EnvironmentDeclarationIndex(mathlib=mathlib, core_std=core)
    results = merged.search(
        "Nat.obsolete_add_zero",
        parameter_type="Nat",
        domain="data_nat",
        limit=10,
    )
    names = [candidate.declaration.full_name for candidate in results]
    assert names[0] == "Nat.add_zero"
    assert any(
        candidate.declaration.declaration_origin == "mathlib"
        for candidate in merged.search("add_comm_demo", limit=10)
    )


def test_error_retriever_does_not_let_generic_context_crowd_primary_name(
    tmp_path: Path,
) -> None:
    source_root, init_root, std_root = _write_tree(tmp_path)
    core = LeanCoreStdDeclarationIndex(tmp_path / "core.sqlite3")
    core.ensure_core_std_built(
        source_root=source_root,
        init_root=init_root,
        std_root=std_root,
        environment=environment(),
    )
    mathlib_root = tmp_path / "checkout" / "Mathlib"
    mathlib_root.mkdir(parents=True)
    mathlib = LocalMathlibDeclarationIndex(tmp_path / "mathlib.sqlite3")
    mathlib.ensure_built(
        mathlib_root=mathlib_root,
        environment=environment(),
    )
    context = CompilerErrorRetriever(
        EnvironmentDeclarationIndex(mathlib=mathlib, core_std=core)
    ).retrieve(
        "error(lean.unknownIdentifier): Unknown constant `Nat.obsolete_add_zero`",
        current_imports=["Mathlib"],
        statement="lemma L1 (n : Nat) : Nat.obsolete_add_zero n = n",
        informal_statement=(
            "For every natural number n, adding zero to n gives n."
        ),
        limit=10,
    )
    assert context.candidates[0].declaration.full_name == "Nat.add_zero"
    assert context.candidates[0].declaration.first_explicit_parameter_type == "Nat"
