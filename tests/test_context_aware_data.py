from __future__ import annotations

from lean_prover.lean_training.data.context import (
    LeanSourceContext,
    apply_context_to_record,
    assemble_context_aware_declaration,
    context_record_payload,
    stable_json_hash,
)
from lean_prover.lean_training.data.contracts import DataState, LeanDataRecord
from lean_prover.lean_training.data.leandojo_context_builder import (
    extract_active_source_context,
)
from lean_prover.lean_training.data.preparation import ProofFormat
from lean_prover.lean_training.data.lean_workbook_context_builder import (
    build_lean_workbook_context,
)
from lean_prover.lean_training.verification.cache import (
    VerificationCache,
    make_context_cache_key,
)


def fixture_record() -> LeanDataRecord:
    return LeanDataRecord(
        record_id="fixture",
        statement_id="stmt_fixture",
        data_state=DataState.RAW,
        source_dataset="fixture",
        statement="theorem t : True",
        proof="by\n  exact contextTruth",
        proof_format=ProofFormat.FULL_PROOF,
        assembler_version="2",
        normalization_version="2",
    )


def test_context_schema_preserves_structured_source_fields() -> None:
    context = LeanSourceContext(
        imports=("Mathlib.Data.Nat.Basic",),
        namespaces=("Demo",),
        open_namespaces=("Nat",),
        scopes=("BigOperators",),
        variables=(
            {
                "name": "α",
                "type": "Type*",
                "binder": "implicit",
                "raw": "{α : Type*}",
            },
        ),
        hypotheses=(
            {
                "name": "h",
                "type": "True",
                "binder": "explicit",
                "raw": "h : True",
            },
        ),
        local_instances=("Group G",),
        local_notations=('local notation "contextTruth" => True.intro',),
        sections=("Local",),
        active_commands=(
            "namespace Demo",
            "section Local",
            'local notation "contextTruth" => True.intro',
        ),
        closing_commands=("end Local", "end Demo"),
        context_recovered=True,
        recovery_method="fixture",
        recovery_status="full",
        recovery_sources=("source_file",),
        source_hash="source",
    )
    first = apply_context_to_record(
        fixture_record(),
        context,
        environment_hash="environment",
    )
    second = apply_context_to_record(
        fixture_record(),
        context,
        environment_hash="environment",
    )
    payload = context_record_payload(first)
    assert payload["imports"] == ["Mathlib.Data.Nat.Basic"]
    assert payload["namespaces"] == ["Demo"]
    assert payload["scopes"] == ["BigOperators"]
    assert payload["variables"][0]["name"] == "α"
    assert payload["hypotheses"][0]["name"] == "h"
    assert payload["context_recovery_status"] == "full"
    assert first.metadata["context_hashes"] == second.metadata["context_hashes"]


def test_historical_source_parser_extracts_only_active_context() -> None:
    source = """import Mathlib.Data.Nat.Basic
open Nat
noncomputable section
namespace Demo
variable {α : Type*}
section Old
local notation "old" => True
end Old
section Local
open scoped BigOperators
/-!
  a comment that must not become part of `open scoped`.
-/
local macro "contextTruth" : term => `(True.intro)
attribute [local simp] Nat.add_zero
variable (obsolete) in
lemma previous : True := by trivial
theorem t : True := by
  exact contextTruth
end Local
end Demo
end
"""
    context = extract_active_source_context(
        source,
        declaration_line=18,
        source_file="Mathlib/Demo.lean",
        source_commit="abc",
    )
    rendered = "\n".join(context.active_commands)
    assert context.imports == ("Mathlib.Data.Nat.Basic",)
    assert context.namespaces == ("Demo",)
    assert context.sections == ("Local",)
    assert context.open_namespaces == ("Nat",)
    assert context.scopes == ("BigOperators",)
    assert "contextTruth" in rendered
    assert '"old"' not in rendered
    assert "a comment" not in rendered
    assert "obsolete" not in rendered
    assert context.closing_commands == ("end Local", "end Demo", "end")


def test_context_aware_assembly_replays_context_around_declaration() -> None:
    context = LeanSourceContext(
        active_commands=('local macro "contextTruth" : term => `(True.intro)',),
        context_recovered=True,
        recovery_method="fixture",
        recovery_status="full",
        recovery_sources=("source_file",),
    )
    record = apply_context_to_record(
        fixture_record(),
        context,
        environment_hash="environment",
    )
    legacy = "theorem t : True := by\n  exact contextTruth"
    rebuilt = assemble_context_aware_declaration(record)
    assert "local macro" not in legacy
    assert "local macro" in rebuilt
    assert "exact contextTruth" in rebuilt


def test_cache_key_misses_when_any_context_component_changes() -> None:
    base = {
        "statement_hash": "statement",
        "proof_hash": "proof",
        "imports_hash": "imports",
        "namespace_hash": "namespace",
        "scope_hash": "scope",
        "variable_context_hash": "variables",
        "context_hash": "context",
        "assembled_source_hash": "assembled",
        "environment_hash": "environment",
        "verifier_version": "pantograph",
        "assembler_version": "assembler",
        "normalization_version": "normalization",
    }
    original = make_context_cache_key(**base)
    for field in (
        "imports_hash",
        "namespace_hash",
        "scope_hash",
        "variable_context_hash",
        "context_hash",
        "assembled_source_hash",
        "environment_hash",
        "verifier_version",
        "assembler_version",
        "normalization_version",
    ):
        changed = {**base, field: base[field] + "-changed"}
        assert make_context_cache_key(**changed) != original


def test_cache_hits_only_for_exact_source_and_environment(tmp_path) -> None:
    cache = VerificationCache(tmp_path / "verification.sqlite")
    try:
        cache.put(
            cache_key="exact",
            proof_hash="proof",
            statement_hash="statement",
            environment_hash="environment",
            assembler_version="assembler",
            normalization_version="normalization",
            status="success",
            error_type=None,
            source_path=None,
            verification={"success": True, "cache_hit": False},
        )
        assert cache.get(
            "exact",
            environment_hash="environment",
            assembler_version="assembler",
            normalization_version="normalization",
        ) == {"success": True, "cache_hit": False}
        assert (
            cache.get(
                "exact",
                environment_hash="changed-environment",
                assembler_version="assembler",
                normalization_version="normalization",
            )
            is None
        )
    finally:
        cache.close()


def test_stable_hash_ignores_mapping_field_order_but_not_content() -> None:
    first = {"imports": ["Mathlib"], "context": {"x": "Nat", "h": "x > 0"}}
    reordered = {"context": {"h": "x > 0", "x": "Nat"}, "imports": ["Mathlib"]}
    changed = {"context": {"h": "x ≥ 0", "x": "Nat"}, "imports": ["Mathlib"]}
    assert stable_json_hash(first) == stable_json_hash(reordered)
    assert stable_json_hash(first) != stable_json_hash(changed)


def test_workbook_context_is_partial_and_never_replays_proof_state() -> None:
    record = fixture_record().model_copy(
        update={
            "record_id": "workbook",
            "source_dataset": "InternLM/Lean-Workbook",
            "statement": "theorem t (x : Nat) (h : x = x) : True",
            "proof": "by\n  trivial",
            "imports": ["Mathlib"],
        }
    )
    raw_rows = [
        {
            "state_before": "x : Nat\nh : x = x\n⊢ True",
            "state_after": "no goals",
            "tactic": "trivial",
        }
    ]
    rebuilt = build_lean_workbook_context(
        record,
        raw_rows,
        environment_hash="environment",
    )
    source = assemble_context_aware_declaration(rebuilt)
    assert rebuilt.context_recovery_status == "partial"
    assert rebuilt.context_recovered is False
    assert rebuilt.context_recovery_sources == [
        "proof_state",
        "statement",
        "tactic_trace",
    ]
    assert rebuilt.variables[0]["name"] == "x"
    assert rebuilt.hypotheses[0]["name"] == "h"
    assert source == "theorem t (x : Nat) (h : x = x) : True := by\n  trivial"
    assert "\nx : Nat\n" not in source


def test_nested_namespace_and_section_closings_are_exact() -> None:
    source = """import Mathlib
namespace Outer
section Local
namespace Inner
variable {α : Type*}
theorem t : True := by trivial
end Inner
end Local
end Outer
"""
    context = extract_active_source_context(
        source,
        declaration_line=6,
        source_file="Mathlib/Nested.lean",
        source_commit="abc",
    )
    assert context.namespaces == ("Outer", "Inner")
    assert context.sections == ("Local",)
    assert context.closing_commands == (
        "end Inner",
        "end Local",
        "end Outer",
    )
    assert len(context.closing_commands) == 3
