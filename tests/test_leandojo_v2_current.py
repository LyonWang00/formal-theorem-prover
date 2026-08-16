from __future__ import annotations

from lean_prover.lean_training.data.leandojo_context_builder import (
    extract_active_source_context,
)
from lean_prover.lean_training.data.leandojo_v2_current import (
    CURRENT_ENVIRONMENT_HASH,
    CURRENT_LEAN_VERSION,
    CURRENT_MATHLIB_COMMIT,
    LEANDOJO_V2_COMMIT,
    assemble_source_prefix,
    sha256_text,
    validate_current_record,
)
from lean_prover.lean_training.verification.pantograph import (
    PantographCheckResult,
    PantographTaskVerifier,
)
from lean_prover.lean_training.verification.schema import VerificationTask
from scripts.verify_leandojo_v2_current_sample import _remove_import_commands


def _record() -> dict:
    assembled = "import Mathlib.Data.Nat.Basic\n\nexample : True := by trivial\n"
    return {
        "id": "ldv2_test",
        "source": "leandojo_v2_current_mathlib",
        "repository": "mathlib4",
        "repository_commit": CURRENT_MATHLIB_COMMIT,
        "lean_version": CURRENT_LEAN_VERSION,
        "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
        "source_file": "Mathlib/Test.lean",
        "qualified_name": "Example.target",
        "declaration_kind": "theorem",
        "source_span": {
            "start_line": 8,
            "start_column": 1,
            "end_line": 8,
            "end_column": 39,
        },
        "imports": ["Mathlib.Data.Nat.Basic"],
        "file_dependencies": ["Mathlib/Data/Nat/Basic.lean"],
        "namespace_stack": ["Example"],
        "open_namespaces": [],
        "open_scopes": ["BigOperators"],
        "section_context": ["Local"],
        "variables": [{"name": "α", "type": "Type"}],
        "hypotheses": [],
        "local_instances": [],
        "local_notations": ["local notation \"𝟙\" => 1"],
        "local_attributes": [],
        "statement": "theorem target : True :=",
        "proof": "by trivial",
        "declaration_source": "theorem target : True := by trivial",
        "proof_style": "tactic",
        "tactic_trace": [
            {
                "step_index": 0,
                "state_before": "⊢ True",
                "tactic": "trivial",
                "state_after": "no goals",
                "premises_used": [],
            }
        ],
        "premises": [
            {
                "qualified_name": "Example.helper",
                "source_file": "Mathlib/Test.lean",
                "definition_span": {},
                "usage_span": {},
                "is_same_file": True,
            }
        ],
        "environment_hash": CURRENT_ENVIRONMENT_HASH,
        "trace_hash": "a" * 64,
        "assembled_source_hash": sha256_text(assembled),
        "pantograph_verified": False,
        "metadata": {"assembled_source": assembled},
    }


def test_current_record_schema_and_environment_alignment() -> None:
    assert validate_current_record(_record()) == []


def test_source_prefix_preserves_same_file_context_and_closes_scopes() -> None:
    source = """import Mathlib.Data.Nat.Basic

namespace Example
open scoped BigOperators
section Local
variable (n : Nat)
local notation "myOne" => 1
theorem helper : myOne = 1 := rfl
theorem target : n = n := by rfl
end Local
end Example
"""
    context = extract_active_source_context(
        source,
        declaration_line=9,
        source_file="Mathlib/Test.lean",
        source_commit=CURRENT_MATHLIB_COMMIT,
    )
    assembled = assemble_source_prefix(
        source,
        end_line=9,
        end_column=len("theorem target : n = n := by rfl") + 1,
        closing_commands=context.closing_commands,
    )
    assert "import Mathlib.Data.Nat.Basic" in assembled
    assert "namespace Example" in assembled
    assert "open scoped BigOperators" in assembled
    assert 'local notation "myOne" => 1' in assembled
    assert "theorem helper : myOne = 1 := rfl" in assembled
    assert "theorem target : n = n := by rfl" in assembled
    assert assembled.rstrip().endswith("end Example")


def test_pantograph_body_removes_only_server_loaded_import_commands() -> None:
    source = """-- copyright
module
public import Mathlib.Data.Nat.Basic
public meta import Mathlib.Tactic.ToDual

namespace Example
@[expose] public section
theorem target : True := by trivial
end
end Example
"""
    body = _remove_import_commands(source)
    assert "import Mathlib" not in body
    assert "import Mathlib.Tactic" not in body
    assert "\nmodule\n" not in body
    assert body.startswith("-- copyright")
    assert "theorem target : True := by trivial" in body


def test_context_tracks_modified_anonymous_sections() -> None:
    source = """module
public import Mathlib.Data.Nat.Basic
namespace Example
@[expose] public section
theorem target : True := by trivial
end
end Example
"""
    context = extract_active_source_context(
        source,
        declaration_line=5,
        source_file="Mathlib/Test.lean",
        source_commit=CURRENT_MATHLIB_COMMIT,
    )
    assert context.imports == ("Mathlib.Data.Nat.Basic",)
    assert context.closing_commands == ("end", "end Example")


def test_context_end_with_inline_comment_closes_anonymous_section() -> None:
    source = """module
public import Mathlib.Data.Nat.Basic
@[expose] public section
namespace Function
noncomputable section
theorem helper : True := by trivial
end -- noncomputable
end Function
namespace Target
theorem target : True := by trivial
end Target
end
"""
    context = extract_active_source_context(
        source,
        declaration_line=10,
        source_file="Mathlib/Test.lean",
        source_commit=CURRENT_MATHLIB_COMMIT,
    )
    assert context.namespaces == ("Target",)
    assert context.closing_commands == ("end Target", "end")


class _FakeVerifier:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def check_source(self, source: str, **_: object) -> PantographCheckResult:
        self.sources.append(source)
        return PantographCheckResult(True, "", 0.01)


def test_preassembled_source_bypasses_evaluation_namespace() -> None:
    verifier = object.__new__(PantographTaskVerifier)
    verifier.default_imports = ("Mathlib.Data.Nat.Basic",)
    verifier.timeout = 30
    verifier.verifier = _FakeVerifier()
    verifier.restart_count = 0
    source = "example : True := by trivial\n"
    task = VerificationTask(
        priority=0,
        problem_index=0,
        attempt_index=0,
        problem_id="test",
        prompt="",
        generated_proof="by trivial",
        raw_completion="by trivial",
        lean_code=source,
        imports=("Mathlib.Data.Nat.Basic",),
        payload={"preassembled_source": True},
    )
    result = verifier.verify(task)
    assert result["success"] is True
    assert verifier.verifier.sources == [source]
    assert "EvalProblem" not in verifier.verifier.sources[0]


def test_preassembled_mode_is_opt_in() -> None:
    verifier = object.__new__(PantographTaskVerifier)
    verifier.default_imports = ("Mathlib",)
    verifier.timeout = 30
    verifier.verifier = _FakeVerifier()
    verifier.restart_count = 0
    task = VerificationTask(
        priority=0,
        problem_index=4,
        attempt_index=2,
        problem_id="test",
        prompt="",
        generated_proof="by trivial",
        raw_completion="by trivial",
        lean_code="example : True := by trivial",
        imports=("Mathlib",),
    )
    verifier.verify(task)
    assert "namespace EvalProblem_4_Attempt_2" in verifier.verifier.sources[0]
