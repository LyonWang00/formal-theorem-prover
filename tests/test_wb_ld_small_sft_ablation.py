from lean_prover.lean_training.expert_iteration.discovery_verifier import (
    _remove_preloaded_import_commands,
)


def test_source_faithful_template_removes_only_preloaded_commands() -> None:
    source = """module

import Mathlib.Algebra.Group.Defs
public import Mathlib.Data.Nat.Basic

namespace Example
theorem kept : True := __CODEX_GENERATED_PROOF__
end Example
"""

    cleaned = _remove_preloaded_import_commands(source)

    assert "module" not in cleaned
    assert "import Mathlib" not in cleaned
    assert cleaned.startswith("namespace Example")
    assert "theorem kept : True := __CODEX_GENERATED_PROOF__" in cleaned
    assert cleaned.endswith("end Example\n")


def test_source_faithful_template_preserves_import_text_inside_code() -> None:
    source = """def message : String := "import Mathlib"
-- import Mathlib is documentation here
theorem kept : True := __CODEX_GENERATED_PROOF__
"""

    assert _remove_preloaded_import_commands(source) == source
