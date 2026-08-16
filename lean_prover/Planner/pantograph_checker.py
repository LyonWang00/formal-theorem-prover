"""Pantograph-backed declaration checks for Planner blueprints."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from lean_prover.backends.pantograph_backend import (
    PantographBackend,
    PantographOptions,
)
from lean_prover.lean_training.expert_iteration.utils import (
    environment_identity as resolve_environment_identity,
)

from .lean_checker import BlueprintLeanChecker
from .preamble import header_context_without_imports, wrap_source_with_preamble
from .schemas import (
    Blueprint,
    LeanCheckResult,
    LeanEnvironmentIdentity,
    LeanPreamble,
    TheoremProblem,
    ValidationIssue,
)


class PantographDeclarationCheckingBackend:
    """Check Planner lemma statements with Pantograph.

    Planner nodes store only declaration statements, for example:

        lemma L1 (n : Nat) : n + 0 = n

    Lean cannot compile a bare theorem/lemma declaration, so this adapter
    builds a temporary source snippet where preceding declarations and the
    current declaration are given placeholder ``sorry`` bodies. The ``sorry``
    bodies are used only inside this checker to ask Pantograph whether the
    statements elaborate in the accumulated environment; they do not modify the
    blueprint.
    """

    def __init__(
        self,
        *,
        project_path: str | Path,
        timeout: int = 120,
        _server: Any = None,
        _api: dict[str, Any] | None = None,
    ) -> None:
        self.project_path = project_path
        self.timeout = timeout
        self._server = _server
        self._api = _api
        self._backend = PantographBackend(
            PantographOptions(
                project_path=project_path,
                imports=(),
                timeout=timeout,
            ),
            _server=_server,
            _api=_api,
        )

    def _backend_for_imports(
        self,
        imports: list[str],
    ) -> PantographBackend:
        selected_imports = tuple(imports)
        if self._backend.options.imports == selected_imports:
            return self._backend

        if self._backend._server is not None:
            self._backend.close()

        self._backend = PantographBackend(
            PantographOptions(
                project_path=self.project_path,
                imports=selected_imports,
                timeout=self.timeout,
            ),
            _server=self._server,
            _api=self._api,
        )
        return self._backend

    @staticmethod
    def _add_sorry_body(declaration: str) -> str:
        stripped = declaration.strip()
        if ":=" in stripped:
            return stripped
        return f"{stripped} := by\n  sorry"

    @staticmethod
    def _source_for_check(
        *,
        imports: list[str],
        declaration: str,
        preceding_declarations: list[str],
        preamble: LeanPreamble,
    ) -> str:
        preceding_block = "\n\n".join(
            PantographDeclarationCheckingBackend._add_sorry_body(item)
            for item in preceding_declarations
        )
        current_block = (
            PantographDeclarationCheckingBackend._add_sorry_body(
                declaration
            )
        )
        declaration_block = "\n\n".join(
            part
            for part in (
                preceding_block,
                current_block,
            )
            if part.strip()
        )
        runtime_preamble = preamble.model_copy(
            update={
                "raw_header": header_context_without_imports(
                    preamble.raw_header
                )
            }
        )
        return wrap_source_with_preamble(declaration_block, runtime_preamble)

    def check_declaration(
        self,
        *,
        imports: list[str],
        declaration: str,
        preceding_declarations: list[str],
        preamble: LeanPreamble,
    ) -> LeanCheckResult:
        source = self._source_for_check(
            imports=imports,
            declaration=declaration,
            preceding_declarations=preceding_declarations,
            preamble=preamble,
        )

        try:
            backend = self._backend_for_imports(imports)
            server = backend._ensure_server()
            units = server.load_sorry(source, ignore_values=True)
        except Exception as error:
            return LeanCheckResult(
                success=False,
                declaration=declaration,
                stderr=str(error),
                error_message=str(error),
            )

        return LeanCheckResult(
            success=True,
            declaration=declaration,
            stdout=(
                "Pantograph accepted declaration; "
                f"distilled {len(units)} search target(s)."
            ),
        )

    def environment_identity(
        self,
        imports: list[str],
    ) -> LeanEnvironmentIdentity:
        """Resolve the exact local identity used in formalization prompts."""

        raw = resolve_environment_identity(self.project_path, imports)
        lean_version = str(raw.get("lean_version") or "")
        match = re.search(r"commit\s+([0-9a-f]{40})", lean_version)
        lean_commit = match.group(1) if match else ""
        mathlib_commit = str(raw.get("mathlib_commit") or "")
        environment_hash = str(raw.get("environment_hash") or "")
        if not lean_commit or not mathlib_commit or not environment_hash:
            raise RuntimeError(
                "Could not resolve exact Lean commit, Mathlib commit, and "
                "environment hash for Planner formalization"
            )
        return LeanEnvironmentIdentity(
            lean_version=lean_version,
            lean_commit=lean_commit,
            mathlib_commit=mathlib_commit,
            environment_hash=environment_hash,
        )

    def check_blueprint(
        self,
        problem: TheoremProblem,
        blueprint: Blueprint,
    ) -> list[ValidationIssue]:
        """Convenience wrapper returning Planner validation issues."""

        return BlueprintLeanChecker(self).check_blueprint(
            problem,
            blueprint,
        )

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "PantographDeclarationCheckingBackend":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
