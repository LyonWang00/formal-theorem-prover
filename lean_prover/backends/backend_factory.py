"""Construction of the configured interactive Lean backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .pantograph_backend import PantographBackend, PantographOptions
from .proof_backend import InteractiveProofBackend


def _default_project_path() -> Path:
    """Return the repository-local Lean project without host assumptions."""

    return Path(__file__).resolve().parents[2] / "lean_project"


@dataclass(frozen=True)
class BackendConfig:
    """Backend selection shared by CLI/search entry points."""

    name: str = "pantograph"
    project_path: str | Path = field(default_factory=_default_project_path)
    imports: tuple[str, ...] = ("Mathlib",)
    timeout: int = 120

    @classmethod
    def from_environment(cls) -> "BackendConfig":
        return cls(
            name=os.environ.get("LEAN_BACKEND", "pantograph"),
            # Docker and external runtimes override this variable. A native
            # checkout otherwise uses its own Lean project.
            project_path=os.environ.get(
                "LEAN_PROJECT_PATH", str(_default_project_path())
            ),
            timeout=int(os.environ.get("LEAN_BACKEND_TIMEOUT", "120")),
        )


def create_backend(
    config: BackendConfig | None = None,
) -> InteractiveProofBackend:
    """Create the configured Pantograph backend."""

    selected = config or BackendConfig.from_environment()
    name = selected.name.strip().lower()
    if name == "pantograph":
        return PantographBackend(
            PantographOptions(
                project_path=selected.project_path,
                imports=selected.imports,
                timeout=selected.timeout,
            )
        )
    raise ValueError(
        f"unknown Lean backend {selected.name!r}; "
        "expected 'pantograph'"
    )
