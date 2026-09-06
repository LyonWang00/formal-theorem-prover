"""Read-only runtime identity helpers for reproducible Lean verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
)


def _stable_hash(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip() or None


def _mathlib_commit(project: Path) -> str | None:
    manifest_path = project / "lake-manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for package in manifest.get("packages", []):
                if package.get("name") == "mathlib" and package.get("rev"):
                    return str(package["rev"])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return _command_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project / ".lake" / "packages" / "mathlib",
    )


def environment_identity(
    lean_project_path: str | Path,
    imports: Iterable[str],
) -> dict[str, Any]:
    """Return a reproducibility fingerprint without mutating the Lean project."""

    project = Path(lean_project_path).expanduser().resolve()
    lean_version = _command_output(["lean", "--version"], cwd=project) or "unknown"
    mathlib_commit = _mathlib_commit(project)
    environment_hash = _stable_hash(
        lean_version,
        mathlib_commit or "unknown",
        ASSEMBLER_VERSION,
        NORMALIZATION_VERSION,
        *sorted(imports),
    )
    return {
        "lean_version": lean_version,
        "mathlib_commit": mathlib_commit,
        "assembler_version": ASSEMBLER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "environment_hash": environment_hash,
    }
