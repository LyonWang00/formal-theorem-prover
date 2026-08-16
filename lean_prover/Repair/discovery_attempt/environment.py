"""Load and fingerprint the exact EI discovery repair environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from .schema import EnvironmentContract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_environment_contract(
    lean_project: str | Path,
    *,
    imports: tuple[str, ...] = ("Mathlib",),
    lean_version_output: str | None = None,
    pantograph_version: str | None = None,
) -> EnvironmentContract:
    """Read exact versions; fail closed when Lean or Mathlib cannot be pinned."""

    project = Path(lean_project).expanduser().resolve()
    toolchain_path = project / "lean-toolchain"
    manifest_path = project / "lake-manifest.json"
    if not toolchain_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("lean-toolchain and lake-manifest.json are required")

    toolchain = toolchain_path.read_text(encoding="utf-8").strip()
    version_match = re.search(r"v(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", toolchain)
    if not version_match:
        raise RuntimeError(f"cannot parse Lean version from {toolchain_path}")
    lean_version = version_match.group(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mathlib = next(
        (row for row in manifest.get("packages", []) if row.get("name") == "mathlib"),
        None,
    )
    mathlib_commit = str((mathlib or {}).get("rev") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", mathlib_commit):
        raise RuntimeError("lake-manifest.json does not pin Mathlib to a full commit")

    if lean_version_output is None:
        completed = subprocess.run(
            ["lake", "env", "lean", "--version"],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        lean_version_output = completed.stdout + completed.stderr
    commit_match = re.search(r"commit\s+([0-9a-f]{40})", lean_version_output)
    if not commit_match:
        raise RuntimeError("Lean --version output does not contain a full commit")
    lean_commit = commit_match.group(1)
    reported_version = re.search(r"version\s+(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", lean_version_output)
    if reported_version and reported_version.group(1) != lean_version:
        raise RuntimeError(
            f"Lean toolchain mismatch: file={lean_version}, executable={reported_version.group(1)}"
        )

    if pantograph_version is None:
        try:
            pantograph_version = importlib.metadata.version("pantograph")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("Pantograph package version cannot be determined") from error

    repository_commit = "unknown"
    try:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return EnvironmentContract(
        lean_version=lean_version,
        lean_commit=lean_commit,
        mathlib_commit=mathlib_commit,
        pantograph_version=str(pantograph_version),
        imports=list(imports),
        lean_toolchain_sha256=file_sha256(toolchain_path),
        lake_manifest_sha256=file_sha256(manifest_path),
        repository_commit=repository_commit or "unknown",
    )
