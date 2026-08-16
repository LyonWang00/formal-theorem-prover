"""Atomic artifact I/O and environment fingerprints."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
)

from .schemas import stable_hash


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows without retaining the artifact in process memory."""

    source = Path(path).expanduser()
    if not source.exists():
        return
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {source}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {source}:{line_number}")
            yield value


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append and flush durable JSONL rows, returning the appended row count."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    return count


def count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], *, cwd: str | Path | None = None) -> str | None:
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


def environment_identity(lean_project_path: str | Path, imports: Iterable[str]) -> dict[str, Any]:
    project = Path(lean_project_path).expanduser().resolve()
    lean_version = (
        # Version discovery must be read-only.  Invoking ``lake env`` can
        # resolve or refresh packages and has no place in an identity probe.
        command_output(["lean", "--version"], cwd=project)
        or "unknown"
    )
    mathlib_commit = _mathlib_commit(project)
    environment_hash = stable_hash(
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
    return command_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project / ".lake" / "packages" / "mathlib",
    )
