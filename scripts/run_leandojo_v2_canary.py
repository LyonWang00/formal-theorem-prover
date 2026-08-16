"""Trace a fixed, explicit set of files with LeanDojo-v2's ExtractData.lean.

This utility intentionally does not call LeanDojo's repository-wide tracing
entry point.  It runs the extractor once per selected source file inside an
already isolated and built repository clone, recording all commands, hashes,
return codes, elapsed time, and peak RSS reported by ``/usr/bin/time``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_FILES = (
    "Mathlib/Algebra/Order/Field/Defs.lean",
    "Mathlib/NumberTheory/PrimeCounting.lean",
    "Mathlib/Data/Nat/Choose/Basic.lean",
    "Mathlib/Analysis/Calculus/Deriv/Basic.lean",
    "Mathlib/Tactic/Linarith/Frontend.lean",
    "Mathlib/Topology/UnitInterval.lean",
    "Mathlib/Combinatorics/SimpleGraph/Basic.lean",
    "Mathlib/CategoryTheory/Functor/Basic.lean",
    "Mathlib/MeasureTheory/MeasurableSpace/Basic.lean",
    "Mathlib/LinearAlgebra/FiniteDimensional/Basic.lean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_artifact(repo: Path, source: Path, suffix: str) -> Path:
    relative = source.relative_to(repo)
    return repo / ".lake/build/ir" / relative.with_suffix(suffix)


def _max_rss_kib(stderr: str) -> int | None:
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr)
    return int(match.group(1)) if match else None


def run_canary(repo: Path, tool_root: Path, output: Path, files: list[str]) -> None:
    repo = repo.resolve(strict=True)
    tool_root = tool_root.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    raw = output / "raw"
    logs.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)

    extractor_source = (
        tool_root
        / "lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean"
    ).resolve(strict=True)
    extractor_target = repo / "ExtractData.lean"
    source_hash = _sha256(extractor_source)
    if extractor_target.exists() and _sha256(extractor_target) != source_hash:
        raise RuntimeError(
            f"refusing to overwrite a different extractor: {extractor_target}"
        )
    if not extractor_target.exists():
        shutil.copy2(extractor_source, extractor_target)

    selected: list[dict[str, Any]] = []
    for relative in files:
        source = (repo / relative).resolve(strict=True)
        if not source.is_relative_to(repo):
            raise ValueError(f"source escapes repository: {source}")
        selected.append(
            {
                "source_file": relative,
                "source_sha256": _sha256(source),
                "source_bytes": source.stat().st_size,
            }
        )
    (output / "selected_files.json").write_text(
        json.dumps(
            {
                "selected_at": datetime.now(UTC).isoformat(),
                "repository": str(repo),
                "extractor": str(extractor_source),
                "extractor_sha256": source_hash,
                "files": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        relative = item["source_file"]
        command = [
            "/usr/bin/time",
            "-v",
            "lake",
            "env",
            "lean",
            "--threads",
            "2",
            "--run",
            "ExtractData.lean",
            relative,
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.monotonic() - started
        log_base = logs / f"{index:02d}_{Path(relative).stem}"
        log_base.with_suffix(".stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        log_base.with_suffix(".stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )

        source = repo / relative
        ast = _build_artifact(repo, source, ".ast.json")
        dependencies = _build_artifact(repo, source, ".dep_paths")
        if ast.exists():
            destination = raw / ast.relative_to(repo / ".lake/build/ir")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ast, destination)
        if dependencies.exists():
            destination = raw / dependencies.relative_to(repo / ".lake/build/ir")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependencies, destination)

        results.append(
            {
                **item,
                "command": command,
                "returncode": completed.returncode,
                "elapsed_seconds": round(elapsed, 4),
                "max_rss_kib": _max_rss_kib(completed.stderr),
                "ast_path": str(ast),
                "ast_exists": ast.is_file(),
                "ast_sha256": _sha256(ast) if ast.is_file() else None,
                "dep_paths_path": str(dependencies),
                "dep_paths_exists": dependencies.is_file(),
                "dep_paths_sha256": (
                    _sha256(dependencies) if dependencies.is_file() else None
                ),
            }
        )
        (output / "trace_results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0 or not ast.is_file():
            raise RuntimeError(f"canary trace failed for {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", dest="files")
    args = parser.parse_args()
    run_canary(
        args.repo,
        args.tool_root,
        args.output,
        list(args.files or DEFAULT_FILES),
    )


if __name__ == "__main__":
    main()
