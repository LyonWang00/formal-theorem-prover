"""Trace a diverse, resumable pool of current-mathlib files with LeanDojo-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_DECLARATION = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|example)\b",
    re.MULTILINE,
)
_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_USER_TIME = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
_SYSTEM_TIME = re.compile(r"System time \(seconds\):\s*([0-9.]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(repo: Path, relative: str, suffix: str) -> Path:
    return repo / ".lake/build/ir" / Path(relative).with_suffix(suffix)


def _metric(pattern: re.Pattern[str], stderr: str, cast: type) -> Any:
    match = pattern.search(stderr)
    return cast(match.group(1)) if match else None


def select_diverse_files(
    repo: Path,
    *,
    count: int,
    seed: int,
    exclude: set[str],
    minimum_declarations: int,
) -> list[dict[str, Any]]:
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((repo / "Mathlib").rglob("*.lean")):
        relative = path.relative_to(repo).as_posix()
        if relative in exclude:
            continue
        source = path.read_text(encoding="utf-8")
        declarations = len(_DECLARATION.findall(source))
        if declarations < minimum_declarations:
            continue
        module = Path(relative).parts[1]
        by_module[module].append(
            {
                "source_file": relative,
                "top_level_module": module,
                "estimated_declarations": declarations,
                "source_bytes": path.stat().st_size,
                "source_sha256": _sha256(path),
            }
        )

    rng = random.Random(seed)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for module, rows in by_module.items():
        rng.shuffle(rows)
        rows.sort(
            key=lambda row: (
                -min(int(row["estimated_declarations"]), 100),
                int(row["source_bytes"]),
                rng.random(),
            )
        )
        queues[module] = deque(rows)
    modules = sorted(queues)
    rng.shuffle(modules)
    selected: list[dict[str, Any]] = []
    while modules and len(selected) < count:
        next_modules: list[str] = []
        for module in modules:
            queue = queues[module]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_modules.append(module)
        modules = next_modules
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} eligible source files; requested {count}"
        )
    return selected


def _trace_one(
    *,
    index: int,
    item: dict[str, Any],
    repo: Path,
    output: Path,
    timeout: int,
) -> dict[str, Any]:
    relative = str(item["source_file"])
    command = [
        "/usr/bin/time",
        "-v",
        "lake",
        "env",
        "lean",
        "--threads",
        "1",
        "--run",
        "ExtractData.lean",
        relative,
    ]
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = (error.stdout or "") if isinstance(error.stdout, str) else ""
        stderr = (error.stderr or "") if isinstance(error.stderr, str) else ""
        stderr += f"\ntrace timeout after {timeout} seconds\n"
    elapsed = time.monotonic() - started
    safe_name = relative.removesuffix(".lean").replace("/", "__")
    log_base = output / "logs" / f"{index:03d}_{safe_name}"
    log_base.parent.mkdir(parents=True, exist_ok=True)
    log_base.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
    log_base.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")

    ast = _artifact(repo, relative, ".ast.json")
    dependencies = _artifact(repo, relative, ".dep_paths")
    for source_artifact in (ast, dependencies):
        if source_artifact.is_file():
            destination = (
                output
                / "raw"
                / source_artifact.relative_to(repo / ".lake/build/ir")
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_artifact, destination)
    return {
        **item,
        "index": index,
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 4),
        "max_rss_kib": _metric(_RSS, stderr, int),
        "user_cpu_seconds": _metric(_USER_TIME, stderr, float),
        "system_cpu_seconds": _metric(_SYSTEM_TIME, stderr, float),
        "ast_path": str(ast),
        "ast_exists": ast.is_file(),
        "ast_sha256": _sha256(ast) if ast.is_file() else None,
        "dep_paths_path": str(dependencies),
        "dep_paths_exists": dependencies.is_file(),
        "dep_paths_sha256": (
            _sha256(dependencies) if dependencies.is_file() else None
        ),
    }


def run_pool(
    *,
    repo: Path,
    tool_root: Path,
    output: Path,
    file_count: int,
    workers: int,
    seed: int,
    minimum_declarations: int,
    timeout: int,
    exclude_files: set[str],
) -> None:
    repo = repo.resolve(strict=True)
    tool_root = tool_root.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    extractor_source = (
        tool_root / "lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean"
    ).resolve(strict=True)
    extractor_target = repo / "ExtractData.lean"
    if (
        extractor_target.exists()
        and _sha256(extractor_target) != _sha256(extractor_source)
    ):
        raise RuntimeError(f"unexpected extractor already exists: {extractor_target}")
    if not extractor_target.exists():
        shutil.copy2(extractor_source, extractor_target)

    selected_path = output / "selected_files.json"
    if selected_path.exists():
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = list(selected_payload["files"])
        if (
            int(selected_payload["seed"]) != seed
            or int(selected_payload["file_count"]) != file_count
        ):
            raise RuntimeError("existing selected_files.json has different settings")
    else:
        selected = select_diverse_files(
            repo,
            count=file_count,
            seed=seed,
            exclude=exclude_files,
            minimum_declarations=minimum_declarations,
        )
        selected_payload = {
            "selected_at": datetime.now(UTC).isoformat(),
            "repository": str(repo),
            "repository_commit": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "extractor": str(extractor_source),
            "extractor_sha256": _sha256(extractor_source),
            "seed": seed,
            "file_count": file_count,
            "minimum_estimated_declarations": minimum_declarations,
            "workers": workers,
            "files": selected,
        }
        selected_path.write_text(
            json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8"
        )

    results_path = output / "trace_results.json"
    results_by_file: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        results_by_file = {
            str(row["source_file"]): row
            for row in json.loads(results_path.read_text(encoding="utf-8"))
        }
    pending = [
        (index, item)
        for index, item in enumerate(selected)
        if not (
            (prior := results_by_file.get(str(item["source_file"])))
            and prior.get("returncode") == 0
            and prior.get("ast_exists")
            and prior.get("dep_paths_exists")
        )
    ]
    lock = threading.Lock()

    def save_result(row: dict[str, Any]) -> None:
        with lock:
            results_by_file[str(row["source_file"])] = row
            ordered = [
                results_by_file[str(item["source_file"])]
                for item in selected
                if str(item["source_file"]) in results_by_file
            ]
            results_path.write_text(
                json.dumps(ordered, indent=2) + "\n", encoding="utf-8"
            )
            runtime_path = output.parent / "runtime" / "tracing_resources.jsonl"
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with runtime_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "stage": "trace_pool",
                            "source_file": row["source_file"],
                            "started_at": row["started_at"],
                            "finished_at": row["finished_at"],
                            "elapsed_seconds": row["elapsed_seconds"],
                            "max_rss_kib": row["max_rss_kib"],
                            "user_cpu_seconds": row["user_cpu_seconds"],
                            "system_cpu_seconds": row["system_cpu_seconds"],
                            "returncode": row["returncode"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _trace_one,
                index=index,
                item=item,
                repo=repo,
                output=output,
                timeout=timeout,
            ): item
            for index, item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            save_result(row)
            print(
                json.dumps(
                    {
                        "source_file": row["source_file"],
                        "returncode": row["returncode"],
                        "elapsed_seconds": row["elapsed_seconds"],
                        "max_rss_kib": row["max_rss_kib"],
                    }
                ),
                flush=True,
            )

    failures = [
        row
        for row in results_by_file.values()
        if row.get("returncode") != 0
        or not row.get("ast_exists")
        or not row.get("dep_paths_exists")
    ]
    if failures:
        raise RuntimeError(f"{len(failures)} trace-pool files failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-count", type=int, default=52)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--minimum-declarations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--exclude-file", action="append", default=[])
    args = parser.parse_args()
    run_pool(
        repo=args.repo,
        tool_root=args.tool_root,
        output=args.output,
        file_count=args.file_count,
        workers=args.workers,
        seed=args.seed,
        minimum_declarations=args.minimum_declarations,
        timeout=args.timeout,
        exclude_files=set(args.exclude_file),
    )


if __name__ == "__main__":
    main()

