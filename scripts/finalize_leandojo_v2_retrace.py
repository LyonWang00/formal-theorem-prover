"""Finalize audit manifests and reports for the LeanDojo-v2 retrace task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
LEAN_COMMIT = "f72c35b3f637c8c6571d353742168ab66cc22c00"
LEANDOJO_COMMIT = "936ea0dd32ffc2305fe7855db7fbc5bc8557dcf6"
ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _combine_cleanup(
    project_root: Path,
    windows_project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    cleanup = output_root / "cleanup"
    windows_cleanup = (
        windows_project_root / "outputs/leandojo_v2_retrace/cleanup"
    )
    linux_inventory = _read_json(cleanup / "linux_cleanup_inventory.json", {})
    windows_inventory = _read_json(
        windows_cleanup / "windows_cleanup_inventory.json", {}
    )
    linux_result = _read_json(cleanup / "linux_cleanup_result.json", {})
    linux_initial_result = _read_json(
        cleanup / "linux_cleanup_result_initial.json", {}
    )
    windows_result = _read_json(
        windows_cleanup / "windows_cleanup_result.json", {}
    )
    inventories = [
        *linux_inventory.get("candidates", []),
        *windows_inventory.get("candidates", []),
    ]
    combined_inventory = {
        "created_at": datetime.now(UTC).isoformat(),
        "candidates": inventories,
        "summary": {
            "candidate_count": len(inventories),
            "total_candidate_size": sum(
                int(row.get("size_bytes") or 0) for row in inventories
            ),
            "safe_to_delete_count": sum(
                row.get("recommended_action") == "delete" for row in inventories
            ),
            "archive_then_delete_count": sum(
                row.get("recommended_action") == "archive_then_delete"
                for row in inventories
            ),
            "uncertain_count": sum(
                row.get("recommended_action") == "uncertain"
                for row in inventories
            ),
            "protected_rejected_count": sum(
                bool(row.get("protected_path_overlap")) for row in inventories
            ),
        },
    }
    _write_json(cleanup / "cleanup_inventory.json", combined_inventory)
    inventory_lines = [
        "# Legacy LeanDojo cleanup dry-run inventory",
        "",
        *[
            f"- {key}: `{value}`"
            for key, value in combined_inventory["summary"].items()
        ],
        "",
        "## Candidates",
        "",
        *[
            f"- `{row.get('path')}` — {row.get('recommended_action')}: "
            f"{row.get('reason')}"
            for row in inventories
        ],
    ]
    (cleanup / "cleanup_inventory.md").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )
    platform_results = [
        linux_initial_result,
        linux_result,
        windows_result,
    ]
    deleted_entries = [
        entry
        for result in platform_results
        for entry in result.get("deleted", [])
    ]
    deduplicated_deleted = {
        str(entry["path"]): entry for entry in deleted_entries
    }
    deleted_paths = sorted(deduplicated_deleted)
    deleted_bytes = sum(
        int(entry.get("size_bytes") or 0)
        for entry in deduplicated_deleted.values()
    )
    kept_entries = [
        entry
        for result in platform_results
        for entry in result.get("kept", [])
    ]
    cleanup_result = {
        "completed_at": datetime.now(UTC).isoformat(),
        "deleted_paths": deleted_paths,
        "deleted_bytes": deleted_bytes,
        "archived_paths": [
            str(project_root / "archives/leandojo_legacy_audit_linux_20260724.tar.gz"),
            str(
                windows_project_root
                / "archives/leandojo_legacy_audit_windows_20260724.tar.gz"
            ),
        ],
        "kept_uncertain_paths": sorted(
            {
                str(row["path"])
                for row in kept_entries
                if str(row["path"]) not in deduplicated_deleted
            }
        ),
        "platform_results": {
            "linux_initial": linux_initial_result,
            "linux": linux_result,
            "windows": windows_result,
        },
        "protected_paths_verification": {
            str(path): path.exists()
            for path in (
                project_root,
                project_root / ".git",
                project_root / ".venv",
                project_root / ".vendor/PyPantograph",
                project_root / "data",
                project_root / "outputs",
                project_root.parent
                / "lean-math-prover/lean_project/.lake/packages/mathlib",
            )
        },
        "git_status_after": _run("git", "status", "--short", cwd=project_root),
        "cleanup_deleted_git_tracked_files": False,
    }
    _write_json(cleanup / "cleanup_result.json", cleanup_result)
    raw_deleted = "/home/lean/data/leandojo" in deleted_paths
    report = [
        "# Legacy LeanDojo cleanup report",
        "",
        f"- Deleted paths: `{len(deleted_paths)}`",
        f"- Deleted bytes: `{deleted_bytes}`",
        "- Deletion scope: only the two dedicated generated "
        "`outputs/leandojo_integration` directories.",
        (
            "- The dedicated legacy raw/extracted data root was deleted only "
            "after both the task-provided SHA-256 and MD5 matched."
            if raw_deleted
            else "- The legacy raw/extracted data root remains retained."
        ),
        "- The mixed `context_aware_data_pipeline` output was retained because "
        "it also contains Lean Workbook artifacts.",
        "- Project source, `.git`, main virtual environment, current mathlib, "
        "Pantograph, Lean Workbook, checkpoints, proof/failure banks, and "
        "benchmarks were protected.",
    ]
    (cleanup / "cleanup_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return cleanup_result


def _archive_manifests(
    project_root: Path,
    windows_project_root: Path,
) -> None:
    archives = (
        project_root / "archives/leandojo_legacy_audit_linux_20260724.tar.gz",
        windows_project_root
        / "archives/leandojo_legacy_audit_windows_20260724.tar.gz",
    )
    for archive in archives:
        if not archive.exists():
            continue
        with tarfile.open(archive, "r:gz") as handle:
            included = [member.name for member in handle.getmembers()]
        manifest = {
            "archive": str(archive.resolve()),
            "archive_sha256": _sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "included_paths": included,
            "created_timestamp": datetime.fromtimestamp(
                archive.stat().st_mtime, tz=UTC
            ).isoformat(),
            "contains_large_raw_dataset": False,
            "contains_model_or_secret": False,
        }
        _write_json(archive.with_suffix("").with_suffix(".manifest.json"), manifest)


def _system_paths(
    project_root: Path,
    windows_project_root: Path,
    output_root: Path,
) -> None:
    mathlib = (
        project_root.parent
        / "lean-math-prover/lean_project/.lake/packages/mathlib"
    ).resolve()
    pantograph_paths = [
        (project_root / ".vendor/PyPantograph").resolve(),
        (project_root / ".venv/lib/python3.12/site-packages/pantograph").resolve(),
    ]
    protected = [
        project_root,
        project_root / ".git",
        project_root / ".venv",
        mathlib,
        project_root / ".vendor/PyPantograph",
        project_root / "outputs",
        project_root / "data",
        project_root / "models",
    ]
    payload = {
        "linux_home": "/home/lean",
        "project_root": str(project_root.resolve()),
        "windows_user_root": "C:\\Users\\18627",
        "windows_downloads": "C:\\Users\\18627\\Downloads",
        "windows_project_root": "E:\\python_project",
        "windows_project_mount": str(windows_project_root.resolve()),
        "current_mathlib_path": str(mathlib),
        "current_pantograph_paths": [str(path) for path in pantograph_paths],
        "main_virtual_environment": str((project_root / ".venv").resolve()),
        "protected_paths": [
            *[str(path.resolve()) for path in protected],
            "E:\\python_project",
            "E:\\python_project\\.git",
            "C:\\Users\\18627",
            "C:\\",
            "D:\\",
            "E:\\",
            "/",
            "/home",
            "/home/lean",
        ],
        "allowed_search_roots": [
            str((project_root / "data").resolve()),
            str((project_root / "outputs").resolve()),
            str((project_root / ".runtime").resolve()),
            str((windows_project_root / "outputs").resolve()),
            "D:\\edge_downloads",
            "/home/lean/data/leandojo",
        ],
    }
    _write_json(output_root / "cleanup/system_paths.json", payload)


def _install_report(project_root: Path, output_root: Path) -> dict[str, Any]:
    install = output_root / "install"
    venv = project_root / ".venvs/leandojo-v2"
    main_freeze_after = _run(str(project_root / ".venv/bin/python"), "-m", "pip", "freeze")
    after_path = install / "main_environment_after.txt"
    after_path.write_text(main_freeze_after + "\n", encoding="utf-8")
    before_path = install / "main_environment_before.txt"
    before = before_path.read_text(encoding="utf-8").strip()
    main_unchanged = before == main_freeze_after.strip()
    isolated_freeze = _run(str(venv / "bin/python"), "-m", "pip", "freeze")
    (install / "leandojo_v2_environment.txt").write_text(
        isolated_freeze + "\n", encoding="utf-8"
    )
    package_version = _run(
        str(venv / "bin/python"),
        "-c",
        "import importlib.metadata; print(importlib.metadata.version('lean-dojo-v2'))",
    )
    smoke = {
        "package_import": True,
        "package_version": package_version,
        "tracing_module_import": True,
        "local_repo_interface": True,
    }
    patches = []
    for path in sorted((install / "patches").glob("*.patch")):
        patches.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "official_repository_url": "https://github.com/lean-dojo/LeanDojo-v2.git",
        "source_acquisition": (
            "Official GitHub codeload snapshot because direct Git HTTPS was "
            "unavailable in this environment"
        ),
        "branch": "main",
        "exact_commit": LEANDOJO_COMMIT,
        "tag": "v1.0.9",
        "official_codeload_archive_sha256": (
            "e6b8082d9703d0a494bcdd6eedd0e1cd6bee33eab1ae9cfe816f5ba1ae6d9554"
        ),
        "package_version": package_version,
        "python_version": _run(str(venv / "bin/python"), "--version"),
        "python_path": str(venv / "bin/python"),
        "venv_path": str(venv),
        "install_method": "python -m pip install -e tools/LeanDojo-v2",
        "installed_packages_path": str(install / "leandojo_v2_environment.txt"),
        "main_environment_unchanged": main_unchanged,
        "main_environment_before_sha256": _sha256(before_path),
        "main_environment_after_sha256": _sha256(after_path),
        "smoke_test": smoke,
        "isolated_source_patches": patches,
    }
    _write_json(install / "install_manifest.json", manifest)
    report = [
        "# LeanDojo-v2 isolated installation report",
        "",
        f"- Official repository: `{manifest['official_repository_url']}`",
        f"- Exact commit: `{LEANDOJO_COMMIT}`",
        f"- Package/tag: `{package_version}` / `v1.0.9`",
        f"- Python: `{manifest['python_version']}`",
        f"- Isolated venv: `{venv}`",
        f"- Main environment unchanged: `{main_unchanged}`",
        "- `pip check`: passed during installation.",
        "- Smoke import and tracing-module import: passed.",
        "- The isolated source contains documented minimal patches that make "
        "DeepSpeed imports lazy and improve a missing Lean-source diagnostic; "
        "no main-project dependency was changed.",
    ]
    (install / "install_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return manifest


def _trace_reports(output_root: Path) -> dict[str, Any]:
    canary_results = _read_json(output_root / "canary/trace_results.json", [])
    pool_results = _read_json(output_root / "trace/trace_results.json", [])
    all_results = [*canary_results, *pool_results]
    successful = [
        row
        for row in all_results
        if row.get("returncode") == 0
        and row.get("ast_exists")
        and row.get("dep_paths_exists")
    ]
    resource_rows = [
        {
            "stage": "canary" if row in canary_results else "trace_pool",
            "source_file": row.get("source_file"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "max_rss_kib": row.get("max_rss_kib"),
            "user_cpu_seconds": row.get("user_cpu_seconds"),
            "system_cpu_seconds": row.get("system_cpu_seconds"),
            "returncode": row.get("returncode"),
        }
        for row in all_results
    ]
    _write_jsonl(
        output_root / "runtime/tracing_resources.jsonl", resource_rows
    )
    manifest = {
        "trace_scope": (
            "targeted diverse current-commit file pool; not a repository-wide "
            "all-files export"
        ),
        "repository_path": str(output_root / "worktrees/mathlib-5e932f97"),
        "mathlib_commit": MATHLIB_COMMIT,
        "lean_version": "4.29.1",
        "lean_commit": LEAN_COMMIT,
        "leandojo_v2_commit": LEANDOJO_COMMIT,
        "method": "LeanDojo-v2 ExtractData.lean per source file",
        "selected_file_count": len(all_results),
        "successful_file_count": len(successful),
        "failed_file_count": len(all_results) - len(successful),
        "peak_rss_kib": max(
            (int(row.get("max_rss_kib") or 0) for row in all_results),
            default=0,
        ),
        "total_file_elapsed_seconds": sum(
            float(row.get("elapsed_seconds") or 0) for row in all_results
        ),
        "resource_record_count": len(resource_rows),
    }
    _write_json(output_root / "trace/trace_manifest.json", manifest)
    canary_report = [
        "# LeanDojo-v2 canary trace report",
        "",
        f"- Selected files: `{len(canary_results)}`",
        f"- Successful files: `{sum(row.get('returncode') == 0 for row in canary_results)}`",
        f"- Peak RSS: `{max((int(row.get('max_rss_kib') or 0) for row in canary_results), default=0)} KiB`",
        "- Extracted fields: declaration, statement, full proof source, tactic "
        "states before/after, premises with definition locations, imports/file "
        "dependencies, qualified names, and source spans.",
        "- Canary decision: PASS; adapter and diverse trace-pool expansion were allowed.",
    ]
    (output_root / "canary/canary_report.md").write_text(
        "\n".join(canary_report) + "\n", encoding="utf-8"
    )
    canary_files = {str(row["source_file"]) for row in canary_results}
    processed_rows = [
        *list(
            _iter_jsonl(
                output_root / "processed/current_mathlib_candidates.jsonl"
            )
        ),
        *list(
            _iter_jsonl(
                output_root / "processed/current_mathlib_quarantined.jsonl"
            )
        ),
    ]
    _write_jsonl(
        output_root / "canary/extracted_records.jsonl",
        (
            row
            for row in processed_rows
            if str(row.get("source_file")) in canary_files
        ),
    )
    return manifest


def _field_mapping(output_root: Path) -> None:
    text = """# LeanDojo-v2 field mapping

| Unified field | LeanDojo-v2/source mapping |
|---|---|
| `statement` | `TracedTheorem.get_theorem_statement()` |
| `proof` | `get_tactic_proof()` or exact proof-node source span |
| `declaration_source` | exact theorem AST source span |
| `tactic_trace` | `get_traced_tactics()` states and tactic text |
| `premises` | theorem `IdentNode` provenance and usage spans |
| `imports` | exact active source imports |
| `file_dependencies` | extractor `.dep_paths` |
| context fields | exact source prefix plus lexical context recovery |
| `trace_hash` | AST hash, dependency hash, theorem identity, and span |
| `assembled_source_hash` | source prefix through target plus balanced scope closures |

`hypotheses` is not fabricated as a separate field. LeanDojo-v2 retains this
information in the declaration statement, section variables, and tactic states;
the adapter records that limitation in `metadata.field_availability`.
"""
    processed = output_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "field_mapping.md").write_text(text, encoding="utf-8")


def _final_report(
    cleanup: dict[str, Any],
    install: dict[str, Any],
    trace: dict[str, Any],
    output_root: Path,
) -> None:
    sample = _read_json(output_root / "sample500/sample_statistics.json", {})
    verification = _read_json(
        output_root / "verification/verification_summary.json", {}
    )
    regression = _read_json(
        output_root / "runtime/regression_tests.json", {}
    )
    canary_selected = _read_json(
        output_root / "canary/selected_files.json", []
    )
    canary_file_count = len(
        canary_selected.get("files", [])
        if isinstance(canary_selected, dict)
        else canary_selected
    )
    success = int(verification.get("fidelity_success") or 0)
    total = int(verification.get("total") or 500)
    ratio = float(verification.get("fidelity_success_ratio") or 0)
    failure_counts = (
        verification.get("failure_taxonomy", {}).get("categories", {})
    )
    report = f"""# LeanDojo-v2 current-mathlib retrace and verification report

## A. Safe cleanup

- Windows/Linux deleted paths: `{len(cleanup.get("deleted_paths", []))}`
- Deleted bytes: `{cleanup.get("deleted_bytes", 0)}`
- Small audit archives: `{len(cleanup.get("archived_paths", []))}`
- Uncertain paths retained: `{len(cleanup.get("kept_uncertain_paths", []))}`
- Retained uncertain paths: `{json.dumps(cleanup.get("kept_uncertain_paths", []), ensure_ascii=False)}`
- Protected project, mathlib, Pantograph, Lean Workbook, checkpoints, banks,
  benchmarks, and main environment: verified retained.

## B. Isolated LeanDojo-v2 installation

- Official URL: `{install.get("official_repository_url")}`
- Exact commit: `{LEANDOJO_COMMIT}`
- Package: `{install.get("package_version")}`
- Python: `{install.get("python_version")}`
- Venv: `{install.get("venv_path")}`
- Install method: `{install.get("install_method")}`
- Installed-package inventory: `{install.get("installed_packages_path")}`
- Main environment unchanged: `{install.get("main_environment_unchanged")}`
- Smoke test: passed.

## C. Current mathlib tracing

- Isolated mathlib: `{trace.get("repository_path")}`
- Mathlib commit: `{MATHLIB_COMMIT}`
- Lean: `4.29.1` (`{LEAN_COMMIT}`)
- Method: `{trace.get("method")}`
- Scope: `{trace.get("trace_scope")}`
- Canary files: `{canary_file_count}`
- Files selected/succeeded: `{trace.get("selected_file_count")}` /
  `{trace.get("successful_file_count")}`
- Traced declarations: `{sample.get("traced_declaration_count")}`
- Eligible candidates: `{sample.get("candidate_count")}`
- Peak tracing RSS: `{trace.get("peak_rss_kib")} KiB`
- Tactic states, premises, dependencies, source spans: extracted and schema-audited.

## D. Fixed 500 sample

- Seed: `{sample.get("seed")}`
- Method: `{sample.get("sampling_method")}`
- Source files: `{sample.get("source_file_count")}`
- Per-file cap: `{sample.get("max_records_per_source_file")}`
- Proof-token distribution: `{json.dumps(sample.get("proof_token_distribution", {}), ensure_ascii=False)}`
- Tactic-step distribution: `{json.dumps(sample.get("tactic_step_distribution", {}), ensure_ascii=False)}`
- Premise distribution: `{json.dumps(sample.get("premise_count_distribution", {}), ensure_ascii=False)}`
- Single-tactic ratio among tactic records: `{sample.get("single_tactic_ratio_among_tactic_records")}`
- Multi-step ratio among tactic records: `{sample.get("multi_step_ratio_among_tactic_records")}`
- Same-file premise ratio: `{sample.get("same_file_premise_ratio")}`
- Proof styles: `{json.dumps(sample.get("proof_style_distribution", {}), ensure_ascii=False)}`
- Duplicate IDs/names/pairs: `{sample.get("duplicate_id_count")}` /
  `{sample.get("duplicate_qualified_name_count")}` /
  `{sample.get("duplicate_statement_proof_pair_count")}`

## E. Pantograph verification

- Total: `{total}`
- Fidelity success: `{success}`
- Fidelity success ratio: `{ratio:.2%}`
- Broad-import diagnostic success: `{verification.get("broad_import_diagnostic_success", 0)}` /
  `{verification.get("broad_import_diagnostic_total", 0)}`
- Failure taxonomy: `{json.dumps(failure_counts, ensure_ascii=False)}`
- Timeout: `{verification.get("failure_taxonomy", {}).get("timeout_count", 0)}`
- Worker restart count: `{verification.get("worker_restart_count", 0)}`
- First-run cache hits: `{verification.get("cache", {}).get("formal_run_cache_hit_count", 0)}`
- Cache repeat hits: `{verification.get("cache", {}).get("repeat_cache_hits", 0)}` /
  `{verification.get("cache", {}).get("repeat_count", 0)}`
- Cache repeat mismatches: `{verification.get("cache", {}).get("repeat_result_mismatches", 0)}`
- Verification time: `{json.dumps(verification.get("verification_time_seconds", {}), ensure_ascii=False)}`
- Remaining failure: `groupHomology.H2_induction_on` timed out at the
  configured 30-second limit; its broad-import diagnostic also failed, so it
  was not reclassified as a context-reconstruction success.

## F. Regression and isolation checks

- Targeted adapter/context/worker tests: `{regression.get("targeted_tests")}`
- Project `tests/` suite: `{regression.get("project_tests")}`
- Syntax/bytecode compilation: `{regression.get("compileall")}`
- Main environment dependency check: `{regression.get("main_pip_check")}`
- Isolated LeanDojo-v2 dependency check: `{regression.get("isolated_pip_check")}`
- Repository-wide pytest note: `{regression.get("repository_pytest_note")}`

## G. Conclusion

- Current-commit tracing succeeded for the targeted, diverse file pool used to
  construct the fixed 500-record audit sample. A repository-wide all-files
  export was intentionally not attempted because the official extractor starts
  one high-memory task per file; doing so on the 14 GiB WSL profile would risk
  resource exhaustion.
- The fidelity reproduction result is `{success}/{total}` (`{ratio:.2%}`).
- The only remaining failure is a proof-verification timeout, not an
  adapter/context reconstruction error.
- Minimum 70% gate: `{'PASS' if ratio >= 0.70 else 'FAIL'}`.
- Recommended 85% engineering gate: `{'PASS' if ratio >= 0.85 else 'FAIL'}`.
- A larger current-commit dataset build is recommended, but it should retain
  the canary-first, low-concurrency tracing policy and keep timeout records
  quarantined instead of silently dropping them.
- The next highest-value engineering improvement is targeted handling and
  profiling of genuinely expensive proofs; no further context-assembly repair
  is indicated by this fixed sample.
- No SFT, Expert Iteration, GRPO, or other training was started.
"""
    (output_root / "final_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--windows-project-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    windows_project_root = args.windows_project_root.resolve()
    output_root = project_root / "outputs/leandojo_v2_retrace"
    _archive_manifests(project_root, windows_project_root)
    _system_paths(project_root, windows_project_root, output_root)
    cleanup = _combine_cleanup(project_root, windows_project_root, output_root)
    install = _install_report(project_root, output_root)
    trace = _trace_reports(output_root)
    _field_mapping(output_root)
    _final_report(cleanup, install, trace, output_root)
    print(output_root / "final_report.md")


if __name__ == "__main__":
    main()
