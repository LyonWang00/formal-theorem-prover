#!/usr/bin/env python3
"""Rewrite and Pantograph-check all 488 miniF2F Lean statements."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from lean_prover.Dataset.minif2f_compat import (
    MAPPING_VERSION,
    MINIF2F_IMPORT_COMPATIBILITY,
    STATEMENT_COMPATIBILITY_VERSION,
    mapped_imports,
    rewrite_header,
    rewrite_statement,
    validate_mapped_modules_exist,
)
from lean_prover.Planner.pantograph_checker import (
    PantographDeclarationCheckingBackend,
)
from lean_prover.Planner.schemas import LeanPreamble
from lean_prover.Planner.validator import normalize_lean_target_input


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_rev(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _failure_category(error_message: str) -> str:
    if "unexpected token 'in'" in error_message:
        return "deleted_big_operator_in_syntax"
    if "unexpected end of input" in error_message:
        return "commented_out_no_declaration"
    if "Complex.abs" in error_message:
        return "removed_complex_abs"
    if "Ambiguous term" in error_message and "lcm" in error_message:
        return "ambiguous_lcm"
    return "other"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repo / "lean_prover/Dataset/raw_data/minif2f_raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "lean_prover/Dataset/verified_data",
    )
    parser.add_argument(
        "--project-path", type=Path, default=repo / "lean_project"
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    mathlib_root = args.project_path / ".lake/packages/mathlib"
    validate_mapped_modules_exist(mathlib_root)

    all_rows = {
        split: _read_jsonl(args.raw_dir / f"{split}.jsonl")
        for split in ("test", "valid")
    }
    if args.limit is None:
        counts = {split: len(rows) for split, rows in all_rows.items()}
        if counts != {"test": 244, "valid": 244}:
            raise ValueError(f"expected 244+244 miniF2F rows, found {counts}")

    successes: dict[str, list[dict[str, Any]]] = {"test": [], "valid": []}
    failures: list[dict[str, Any]] = []
    checked = 0
    header_hashes: Counter[str] = Counter()
    checker = PantographDeclarationCheckingBackend(
        project_path=args.project_path,
        timeout=args.timeout,
    )
    try:
        for split in ("test", "valid"):
            for index, source_row in enumerate(all_rows[split]):
                if args.limit is not None and checked >= args.limit:
                    break
                checked += 1
                original_header = str(source_row.get("header") or "")
                compatible_header = rewrite_header(original_header)
                imports = mapped_imports(original_header)
                header_hashes[_sha256(original_header)] += 1
                try:
                    legacy_statement = str(
                        source_row.get("formal_statement") or ""
                    )
                    compatible_statement, statement_repairs = (
                        rewrite_statement(legacy_statement)
                    )
                    statement = normalize_lean_target_input(
                        compatible_statement
                    )
                    result = checker.check_declaration(
                        imports=imports,
                        declaration=statement,
                        preceding_declarations=[],
                        preamble=LeanPreamble(
                            imports=imports,
                            raw_header=compatible_header,
                        ),
                    )
                except Exception as error:
                    result = None
                    error_message = f"{type(error).__name__}: {error}"
                else:
                    error_message = result.error_message or result.stderr

                if result is None or not result.success:
                    failures.append(
                        {
                            "id": source_row.get("id"),
                            "split": split,
                            "source_index": index,
                            "failure_category": _failure_category(
                                error_message
                            ),
                            "error_message": error_message,
                        }
                    )
                    print(
                        f"FAIL {checked:03d} {split}/{source_row.get('id')}: "
                        f"{error_message}",
                        flush=True,
                    )
                    continue

                row = dict(source_row)
                row.update(
                    {
                        "formal_statement": statement,
                        "legacy_formal_statement": legacy_statement,
                        "header": compatible_header,
                        "legacy_header": original_header,
                        "pantograph_verified": "success",
                        "mathlib_import_mapping_version": MAPPING_VERSION,
                        "mathlib_statement_compatibility_version": (
                            STATEMENT_COMPATIBILITY_VERSION
                        ),
                        "statement_compatibility_repairs": statement_repairs,
                        "statement_sha256": _sha256(statement),
                        "header_sha256": _sha256(compatible_header),
                    }
                )
                successes[split].append(row)
                print(
                    f"PASS {checked:03d} {split}/{source_row.get('id')}",
                    flush=True,
                )
            if args.limit is not None and checked >= args.limit:
                break
    finally:
        checker.close()

    suffix = "" if args.limit is None else f"_limit{args.limit}"
    output_paths = {
        split: args.output_dir / f"miniF2F_{split}_verified{suffix}.jsonl"
        for split in ("test", "valid")
    }
    for split, path in output_paths.items():
        _write_jsonl(path, successes[split])
    _write_jsonl(
        args.output_dir / f"miniF2F_verification_failures{suffix}.jsonl",
        failures,
    )

    failure_categories = Counter(
        row["failure_category"] for row in failures
    )
    total_success = sum(len(value) for value in successes.values())
    report = {
        "schema_version": "minif2f-pantograph-verification-v1",
        "mapping_version": MAPPING_VERSION,
        "statement_compatibility_version": STATEMENT_COMPATIBILITY_VERSION,
        "mapping": {
            key: list(value)
            for key, value in MINIF2F_IMPORT_COMPATIBILITY.items()
        },
        "project_path": str(args.project_path.resolve()),
        "mathlib_commit": _git_rev(mathlib_root),
        "input_counts": {key: len(value) for key, value in all_rows.items()},
        "checked": checked,
        "success_counts": {key: len(value) for key, value in successes.items()},
        "total_success_count": total_success,
        "verification_rate": total_success / checked if checked else 0.0,
        "failure_count": len(failures),
        "failure_categories": dict(sorted(failure_categories.items())),
        "unique_legacy_header_hashes": dict(header_hashes),
        "outputs": {
            key: {
                "path": str(value.resolve()),
                "sha256": _sha256(value.read_text(encoding="utf-8")),
                "rows": len(successes[key]),
            }
            for key, value in output_paths.items()
        },
    }
    report_path = args.output_dir / f"miniF2F_verification_report{suffix}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
