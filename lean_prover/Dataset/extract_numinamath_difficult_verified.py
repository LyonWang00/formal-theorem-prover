"""Extract Numina placeholder rows whose statement already elaborated cleanly.

The source fail row must show that Pantograph produced no error and only the
expected `declaration uses sorry` warning.  The final placeholder is removed;
no proof target is copied into the GRPO-oriented output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Data import GRPOData
from lean_prover.Dataset.build_verified_datasets import sha256_file
from lean_prover.Dataset.verify_external_datasets import (
    NUMINA_REVISION,
    NUMINA_SOURCE,
    select_numina_source,
    split_imports,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


_FINAL_PROOF_STUB = re.compile(r":=\s*by(?:\s+sorry)?\s*$", re.I | re.S)
_PROOF_MARKER = re.compile(r":=\s*by\b", re.I)
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)
_DIAGNOSTIC_ERROR = re.compile(r"(?:^|\n)\s*\d+:\d+(?:-\d+:\d+)?:\s*error:", re.I)
_WARNING_LINE = re.compile(r"(?:^|\n)\s*\d+:\d+(?:-\d+:\d+)?:\s*warning:\s*([^\n]+)", re.I)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def statement_only_source(row: Mapping[str, Any], selected_source: str) -> str | None:
    # Numina has two legitimate layouts: a complete source ending in
    # `:= by sorry`, or a proof-free `formal_statement` ending in `:= by`
    # paired with a separate partial proof.  Prefer the explicit statement
    # field, but never retain support declarations containing sorry/admit.
    possible_sources = (
        str(row.get("formal_statement") or "").strip(),
        str(row.get("lean_statement") or "").strip(),
        selected_source.strip(),
    )
    for possible in possible_sources:
        if not possible:
            continue
        match = _FINAL_PROOF_STUB.search(possible)
        if match is None:
            continue
        statement = possible[: match.start()].rstrip()
        if not statement or _FORBIDDEN.search(statement):
            continue
        imports, body = split_imports(statement)
        import_block = "\n".join(f"import {module}" for module in imports)
        return (import_block + "\n\n" + body.strip()).strip()
    return None


def verified_statement_only_source(row: Mapping[str, Any], selected_source: str) -> str | None:
    """Remove the final target proof from an already verified success row."""
    possible_sources = (
        str(row.get("formal_statement") or "").strip(),
        str(row.get("lean_statement") or "").strip(),
        selected_source.strip(),
    )
    for possible in possible_sources:
        if not possible:
            continue
        matches = list(_PROOF_MARKER.finditer(possible))
        statement = possible[: matches[-1].start()].rstrip() if matches else possible.rstrip()
        if not statement or _FORBIDDEN.search(statement):
            continue
        imports, body = split_imports(statement)
        if not body.strip():
            continue
        import_block = "\n".join(f"import {module}" for module in imports)
        return (import_block + "\n\n" + body.strip()).strip()
    return None


def load_manual_labels(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"manual difficulty label file not found: {path}")
    labels = list(iter_jsonl(path))
    ids = [str(row.get("record_id") or "") for row in labels]
    if any(not record_id for record_id in ids):
        raise ValueError("every manual difficulty label needs record_id")
    if len(ids) != len(set(ids)):
        raise ValueError("manual difficulty labels contain duplicate record IDs")
    return labels


def apply_manual_statement_edits(source: str, label: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    repaired = source
    applied: list[dict[str, Any]] = []
    operations = label.get("operations") or []
    if not operations:
        raise ValueError(f"manual fail label {label.get('record_id')} has no operations")
    for index, operation in enumerate(operations):
        old = str(operation.get("old") or "")
        new = str(operation.get("new") or "")
        expected = int(operation.get("expected_count", 1))
        actual = repaired.count(old)
        if not old or actual != expected:
            raise ValueError(
                f"manual fail label {label.get('record_id')} operation {index} expected "
                f"{expected} occurrence(s), found {actual}"
            )
        repaired = repaired.replace(old, new)
        applied.append({
            "kind": str(operation.get("kind") or "manual_statement_api_repair"),
            "old_sha256": sha256_text(old),
            "new_sha256": sha256_text(new),
            "expected_count": expected,
        })
    return repaired, applied


def only_sorry_warning(message: str) -> bool:
    if "forbidden proof token `sorry`" not in message:
        return False
    if _DIAGNOSTIC_ERROR.search(message):
        return False
    warnings = _WARNING_LINE.findall(message)
    return bool(warnings) and all("declaration uses `sorry`" in warning for warning in warnings)


def statement_elaborated_with_placeholder(checked: Any) -> bool:
    """Accept sorry-only diagnostics for statement-scope verification only."""
    if checked.timed_out:
        return False
    messages = list(checked.errors) + list(checked.warnings)
    if messages:
        return all(
            "warning:" in message and "declaration uses `sorry`" in message
            for message in messages
        )
    diagnostics = str(checked.diagnostics or "")
    if _DIAGNOSTIC_ERROR.search(diagnostics):
        return False
    parsed = _WARNING_LINE.findall(diagnostics)
    return bool(parsed) and all("declaration uses `sorry`" in warning for warning in parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--success-file", type=Path)
    parser.add_argument("--manual-labels-file", type=Path)
    parser.add_argument("--manual-fail-labels-file", type=Path)
    args = parser.parse_args()

    identity = environment_identity(args.lean_project, ("Mathlib",))
    output: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_statements: set[str] = set()
    scanned = 0
    for row in iter_jsonl(args.fail_file):
        scanned += 1
        original_message = str(row.get("error_message") or "")
        if not only_sorry_warning(original_message):
            exclusions["not_only_sorry_warning"] += 1
            continue
        selected_field, selected_source = select_numina_source(row)
        lean_statement = statement_only_source(row, selected_source)
        if lean_statement is None:
            exclusions["not_single_final_sorry_placeholder"] += 1
            continue
        record_id = str(row.get("record_id") or "")
        statement_hash = sha256_text(lean_statement)
        if record_id in seen_ids:
            exclusions["duplicate_record_id"] += 1
            continue
        if statement_hash in seen_statements:
            exclusions["duplicate_exact_statement"] += 1
            continue
        seen_ids.add(record_id)
        seen_statements.add(statement_hash)
        metadata = {
            "difficulty_label": "difficult",
            "difficulty_basis": "proof_missing_or_placeholder_but_statement_elaborates",
            "intended_training_stage": "grpo",
            "proof_target_included": False,
            "original_record_hash": str(row.get("record_hash") or ""),
            "original_selected_source_field": selected_field,
            "original_selected_source_sha256": sha256_text(selected_source),
            "statement_sha256": statement_hash,
            "dataset_revision": NUMINA_REVISION,
            "pantograph_diagnostic": original_message,
            "lean_version": identity["lean_version"],
            "mathlib_commit": identity["mathlib_commit"],
            "environment_hash": identity["environment_hash"],
            "assembler_version": identity["assembler_version"],
            "normalization_version": identity["normalization_version"],
        }
        grpo = GRPOData(
            record_id=record_id,
            lean_statement=lean_statement,
            source=NUMINA_SOURCE,
            metadata=metadata,
        )
        output.append(grpo.model_dump(mode="json"))

    manual_labels = load_manual_labels(args.manual_labels_file)
    manual_added = 0
    if manual_labels:
        if args.success_file is None:
            raise ValueError("--success-file is required with --manual-labels-file")
        requested_ids = {str(label["record_id"]) for label in manual_labels}
        success_rows = {
            str(row.get("record_id") or ""): row
            for row in iter_jsonl(args.success_file)
            if str(row.get("record_id") or "") in requested_ids
        }
        missing = sorted(requested_ids - set(success_rows))
        if missing:
            raise ValueError(f"manual difficult records absent from success file: {missing[:5]}")
        for label in manual_labels:
            record_id = str(label["record_id"])
            row = success_rows[record_id]
            expected_hash = str(label.get("original_record_hash") or "")
            actual_hash = str(row.get("record_hash") or "")
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(f"record hash mismatch for manually difficult row {record_id}")
            selected_field, selected_source = select_numina_source(row)
            lean_statement = verified_statement_only_source(row, selected_source)
            if lean_statement is None:
                raise ValueError(f"cannot extract theorem-only view for {record_id}")
            statement_hash = sha256_text(lean_statement)
            if record_id in seen_ids or statement_hash in seen_statements:
                exclusions["manual_duplicate"] += 1
                continue
            seen_ids.add(record_id)
            seen_statements.add(statement_hash)
            proof = str(row.get("proof") or "")
            metadata = {
                "difficulty_label": "difficult",
                "difficulty_basis": str(label.get("difficulty_basis") or "codex_manual_review"),
                "difficulty_notes": str(label.get("difficulty_notes") or ""),
                "difficulty_reviewer": str(label.get("difficulty_reviewer") or "codex_manual"),
                "intended_training_stage": "grpo",
                "proof_target_included": False,
                "original_split": "proof_verified_success",
                "original_record_hash": actual_hash,
                "original_selected_source_field": selected_field,
                "original_selected_source_sha256": sha256_text(selected_source),
                "original_proof_sha256": sha256_text(proof) if proof else None,
                "original_proof_character_length": len(proof),
                "statement_sha256": statement_hash,
                "dataset_revision": NUMINA_REVISION,
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": identity["environment_hash"],
                "assembler_version": identity["assembler_version"],
                "normalization_version": identity["normalization_version"],
            }
            grpo = GRPOData(
                record_id=record_id,
                lean_statement=lean_statement,
                source=NUMINA_SOURCE,
                metadata=metadata,
            )
            output.append(grpo.model_dump(mode="json"))
            manual_added += 1

    manual_fail_labels = load_manual_labels(args.manual_fail_labels_file)
    manual_fail_added = 0
    manual_fail_rejected: list[dict[str, Any]] = []
    if manual_fail_labels:
        requested_ids = {str(label["record_id"]) for label in manual_fail_labels}
        fail_rows = {
            str(row.get("record_id") or ""): row
            for row in iter_jsonl(args.fail_file)
            if str(row.get("record_id") or "") in requested_ids
        }
        missing = sorted(requested_ids - set(fail_rows))
        if missing:
            raise ValueError(f"manual difficult records absent from fail file: {missing[:5]}")
        verifier = PantographTheoremVerifier(
            args.lean_project,
            imports=("Mathlib",),
            timeout=30,
            startup_timeout=900,
        )
        warmup = verifier.warmup(timeout=180)
        if not warmup.success:
            verifier.close()
            raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
        try:
            for label in manual_fail_labels:
                record_id = str(label["record_id"])
                row = fail_rows[record_id]
                expected_hash = str(label.get("original_record_hash") or "")
                actual_hash = str(row.get("record_hash") or "")
                if expected_hash and expected_hash != actual_hash:
                    raise ValueError(f"record hash mismatch for manual fail difficult row {record_id}")
                selected_field, selected_source = select_numina_source(row)
                repaired_source, operations = apply_manual_statement_edits(selected_source, label)
                lean_statement = statement_only_source({}, repaired_source)
                if lean_statement is None:
                    raise ValueError(f"cannot extract repaired theorem-only view for {record_id}")
                _, body = split_imports(lean_statement)
                checked = verifier.check_source(body.rstrip() + "\n:= by sorry", timeout=30, reject_forbidden=False)
                if not checked.success and not statement_elaborated_with_placeholder(checked):
                    manual_fail_rejected.append({
                        "record_id": record_id,
                        "error_type": checked.error_type,
                        "diagnostics": checked.diagnostics,
                        "errors": list(checked.errors),
                        "warnings": list(checked.warnings),
                    })
                    continue
                statement_hash = sha256_text(lean_statement)
                if record_id in seen_ids or statement_hash in seen_statements:
                    exclusions["manual_fail_duplicate"] += 1
                    continue
                seen_ids.add(record_id)
                seen_statements.add(statement_hash)
                metadata = {
                    "difficulty_label": "difficult",
                    "difficulty_basis": str(label.get("difficulty_basis") or "codex_manual_placeholder_review"),
                    "difficulty_notes": str(label.get("difficulty_notes") or ""),
                    "difficulty_reviewer": str(label.get("difficulty_reviewer") or "codex_manual"),
                    "intended_training_stage": "grpo",
                    "proof_target_included": False,
                    "original_split": "proof_missing_or_placeholder",
                    "statement_repaired_before_verification": True,
                    "statement_repair_operations": operations,
                    "original_record_hash": actual_hash,
                    "original_selected_source_field": selected_field,
                    "original_selected_source_sha256": sha256_text(selected_source),
                    "statement_sha256": statement_hash,
                    "statement_verification_diagnostics": checked.diagnostics,
                    "statement_verification_warnings": list(checked.warnings),
                    "dataset_revision": NUMINA_REVISION,
                    "lean_version": identity["lean_version"],
                    "mathlib_commit": identity["mathlib_commit"],
                    "environment_hash": identity["environment_hash"],
                    "assembler_version": identity["assembler_version"],
                    "normalization_version": identity["normalization_version"],
                }
                grpo = GRPOData(
                    record_id=record_id,
                    lean_statement=lean_statement,
                    source=NUMINA_SOURCE,
                    metadata=metadata,
                )
                output.append(grpo.model_dump(mode="json"))
                manual_fail_added += 1
        finally:
            verifier.close()

    write_jsonl(args.output_file, output)
    report = {
        "schema_version": "numinamath_difficult_verified_report_v1",
        "input_fail_rows": scanned,
        "output_rows": len(output),
        "pantograph_verified": "success",
        "verification_scope": "statement_only",
        "proof_target_included": False,
        "intended_training_stage": "grpo",
        "selection_rule": "existing Pantograph diagnostic contains no error and only declaration-uses-sorry warning; one final placeholder removed",
        "manual_verified_hard_rows_added": manual_added,
        "manual_repaired_statement_hard_rows_added": manual_fail_added,
        "manual_repaired_statement_hard_rows_rejected": manual_fail_rejected,
        "manual_labels_file": str(args.manual_labels_file) if args.manual_labels_file else None,
        "manual_fail_labels_file": str(args.manual_fail_labels_file) if args.manual_fail_labels_file else None,
        "success_file": str(args.success_file) if args.success_file else None,
        "exclusions": dict(exclusions),
        "output_sha256": sha256_file(args.output_file),
        "environment": identity,
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
