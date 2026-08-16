"""Package explicit Codex-reviewed Numina repairs for Pantograph verification.

This module deliberately performs no repair inference.  Every edit must name
one record, its current hash, the exact old/new text, and a human rationale.
Python is used only to enforce hashes/schema, apply those explicit edits, and
reject overlap with frozen cloud batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.build_verified_datasets import sha256_file
from lean_prover.Dataset.repair_numinamath_failures import (
    REPAIR_VERSION,
    canonical_hash,
    iter_jsonl,
    write_jsonl,
)
from lean_prover.Dataset.verify_external_datasets import select_numina_source, split_imports


_PROOF_MARKER = re.compile(r":=\s*by\b")
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def frozen_record_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"frozen manifest not found: {path}")
        result.update(str(row["record_id"]) for row in read_jsonl(path))
    return result


def split_source_and_proof(source: str) -> tuple[list[str], str, str]:
    imports, body = split_imports(source)
    marker = _PROOF_MARKER.search(body)
    if marker is None:
        raise ValueError("selected source has no `:= by` proof marker")
    by_start = body.find("by", marker.start())
    source_body = body[: by_start + 2].rstrip()
    proof = body[by_start:].strip()
    if not proof.startswith("by") or not proof.removeprefix("by").strip():
        raise ValueError("manual candidate must contain a nonempty proof")
    return list(imports), source_body, proof


def apply_reviewed_edits(source: str, edit: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    operations = edit.get("operations") or []
    replacement_proof = str(edit.get("replacement_proof") or "").strip()
    repaired = source
    changes: list[dict[str, Any]] = []
    if replacement_proof:
        marker = _PROOF_MARKER.search(repaired)
        if marker is None:
            raise ValueError("selected source has no target `:= by` proof marker")
        by_start = repaired.find("by", marker.start())
        original_proof = repaired[by_start:].strip()
        if not replacement_proof.startswith("by") or _FORBIDDEN.search(replacement_proof):
            raise ValueError("replacement_proof must be a complete forbidden-token-free `by` proof")
        repaired = repaired[:by_start] + replacement_proof
        changes.append({
            "kind": "codex_manual_replace_entire_target_proof",
            "old_sha256": hashlib.sha256(original_proof.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(replacement_proof.encode()).hexdigest(),
            "expected_count": 1,
            "rationale": str(edit.get("rationale") or ""),
        })
    if not operations and not replacement_proof:
        if not edit.get("reverify_unchanged"):
            raise ValueError(
                "manual edit needs operations, replacement_proof, or reverify_unchanged=true"
            )
        return source, [{"kind": "manual_reverify_unchanged_selected_source"}]
    for index, operation in enumerate(operations):
        old = str(operation.get("old") or "")
        new = str(operation.get("new") or "")
        expected = int(operation.get("expected_count", 1))
        if not old:
            raise ValueError(f"operation {index} has empty old text")
        actual = repaired.count(old)
        if actual != expected:
            raise ValueError(
                f"operation {index} expected {expected} occurrence(s), found {actual}"
            )
        repaired = repaired.replace(old, new)
        changes.append({
            "kind": str(operation.get("kind") or "codex_manual_exact_edit"),
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            "expected_count": expected,
            "rationale": str(operation.get("rationale") or edit.get("rationale") or ""),
        })
    return repaired, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--edits-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--frozen-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()

    edits = read_jsonl(args.edits_file)
    if not edits:
        raise ValueError("manual edits file is empty")
    edit_ids = [str(edit["record_id"]) for edit in edits]
    if len(set(edit_ids)) != len(edit_ids):
        raise ValueError("manual edits contain duplicate record IDs")
    frozen_ids = frozen_record_ids(args.frozen_manifest)
    overlap = sorted(set(edit_ids) & frozen_ids)
    if overlap:
        raise ValueError(f"manual batch overlaps frozen cloud records: {overlap[:5]}")

    rows = {str(row["record_id"]): row for row in iter_jsonl(args.fail_file) if str(row["record_id"]) in set(edit_ids)}
    missing = sorted(set(edit_ids) - set(rows))
    if missing:
        raise ValueError(f"manual records absent from current fail file: {missing[:5]}")

    candidates: list[dict[str, Any]] = []
    for edit in edits:
        record_id = str(edit["record_id"])
        row = rows[record_id]
        expected_hash = str(edit.get("original_record_hash") or "")
        actual_hash = str(row.get("record_hash") or "")
        if expected_hash != actual_hash:
            raise ValueError(f"record hash mismatch for {record_id}")
        selected_field, original_source = select_numina_source(row)
        expected_source_hash = str(edit.get("selected_source_sha256") or "")
        actual_source_hash = hashlib.sha256(original_source.encode()).hexdigest()
        if expected_source_hash and expected_source_hash != actual_source_hash:
            raise ValueError(f"selected source hash mismatch for {record_id}")
        repaired_source, changes = apply_reviewed_edits(original_source, edit)
        if _FORBIDDEN.search(repaired_source):
            raise ValueError(f"forbidden proof token remains in {record_id}")
        imports, source_body, proof = split_source_and_proof(repaired_source)
        payload = {
            "schema_version": "numinamath_repair_candidate_v1",
            "repair_version": REPAIR_VERSION,
            "batch_id": args.batch_id,
            "record_id": record_id,
            "category": "codex_manual_reviewed",
            "lane": "codex_manual",
            "imports": imports,
            "source_body": source_body,
            "original_source_body": split_imports(original_source)[1],
            "selected_source_field": selected_field,
            "selected_source_sha256": actual_source_hash,
            "statement_changed": bool(edit.get("statement_changed", False)),
            "changes": changes,
            "manual_rationale": str(edit.get("rationale") or ""),
            "original_error_message": str(row.get("repair_error") or row.get("error_message") or ""),
            "original_record_hash": actual_hash,
            "variants": [{"strategy": "codex_manual_proof", "proof": proof}],
        }
        payload["candidate_hash"] = canonical_hash(payload)
        candidates.append(payload)

    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "candidate_manifest.jsonl"
    write_jsonl(manifest_path, candidates)
    report = {
        "schema_version": "numinamath_manual_prepare_report_v1",
        "repair_version": REPAIR_VERSION,
        "batch_id": args.batch_id,
        "lane": "codex_manual",
        "selected_rows": len(candidates),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "edits_file_sha256": sha256_file(args.edits_file),
        "frozen_manifests": [str(path) for path in args.frozen_manifest],
        "frozen_record_count": len(frozen_ids),
        "overlap_with_frozen": 0,
        "repair_inference_performed_by_script": False,
        "external_api_called": False,
    }
    (batch_dir / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
