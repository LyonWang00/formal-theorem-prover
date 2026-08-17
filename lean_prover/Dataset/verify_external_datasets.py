"""Audit and Pantograph-verify NuminaMath-LEAN and Kimina Promptset.

NuminaMath rows are verified as complete theorem sources.  Kimina rows are
first gated as proof-free prompt records and then elaborated with their sole
``:= by sorry`` placeholder.  Kimina success therefore means statement
elaboration success, never proof correctness.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lean_prover.Dataset.build_verified_datasets import (
    ASSEMBLER_VERSION,
    FAIL,
    HASH_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    SUCCESS,
    _first_text,
    _result_error,
    _write_json,
    add_dataset_contract,
    append_jsonl,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


NUMINA_SOURCE = "AI-MO/NuminaMath-LEAN"
NUMINA_REVISION = "51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94"
NUMINA_ROWS = 104_155
KIMINA_SOURCE = "AI-MO/Kimina-Prover-Promptset"
KIMINA_REVISION = "3009c548d90160d0f5e963d72238610c6732f812"
KIMINA_ROWS = 24_418
EXTERNAL_VERIFICATION_POLICY_VERSION = "3"

_IMPORT = re.compile(r"^\s*import\s+(.+?)\s*$")
_DECLARATION = re.compile(
    r"(?m)^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|example)\b"
)
_DECLARATION_NAME = re.compile(
    r"(?m)^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|example)\s+(?P<name>[^\s(:{]+)"
)
_PROOF_MARKER = re.compile(r":=\s*by\b")
_PLACEHOLDER = re.compile(r":=\s*by\s+sorry\s*$", re.S)
_FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom|unsafe)\b")
_KIMINA_PLACEHOLDER_ENDINGS = (
    ("placeholder_sorry", re.compile(r":=\s*by\s+sorry\s*$", re.S)),
    ("direct_sorry", re.compile(r":=\s*sorry\s*$", re.S)),
    ("empty_by", re.compile(r":=\s*by\s*$", re.S)),
    ("empty_assignment", re.compile(r":=\s*$", re.S)),
)


@dataclass(frozen=True)
class ExternalDatasetSpec:
    key: str
    source: str
    revision: str
    expected_rows: int
    raw_filename: str
    verification_scope: str

    @property
    def success_filename(self) -> str:
        return f"{self.key}_verified_success.jsonl"

    @property
    def fail_filename(self) -> str:
        return f"{self.key}_verified_fail.jsonl"

    @property
    def results_filename(self) -> str:
        return f"{self.key}_verification_results.jsonl"

    @property
    def report_filename(self) -> str:
        return f"{self.key}_verification_report.json"


SPECS = {
    "numinamath": ExternalDatasetSpec(
        key="numinamath",
        source=NUMINA_SOURCE,
        revision=NUMINA_REVISION,
        expected_rows=NUMINA_ROWS,
        raw_filename="numinamath_raw.parquet",
        verification_scope="full_proof",
    ),
    "kimina": ExternalDatasetSpec(
        key="kimina",
        source=KIMINA_SOURCE,
        revision=KIMINA_REVISION,
        expected_rows=KIMINA_ROWS,
        raw_filename="kimina_raw.parquet",
        verification_scope="statement_only",
    ),
}


def iter_parquet_rows(path: Path, *, batch_size: int = 512) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def parquet_metadata(path: Path) -> tuple[int, list[str]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, list(parquet.schema_arrow.names)


def split_imports(source: str) -> tuple[tuple[str, ...], str]:
    """Move imports outside the verifier's isolated namespace wrapper."""

    imports: list[str] = []
    body: list[str] = []
    for line in source.replace("\r\n", "\n").splitlines():
        match = _IMPORT.match(line)
        if match:
            imports.extend(part for part in match.group(1).split() if part)
        else:
            body.append(line)
    if not imports:
        imports = ["Mathlib"]
    return tuple(dict.fromkeys(imports)), "\n".join(body).strip()


def strip_placeholder_statement(source: str) -> str:
    _, body = split_imports(source)
    return _PLACEHOLDER.sub("", body).rstrip()


def extract_numina_proof(source: str) -> str:
    """Extract the first declaration's ``by`` proof for metadata and hashing."""

    _, body = split_imports(source)
    declaration = _DECLARATION.search(body)
    if declaration is None:
        return ""
    marker = _PROOF_MARKER.search(body, declaration.start())
    if marker is None:
        return ""
    by_start = body.find("by", marker.start())
    proof = body[by_start:].strip() if by_start >= 0 else ""
    return proof if re.sub(r"^by\b", "", proof, count=1).strip() else ""


def select_numina_source(row: Mapping[str, Any]) -> tuple[str, str]:
    """Select the most complete source without joining already-complete fields."""

    ground_truth = str(row.get("formal_ground_truth") or "").strip()
    if ground_truth:
        return "formal_ground_truth", ground_truth
    formal_proof = str(row.get("formal_proof") or "").strip()
    if formal_proof:
        return "formal_proof", formal_proof
    return (
        "formal_statement_missing_proof",
        str(row.get("formal_statement") or "").strip(),
    )


def normalize_kimina_target(
    source: str, *, name: str = ""
) -> tuple[tuple[str, ...], str, str, dict[str, Any]]:
    """Extract the target declaration and canonicalize its proof placeholder.

    A handful of upstream rows contain an unrelated proved theorem before the
    target prompt.  Only the named (normally final) declaration is retained;
    imports and declaration-free preamble remain intact.
    """

    imports, body = split_imports(source)
    declarations = list(_DECLARATION_NAME.finditer(body))
    if not declarations:
        raise ValueError("missing target theorem/lemma/example declaration")
    def normalized_segment(
        match: re.Match[str], match_index: int
    ) -> tuple[str, str, int] | None:
        end = (
            declarations[match_index + 1].start()
            if match_index + 1 < len(declarations)
            else len(body)
        )
        segment = body[match.start() : end].strip()
        for candidate_category, pattern in _KIMINA_PLACEHOLDER_ENDINGS:
            if pattern.search(segment):
                return candidate_category, pattern.sub("", segment).rstrip(), end
        return None

    target = None
    requested_name = name.strip()
    if requested_name:
        target = next(
            (
                match
                for match in reversed(declarations)
                if match.group("name") == requested_name
            ),
            None,
        )
    named_target = target
    matched_by_name = named_target is not None
    target_name_fallback = False
    normalized = None
    if target is not None:
        normalized = normalized_segment(target, declarations.index(target))
    if normalized is None:
        target_name_fallback = target is not None
        for candidate_index in range(len(declarations) - 1, -1, -1):
            candidate = declarations[candidate_index]
            candidate_normalized = normalized_segment(candidate, candidate_index)
            if candidate_normalized is not None:
                target = candidate
                normalized = candidate_normalized
                break
    if target is None or normalized is None:
        raise ValueError("no proof-free target declaration was found")
    target_index = declarations.index(target)
    category, lean_statement, target_end = normalized
    if not lean_statement or not _DECLARATION.search(lean_statement):
        raise ValueError("empty target statement after placeholder normalization")
    if _FORBIDDEN.search(lean_statement):
        raise ValueError("forbidden proof token remains in normalized target statement")
    normalized_body = (
        body[: target.start()].rstrip()
        + ("\n\n" if body[: target.start()].strip() else "")
        + f"{lean_statement} := by sorry"
        + ("\n\n" if body[target_end:].strip() else "")
        + body[target_end:].lstrip()
    )
    metadata = {
        "raw_placeholder_category": category,
        "raw_declaration_count": len(declarations),
        "supporting_prefix_declarations": target_index,
        "sibling_declarations": len(declarations) - 1,
        "target_name_matched": matched_by_name,
        "target_name_fallback": target_name_fallback,
        "resolved_target_name": target.group("name"),
    }
    return imports, lean_statement, normalized_body, metadata


def _record_id(
    row: Mapping[str, Any], spec: ExternalDatasetSpec, problem_index: int
) -> str:
    if spec.key == "numinamath":
        base = _first_text(row, ("uuid", "record_id", "id"))
    else:
        base = _first_text(row, ("statement_id", "record_id", "id", "name"))
    if not base:
        base = sha256_text(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
        )[:24]
    # Upstream IDs are group identifiers and are not unique per Parquet row.
    return f"{base}::row_{problem_index:06d}"


def audit_numinamath(path: Path) -> dict[str, Any]:
    row_count, columns = parquet_metadata(path)
    required = {"uuid", "formal_statement", "formal_ground_truth"}
    missing_columns = sorted(required - set(columns))
    ids: set[str] = set()
    duplicate_ids = 0
    empty_ground_truth = 0
    empty_formal_statement = 0
    empty_selected_source = 0
    missing_declaration = 0
    missing_by_proof = 0
    forbidden_sources = 0
    formal_proof_nonempty = 0
    proof_source_counts: Counter[str] = Counter()
    proof_source_samples: dict[str, list[dict[str, Any]]] = {}
    ground_truth_types: Counter[str] = Counter()
    for row in iter_parquet_rows(path):
        record_id = str(row.get("uuid") or "").strip()
        if not record_id or record_id in ids:
            duplicate_ids += 1
        ids.add(record_id)
        source = str(row.get("formal_ground_truth") or "").strip()
        formal_statement = str(row.get("formal_statement") or "").strip()
        if not formal_statement:
            empty_formal_statement += 1
        if not source:
            empty_ground_truth += 1
        proof_source, selected_source = select_numina_source(row)
        if not selected_source:
            empty_selected_source += 1
        if not _DECLARATION.search(selected_source):
            missing_declaration += 1
        if not extract_numina_proof(selected_source):
            missing_by_proof += 1
        if _FORBIDDEN.search(selected_source):
            forbidden_sources += 1
        if str(row.get("formal_proof") or "").strip():
            formal_proof_nonempty += 1
        proof_source_counts[proof_source] += 1
        samples = proof_source_samples.setdefault(proof_source, [])
        if len(samples) < 3:
            samples.append(
                {
                    "uuid": row.get("uuid"),
                    "ground_truth_type": row.get("ground_truth_type"),
                    "formal_statement_tail": str(
                        row.get("formal_statement") or ""
                    )[-300:],
                    "formal_proof_head": str(row.get("formal_proof") or "")[:300],
                    "formal_ground_truth_head": source[:300],
                }
            )
        ground_truth_types[str(row.get("ground_truth_type") or "unknown")] += 1
    passed = (
        row_count == NUMINA_ROWS
        and not missing_columns
        and empty_formal_statement == 0
        and empty_selected_source == 0
        and missing_declaration == 0
    )
    return {
        "dataset": NUMINA_SOURCE,
        "revision": NUMINA_REVISION,
        "rows": row_count,
        "columns": columns,
        "missing_required_columns": missing_columns,
        "unique_ids": len(ids),
        "duplicate_or_missing_ids": duplicate_ids,
        "empty_formal_ground_truth": empty_ground_truth,
        "empty_formal_statement": empty_formal_statement,
        "empty_selected_source": empty_selected_source,
        "missing_declaration": missing_declaration,
        "missing_by_proof": missing_by_proof,
        "forbidden_source_rows": forbidden_sources,
        "formal_proof_nonempty": formal_proof_nonempty,
        "proof_source_counts": dict(proof_source_counts),
        "proof_source_samples": proof_source_samples,
        "ground_truth_types": dict(ground_truth_types),
        "verification_scope": "full_proof",
        "schema_gate_passed": passed,
    }


def audit_kimina(path: Path) -> dict[str, Any]:
    row_count, columns = parquet_metadata(path)
    expected_columns = {
        "statement_id",
        "natural_language",
        "formal_statement",
        "source",
        "name",
    }
    proof_columns = sorted(
        column
        for column in columns
        if column.lower() in {
            "proof",
            "formal_proof",
            "reference_proof",
            "formal_ground_truth",
        }
    )
    ids: set[str] = set()
    duplicate_ids = 0
    empty_statements = 0
    normalized_target_rows = 0
    normalization_errors = 0
    target_name_mismatches = 0
    supporting_prefix_rows = 0
    target_name_fallback_rows = 0
    tail_categories: Counter[str] = Counter()
    tail_samples: dict[str, list[dict[str, Any]]] = {}
    for row in iter_parquet_rows(path):
        record_id = str(row.get("statement_id") or "").strip()
        if not record_id or record_id in ids:
            duplicate_ids += 1
        ids.add(record_id)
        source = str(row.get("formal_statement") or "").strip()
        if not source:
            empty_statements += 1
            continue
        normalization_error = ""
        try:
            _, _, _, metadata = normalize_kimina_target(
                source, name=str(row.get("name") or "")
            )
            category = str(metadata["raw_placeholder_category"])
            normalized_target_rows += 1
            if not metadata["target_name_matched"]:
                target_name_mismatches += 1
            if metadata["supporting_prefix_declarations"]:
                supporting_prefix_rows += 1
            if metadata["target_name_fallback"]:
                target_name_fallback_rows += 1
        except ValueError as exc:
            category = "normalization_error"
            normalization_errors += 1
            normalization_error = str(exc)
        tail_categories[category] += 1
        samples = tail_samples.setdefault(category, [])
        if len(samples) < 5:
            samples.append(
                {
                    "statement_id": row.get("statement_id"),
                    "name": row.get("name"),
                    "formal_statement_tail": source[-500:],
                    **(
                        {"normalization_error": normalization_error}
                        if normalization_error
                        else {}
                    ),
                }
            )
    passed = (
        row_count == KIMINA_ROWS
        and set(columns) == expected_columns
        and not proof_columns
        and empty_statements == 0
        and normalized_target_rows == row_count
        and normalization_errors == 0
    )
    return {
        "dataset": KIMINA_SOURCE,
        "revision": KIMINA_REVISION,
        "rows": row_count,
        "columns": columns,
        "expected_columns": sorted(expected_columns),
        "proof_columns": proof_columns,
        "unique_ids": len(ids),
        "duplicate_or_missing_ids": duplicate_ids,
        "empty_formal_statements": empty_statements,
        "normalized_target_rows": normalized_target_rows,
        "normalization_errors": normalization_errors,
        "target_name_mismatches": target_name_mismatches,
        "supporting_prefix_rows": supporting_prefix_rows,
        "target_name_fallback_rows": target_name_fallback_rows,
        "tail_categories": dict(tail_categories),
        "tail_samples": tail_samples,
        "proof_free_after_target_normalization": passed,
        "verification_scope": "statement_only",
        "success_semantics": "statement elaborates with placeholder; no proof claim",
        "schema_gate_passed": passed,
    }


def audit_dataset(
    *, spec: ExternalDatasetSpec, dataset_root: Path
) -> dict[str, Any]:
    raw_path = dataset_root / "raw_data" / spec.raw_filename
    report = (
        audit_numinamath(raw_path)
        if spec.key == "numinamath"
        else audit_kimina(raw_path)
    )
    report.update(
        {
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": sha256_file(raw_path),
            "raw_bytes": raw_path.stat().st_size,
        }
    )
    _write_json(dataset_root / f"{spec.key}_raw_audit.json", report)
    return report


def build_external_task(
    row: Mapping[str, Any],
    *,
    spec: ExternalDatasetSpec,
    problem_index: int,
    environment_hash: str,
) -> VerificationTask:
    record_id = _record_id(row, spec, problem_index)
    if not record_id:
        raise ValueError("missing record ID")
    if spec.key == "numinamath":
        proof_source, source = select_numina_source(row)
        generated_proof = extract_numina_proof(source) or source
        reject_forbidden = False
        task_metadata: dict[str, Any] = {
            "proof_source": proof_source,
            "proof_present": bool(extract_numina_proof(source)),
            "contains_forbidden": bool(_FORBIDDEN.search(source)),
        }
        imports, lean_code = split_imports(source)
    else:
        source = str(row.get("formal_statement") or "").strip()
        imports, lean_statement, lean_code, task_metadata = normalize_kimina_target(
            source, name=str(row.get("name") or "")
        )
        task_metadata["normalized_lean_statement"] = lean_statement
        generated_proof = "by sorry"
        reject_forbidden = False
    upstream_imports = imports
    # The persistent server is intentionally loaded once with the broad
    # Mathlib umbrella import.  Requiring task import tuples to be textually
    # identical would falsely reject sources that also spell out Aesop or a
    # narrower Mathlib module already covered by Mathlib.
    imports = ("Mathlib",)
    if not lean_code or not _DECLARATION.search(lean_code):
        raise ValueError("missing Lean declaration source")
    task = VerificationTask(
        priority=problem_index,
        problem_index=problem_index,
        attempt_index=0,
        problem_id=record_id,
        prompt="",
        generated_proof=generated_proof,
        raw_completion=generated_proof,
        lean_code=lean_code,
        imports=imports,
        payload={
            "record_id": record_id,
            "source": spec.source,
            "dataset_revision": spec.revision,
            "verification_scope": spec.verification_scope,
            "environment_hash": environment_hash,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "verification_policy_version": EXTERNAL_VERIFICATION_POLICY_VERSION,
            "upstream_imports": list(upstream_imports),
            **task_metadata,
        },
        reject_forbidden=reject_forbidden,
    )
    assembled = build_labeled_lean_code(task, include_imports=True)
    task.payload["assembled_source_hash"] = sha256_text(assembled)
    return task


def _compatible_result(
    result: Mapping[str, Any],
    task: VerificationTask,
    *,
    spec: ExternalDatasetSpec,
    environment_hash: str,
) -> bool:
    payload = task.payload or {}
    return (
        str(result.get("record_id") or result.get("problem_id") or "")
        == task.problem_id
        and result.get("source") == spec.source
        and result.get("dataset_revision") == spec.revision
        and result.get("verification_scope") == spec.verification_scope
        and result.get("environment_hash") == environment_hash
        and result.get("assembler_version") == ASSEMBLER_VERSION
        and result.get("normalization_version") == NORMALIZATION_VERSION
        and result.get("verification_policy_version")
        == EXTERNAL_VERIFICATION_POLICY_VERSION
        and result.get("assembled_source_hash") == payload.get("assembled_source_hash")
    )


def _load_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for result in read_jsonl(path):
        record_id = str(result.get("record_id") or result.get("problem_id") or "")
        if record_id:
            index.setdefault(record_id, []).append(result)
    return index


def _matching_result(
    index: Mapping[str, list[dict[str, Any]]],
    task: VerificationTask,
    *,
    spec: ExternalDatasetSpec,
    environment_hash: str,
) -> dict[str, Any] | None:
    return next(
        (
            result
            for result in reversed(index.get(task.problem_id, ()))
            if _compatible_result(
                result,
                task,
                spec=spec,
                environment_hash=environment_hash,
            )
        ),
        None,
    )


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "problem_id",
        "record_id",
        "problem_index",
        "attempt_index",
        "status",
        "success",
        "diagnostics",
        "compile_errors",
        "compile_warnings",
        "timed_out",
        "verifier_backend",
        "rejected_reason",
        "verification_seconds",
        "assembled_source_hash",
        "source",
        "dataset_revision",
        "verification_scope",
        "environment_hash",
        "assembler_version",
        "normalization_version",
        "verification_policy_version",
        "contains_forbidden",
        "worker_pid",
        "worker_generation",
        "worker_restart_count",
        "pantograph_restart_count",
    )
    return {key: result.get(key) for key in keys if key in result}


def effective_success(
    result: Mapping[str, Any], *, spec: ExternalDatasetSpec
) -> bool:
    """Interpret warning-only placeholder elaboration as Kimina success."""

    if spec.verification_scope == "full_proof":
        return bool(result.get("success")) and not bool(
            result.get("contains_forbidden")
        )
    if bool(result.get("success")):
        return True
    if result.get("timed_out") or result.get("rejected_reason"):
        return False
    diagnostics = [
        str(item).strip()
        for item in (result.get("compile_errors") or [])
        if str(item).strip()
    ]
    if not diagnostics:
        diagnostics = [
            line.strip()
            for line in str(result.get("diagnostics") or "").splitlines()
            if line.strip()
        ]
    return bool(diagnostics) and all("warning:" in item for item in diagnostics)


def _normalized_output_row(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    task: VerificationTask,
    spec: ExternalDatasetSpec,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(row)
    upstream_source = payload.get("source")
    if upstream_source is not None:
        payload["upstream_source"] = upstream_source
    payload["record_id"] = task.problem_id
    if spec.key == "numinamath":
        proof_source, selected_source = select_numina_source(row)
        payload["lean_statement"] = re.sub(
            r":=\s*by\s*$",
            "",
            str(row.get("formal_statement") or "").strip(),
            flags=re.S,
        ).strip()
        payload["proof"] = extract_numina_proof(selected_source)
        payload["proof_present"] = bool(payload["proof"])
        payload["proof_source"] = proof_source
    else:
        _, lean_statement, _, metadata = normalize_kimina_target(
            str(row.get("formal_statement") or ""),
            name=str(row.get("name") or ""),
        )
        payload["lean_statement"] = lean_statement
        payload.pop("proof", None)
        payload["proof_present"] = False
        payload["placeholder_proof"] = "by sorry"
        payload.update(metadata)
    is_success = effective_success(result, spec=spec)
    payload.update(
        {
            "dataset_revision": spec.revision,
            "verification_scope": spec.verification_scope,
            "lean_version": identity["lean_version"],
            "mathlib_commit": identity["mathlib_commit"],
            "environment_hash": identity["environment_hash"],
            "assembled_source_hash": result.get("assembled_source_hash"),
            "verification_status": "success" if is_success else result.get("status"),
            "raw_verification_status": result.get("status"),
            "raw_verifier_success": bool(result.get("success")),
            "verification_backend": result.get("verifier_backend") or "pantograph",
            "timed_out": bool(result.get("timed_out")),
        }
    )
    status = SUCCESS if is_success else FAIL
    error_message = None
    if status == FAIL:
        forbidden_match = (
            _FORBIDDEN.search(selected_source) if spec.key == "numinamath" else None
        )
        if forbidden_match:
            error_message = (
                f"forbidden proof token `{forbidden_match.group(0)}`; "
                f"Pantograph diagnostic: {_result_error(result)}"
            )
        else:
            error_message = _result_error(result)
    return add_dataset_contract(
        payload,
        source=spec.source,
        status=status,
        error_message=error_message,
    )


def verify_external_dataset(
    *,
    spec: ExternalDatasetSpec,
    dataset_root: Path,
    lean_project: Path,
    workers: int,
    timeout: int,
    batch_size: int,
    max_new_compilations: int | None = None,
    reports_only: bool = False,
) -> dict[str, Any]:
    audit = audit_dataset(spec=spec, dataset_root=dataset_root)
    if not audit["schema_gate_passed"]:
        raise RuntimeError(
            f"{spec.source} schema gate failed; see "
            f"{dataset_root / f'{spec.key}_raw_audit.json'}"
        )

    raw_path = dataset_root / "raw_data" / spec.raw_filename
    verified_dir = dataset_root / "verified_data"
    verified_dir.mkdir(parents=True, exist_ok=True)
    results_path = verified_dir / spec.results_filename
    success_path = verified_dir / spec.success_filename
    fail_path = verified_dir / spec.fail_filename
    report_path = verified_dir / spec.report_filename
    identity = environment_identity(lean_project, ("Mathlib",))
    environment_hash = str(identity["environment_hash"])
    result_index = _load_results(results_path)

    compatible_before = 0
    missing_before = 0
    for problem_index, row in enumerate(iter_parquet_rows(raw_path)):
        task = build_external_task(
            row,
            spec=spec,
            problem_index=problem_index,
            environment_hash=environment_hash,
        )
        if _matching_result(
            result_index,
            task,
            spec=spec,
            environment_hash=environment_hash,
        ):
            compatible_before += 1
        else:
            missing_before += 1
    if reports_only and missing_before:
        raise RuntimeError(f"{missing_before} compatible Pantograph results are missing")
    requested = missing_before
    if max_new_compilations is not None:
        requested = min(requested, max_new_compilations)

    completed_new = 0
    runtime: dict[str, Any] = {}

    def persist(raw_result: dict[str, Any]) -> None:
        nonlocal completed_new
        result = _compact_result(raw_result)
        append_jsonl(results_path, result)
        record_id = str(result.get("record_id") or result.get("problem_id") or "")
        result_index.setdefault(record_id, []).append(result)
        completed_new += 1
        if completed_new % 100 == 0:
            print(
                json.dumps(
                    {
                        "dataset": spec.key,
                        "phase": "pantograph_verification",
                        "completed_new": completed_new,
                        "requested_new": requested,
                    }
                ),
                flush=True,
            )

    if requested:
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=str(lean_project),
                imports=("Mathlib",),
                timeout=timeout,
                warmup_timeout=3600,
                num_workers=workers,
                queue_maxsize=max(8, workers * 4),
                max_worker_restarts=3,
                max_task_retries=1,
                shutdown_timeout=15,
                task_spool_dir=str(
                    verified_dir / "runtime" / f"{spec.key}_task_spool"
                ),
            )
        )
        with pool:
            task_batch: list[VerificationTask] = []
            # Numina contains a large statement-only/missing-proof tail plus
            # explicit placeholder proofs.  They still go through Pantograph,
            # but scheduling them first makes the restartable cache useful
            # early without changing row coverage or final output ordering.
            priority_passes: tuple[int | None, ...] = (
                (0, 1, 2) if spec.key == "numinamath" else (None,)
            )
            stop_requested = False
            for priority_class in priority_passes:
                for problem_index, row in enumerate(iter_parquet_rows(raw_path)):
                    if completed_new + len(task_batch) >= requested:
                        stop_requested = True
                        break
                    if priority_class is not None:
                        _, selected_source = select_numina_source(row)
                        row_priority = (
                            0
                            if not extract_numina_proof(selected_source)
                            else 1
                            if _FORBIDDEN.search(selected_source)
                            else 2
                        )
                        if row_priority != priority_class:
                            continue
                    task = build_external_task(
                        row,
                        spec=spec,
                        problem_index=problem_index,
                        environment_hash=environment_hash,
                    )
                    if _matching_result(
                        result_index,
                        task,
                        spec=spec,
                        environment_hash=environment_hash,
                    ):
                        continue
                    task_batch.append(task)
                    if len(task_batch) >= batch_size:
                        run = pool.run_batch(task_batch, on_result=persist)
                        runtime = run.runtime_stats
                        if run.fatal_errors:
                            raise RuntimeError(
                                f"Pantograph pool failed: {run.fatal_errors}"
                            )
                        task_batch = []
                if stop_requested:
                    break
            if task_batch:
                run = pool.run_batch(task_batch, on_result=persist)
                runtime = run.runtime_stats
                if run.fatal_errors:
                    raise RuntimeError(f"Pantograph pool failed: {run.fatal_errors}")

    success_count = 0
    fail_count = 0
    unresolved_count = 0
    failure_statuses: Counter[str] = Counter()
    success_file = success_path.open("w", encoding="utf-8")
    fail_file = fail_path.open("w", encoding="utf-8")
    try:
        for problem_index, row in enumerate(iter_parquet_rows(raw_path)):
            task = build_external_task(
                row,
                spec=spec,
                problem_index=problem_index,
                environment_hash=environment_hash,
            )
            result = _matching_result(
                result_index,
                task,
                spec=spec,
                environment_hash=environment_hash,
            )
            if result is None:
                unresolved_count += 1
                continue
            output = _normalized_output_row(
                row,
                result,
                task=task,
                spec=spec,
                identity=identity,
            )
            is_success = effective_success(result, spec=spec)
            target = success_file if is_success else fail_file
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            if is_success:
                success_count += 1
            else:
                fail_count += 1
                failure_statuses[str(result.get("status") or "unknown")] += 1
    finally:
        success_file.close()
        fail_file.close()

    complete = unresolved_count == 0 and (
        success_count + fail_count == spec.expected_rows
    )
    report = {
        "dataset": spec.source,
        "revision": spec.revision,
        "verification_scope": spec.verification_scope,
        "success_semantics": (
            "complete proof accepted by Pantograph"
            if spec.verification_scope == "full_proof"
            else "statement elaborates with := by sorry placeholder; no proof claim"
        ),
        "environment": identity,
        "raw": {
            "path": str(raw_path.resolve()),
            "rows": audit["rows"],
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "schema_audit": audit,
        "compatible_results_before_run": compatible_before,
        "missing_results_before_run": missing_before,
        "new_compilations_requested": requested,
        "new_compilations_completed": completed_new,
        "success": success_count,
        "fail": fail_count,
        "unresolved": unresolved_count,
        "failure_statuses": dict(failure_statuses),
        "complete": complete,
        "outputs": {
            "success": {
                "path": str(success_path.resolve()),
                "rows": success_count,
                "sha256": sha256_file(success_path),
            },
            "fail": {
                "path": str(fail_path.resolve()),
                "rows": fail_count,
                "sha256": sha256_file(fail_path),
            },
            "results_cache": str(results_path.resolve()),
        },
        "runtime": runtime,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-compilations", type=int)
    parser.add_argument("--reports-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = SPECS[args.dataset]
    if args.audit_only:
        report = audit_dataset(spec=spec, dataset_root=args.dataset_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    verify_external_dataset(
        spec=spec,
        dataset_root=args.dataset_root,
        lean_project=args.lean_project,
        workers=args.workers,
        timeout=args.timeout,
        batch_size=args.batch_size,
        max_new_compilations=args.max_new_compilations,
        reports_only=args.reports_only,
    )


if __name__ == "__main__":
    main()
