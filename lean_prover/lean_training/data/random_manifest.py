"""Deterministic, without-replacement manifests for controlled SFT experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.contracts import make_attestation_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            digest.update(payload)
            handle.write(payload.decode("utf-8"))
    return digest.hexdigest()


def statement_id(row: dict[str, Any]) -> str:
    value = str(row.get("statement_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("row has no stable statement_id/id")
    return value


def record_id(row: dict[str, Any]) -> str:
    value = str(row.get("record_id") or "").strip()
    if not value:
        raise ValueError("row has no stable record_id")
    return value


def normalized_proof(row: dict[str, Any]) -> str:
    proof = str(row.get("completion") or row.get("proof") or "").strip()
    return re.sub(r"\s+", " ", proof)


def proof_homogeneity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    proofs = [normalized_proof(row) for row in rows]
    if any(not proof for proof in proofs):
        raise ValueError("manifest contains an empty proof")
    frequencies = Counter(proofs)
    duplicate_rows = sum(count - 1 for count in frequencies.values())
    return {
        "rows": len(rows),
        "unique_normalized_proofs": len(frequencies),
        "normalized_proof_exact_duplicate_count": duplicate_rows,
        "normalized_proof_duplicate_ratio": duplicate_rows / max(1, len(rows)),
        "top10_proof_coverage": sum(
            count for _, count in frequencies.most_common(10)
        )
        / max(1, len(rows)),
        "top50_proof_coverage": sum(
            count for _, count in frequencies.most_common(50)
        )
        / max(1, len(rows)),
    }


def validate_verified_row(row: dict[str, Any], *, allow_origin_discovery: bool) -> None:
    invalid = []
    for field in ("statement_verified", "proof_verified", "pantograph_verified"):
        if row.get(field) is not True:
            invalid.append(field)
    if row.get("data_state") != "verified":
        invalid.append("data_state")
    if row.get("truncated") is True or row.get("was_truncated") is True:
        invalid.append("truncated")
    if not normalized_proof(row):
        invalid.append("proof")
    expected_attestation = make_attestation_id(
        record_id=record_id(row),
        environment_hash=str(row.get("environment_hash") or ""),
        assembler_version=str(row.get("assembler_version") or ""),
        normalization_version=str(row.get("normalization_version") or ""),
        assembled_source_hash=str(row.get("assembled_source_hash") or ""),
    )
    if row.get("attestation_id") != expected_attestation:
        invalid.append("attestation_id")
    disallowed_roles = {"eval", "monitor", "benchmark"}
    role = str(row.get("data_role") or "")
    origin_role = str(row.get("origin_data_role") or "")
    if role in disallowed_roles:
        invalid.append("data_role")
    if origin_role in disallowed_roles:
        invalid.append("origin_data_role")
    if origin_role == "discovery" and not allow_origin_discovery:
        invalid.append("origin_data_role")
    if invalid:
        raise ValueError(
            f"row {row.get('record_id') or row.get('id')} violates fields {invalid}"
        )


def validate_manifest(
    rows: list[dict[str, Any]], *, allow_origin_discovery: bool
) -> dict[str, Any]:
    if not rows:
        raise ValueError("manifest is empty")
    for row in rows:
        validate_verified_row(row, allow_origin_discovery=allow_origin_discovery)
    statement_ids = [statement_id(row) for row in rows]
    record_ids = [record_id(row) for row in rows]
    if len(set(statement_ids)) != len(statement_ids):
        raise ValueError("manifest statement_id values are not unique")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("manifest record_id values are not unique")
    physical_hashes = [canonical_sha256(row) for row in rows]
    if len(set(physical_hashes)) != len(physical_hashes):
        raise ValueError("manifest contains duplicate physical rows")
    return {
        "rows": len(rows),
        "unique_statement_ids": len(set(statement_ids)),
        "unique_record_ids": len(set(record_ids)),
        "duplicate_physical_rows": len(rows) - len(set(physical_hashes)),
        "all_verified": True,
        "all_untruncated": True,
        "proof_homogeneity": proof_homogeneity(rows),
    }


def annotate_row(
    row: dict[str, Any], *, source: str, experiment: str
) -> dict[str, Any]:
    result = dict(row)
    result.pop("sample_weight", None)
    result["statement_id"] = statement_id(row)
    result["record_id"] = record_id(row)
    result["sampling_source"] = source
    result["sampling_experiment"] = experiment
    result["truncated"] = False
    return result


@dataclass(frozen=True)
class ManifestMetadata:
    source_path: str
    source_sha256: str
    eligible_record_count: int
    selection_seed: int
    sample_size: int
    selected_ids_sha256: str
    manifest_sha256: str


def deterministic_permutation(
    rows: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (statement_id(row), record_id(row)))
    if len({statement_id(row) for row in ordered}) != len(ordered):
        raise ValueError("eligible source statement_id values are not unique")
    if len({record_id(row) for row in ordered}) != len(ordered):
        raise ValueError("eligible source record_id values are not unique")
    random.Random(seed).shuffle(ordered)
    return ordered


def build_random_subset_manifest(
    source_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    *,
    dry_run: bool = False,
) -> ManifestMetadata:
    source_path = source_path.expanduser().resolve()
    rows = read_jsonl(source_path)
    eligible = []
    for row in rows:
        try:
            validate_verified_row(row, allow_origin_discovery=False)
        except ValueError:
            continue
        eligible.append(row)
    permutation = deterministic_permutation(eligible, seed=seed)
    if sample_size > len(permutation):
        raise ValueError(
            f"requested sample_size={sample_size}, eligible={len(permutation)}"
        )
    selected = permutation[:sample_size]
    selected_ids = [record_id(row) for row in selected]
    unselected_ids = [record_id(row) for row in permutation[sample_size:]]
    manifest_hash = canonical_sha256(selected)
    metadata = ManifestMetadata(
        source_path=str(source_path),
        source_sha256=file_sha256(source_path),
        eligible_record_count=len(permutation),
        selection_seed=seed,
        sample_size=sample_size,
        selected_ids_sha256=canonical_sha256(selected_ids),
        manifest_sha256=manifest_hash,
    )
    if not dry_run:
        written_hash = write_jsonl(output_path, selected)
        metadata = ManifestMetadata(
            **{**metadata.__dict__, "manifest_sha256": written_hash}
        )
        write_json(output_path.with_suffix(".selected_ids.json"), selected_ids)
        write_json(output_path.with_suffix(".unselected_ids.json"), unselected_ids)
    return metadata


def master_permutation_metadata(
    *,
    source_path: Path,
    rows: list[dict[str, Any]],
    seed: int,
    manifest_sha256: str,
) -> dict[str, Any]:
    ids = [statement_id(row) for row in rows]
    return {
        "source_path": str(source_path.expanduser().resolve()),
        "source_sha256": file_sha256(source_path),
        "eligible_record_count": len(rows),
        "selection_seed": seed,
        "shuffle_algorithm": (
            "sort(statement_id,record_id) then random.Random(seed).shuffle; "
            "CPython MT19937"
        ),
        "python_version": sys.version,
        "statement_id_hash": canonical_sha256(ids),
        "manifest_sha256": manifest_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
