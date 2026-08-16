"""Freeze the immutable inputs for the LeanDojo-v2 dataset build."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.leandojo_v2_current import (
    ADAPTER_VERSION,
    CURRENT_ENVIRONMENT_HASH,
    CURRENT_LEAN_COMMIT,
    CURRENT_LEAN_VERSION,
    CURRENT_MATHLIB_COMMIT,
    LEANDOJO_V2_COMMIT,
)


EXPECTED_CANDIDATE_COUNT = 5366
EXPECTED_CANDIDATE_SHA256 = (
    "571f3536308ee1e99e5cc1c295ba1752b9f1f8d1212736e7a3cabe648dff1662"
)
EXPECTED_SAMPLE_COUNT = 500
EXPECTED_SAMPLE_SHA256 = (
    "e3ffba8fdfdc42b26f0b50b8287f4f691304adcaa13da1ba6f6962a1d34b28d7"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path(
            "outputs/leandojo_v2_retrace/processed/"
            "current_mathlib_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--sample500",
        type=Path,
        default=Path(
            "outputs/leandojo_v2_retrace/sample500/sample_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--previous-verification",
        type=Path,
        default=Path(
            "outputs/leandojo_v2_retrace/verification/"
            "verification_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/leandojo_v2_dataset_build/audit/input_freeze.json"
        ),
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    candidate = (root / args.candidate).resolve()
    sample = (root / args.sample500).resolve()
    previous_verification = (root / args.previous_verification).resolve()
    output = (root / args.output).resolve()

    candidate_count = count_jsonl(candidate)
    candidate_hash = sha256_file(candidate)
    sample_count = count_jsonl(sample)
    sample_hash = sha256_file(sample)
    if candidate_count != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            f"candidate count changed: {candidate_count} "
            f"!= {EXPECTED_CANDIDATE_COUNT}"
        )
    if candidate_hash != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            f"candidate hash changed: {candidate_hash} "
            f"!= {EXPECTED_CANDIDATE_SHA256}"
        )
    if sample_count != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"sample count changed: {sample_count} != {EXPECTED_SAMPLE_COUNT}"
        )
    if sample_hash != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(
            f"sample hash changed: {sample_hash} != {EXPECTED_SAMPLE_SHA256}"
        )
    previous = json.loads(previous_verification.read_text(encoding="utf-8"))
    if (
        int(previous.get("total") or 0) != 500
        or int(previous.get("fidelity_success") or 0) != 499
    ):
        raise RuntimeError("previous 500-sample verification baseline changed")

    payload = {
        "lean_version": CURRENT_LEAN_VERSION,
        "lean_commit": CURRENT_LEAN_COMMIT,
        "mathlib_commit": CURRENT_MATHLIB_COMMIT,
        "pantograph_version": "0.3.15",
        "environment_hash": CURRENT_ENVIRONMENT_HASH,
        "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
        "adapter_version": ADAPTER_VERSION,
        "candidate_input_path": str(candidate),
        "candidate_count": candidate_count,
        "candidate_file_hash": candidate_hash,
        "sample500_manifest_path": str(sample),
        "sample500_count": sample_count,
        "sample500_manifest_hash": sample_hash,
        "previous_sample_verification_path": str(previous_verification),
        "previous_sample_fidelity_success": 499,
        "previous_sample_total": 500,
        "verification_policy": {
            "mode": "fidelity",
            "workers": 2,
            "worker_process_isolation": True,
            "timeout_seconds": 30,
            "reject_forbidden": True,
            "broad_import_counts_as_success": False,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
