#!/usr/bin/env python3
"""Audit the staged NuminaMath repair campaign before its single writeback."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--fail", type=Path, required=True)
    parser.add_argument("--cloud-frozen", type=Path, required=True)
    parser.add_argument("--need-decompose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_dirs: list[Path] = []
    incomplete: list[dict] = []
    for manifest in args.batch_root.glob("**/batch_*/candidate_manifest.jsonl"):
        batch_dir = manifest.parent
        match = re.search(r"batch_(\d+)", batch_dir.name)
        if not match or not 217 <= int(match.group(1)) <= 283:
            continue
        result_path = batch_dir / "verification_results.jsonl"
        if not result_path.is_file():
            incomplete.append({"batch_dir": str(batch_dir), "reason": "missing_results"})
            continue
        candidates = rows(manifest)
        results = rows(result_path)
        result_hashes = {str(row.get("candidate_hash")) for row in results}
        missing = [row["record_id"] for row in candidates if str(row.get("candidate_hash")) not in result_hashes]
        if missing:
            incomplete.append({"batch_dir": str(batch_dir), "reason": "partially_verified", "missing": len(missing)})
            continue
        batch_dirs.append(batch_dir)

    success_rows = rows(args.success)
    fail_rows = rows(args.fail)
    canonical_success_ids = {str(row["record_id"]) for row in success_rows}
    canonical_fail_ids = {str(row["record_id"]) for row in fail_rows}
    cloud_ids = {str(row["record_id"]) for row in rows(args.cloud_frozen)}
    decompose_ids = {str(row["record_id"]) for row in rows(args.need_decompose)}

    successful_ids: set[str] = set()
    attempted_ids: set[str] = set()
    total_result_rows = 0
    forbidden_success_ids: set[str] = set()
    sorry_warning_success_ids: set[str] = set()
    for batch_dir in batch_dirs:
        candidates = rows(batch_dir / "candidate_manifest.jsonl")
        results = rows(batch_dir / "verification_results.jsonl")
        attempted_ids.update(str(row["record_id"]) for row in candidates)
        for result in results:
            if result.get("success") is not True and result.get("pantograph_verified") != "success":
                continue
            record_id = str(result["record_id"])
            successful_ids.add(record_id)
            proof = str(result.get("proof") or "")
            if re.search(r"\b(?:sorry|admit|axiom)\b", proof, flags=re.I):
                forbidden_success_ids.add(record_id)
            warning_text = "\n".join(str(item) for item in result.get("warnings") or [])
            if "declaration uses 'sorry'" in warning_text.lower():
                sorry_warning_success_ids.add(record_id)
        total_result_rows += len(results)

    promoted = successful_ids & canonical_fail_ids
    report = {
        "schema_version": "numinamath_campaign_0164_precommit_audit_v1",
        "selected_batch_dirs": [str(path) for path in sorted(batch_dirs)],
        "selected_batch_count": len(batch_dirs),
        "incomplete_or_unselected": incomplete,
        "canonical_success_before": len(success_rows),
        "canonical_fail_before": len(fail_rows),
        "unique_attempted_record_ids": len(attempted_ids),
        "total_result_rows": total_result_rows,
        "unique_successful_record_ids": len(successful_ids),
        "new_promotions": len(promoted),
        "forbidden_success_proofs": len(forbidden_success_ids),
        "sorry_warning_successes": len(sorry_warning_success_ids),
        "projected_success_after": len(success_rows) + len(promoted),
        "projected_fail_after": len(fail_rows) - len(promoted),
        "cloud_overlap_attempted": len(attempted_ids & cloud_ids),
        "cloud_overlap_promoted": len(promoted & cloud_ids),
        "need_decompose_overlap_attempted": len(attempted_ids & decompose_ids),
        "need_decompose_overlap_promoted": len(promoted & decompose_ids),
        "cloud_frozen_rows": len(cloud_ids),
        "cloud_frozen_sha256": sha256(args.cloud_frozen),
        "need_decompose_rows": len(decompose_ids),
        "need_decompose_sha256": sha256(args.need_decompose),
        "success_sha256_before": sha256(args.success),
        "fail_sha256_before": sha256(args.fail),
    }
    if any(report[key] for key in (
        "cloud_overlap_attempted", "cloud_overlap_promoted",
        "need_decompose_overlap_attempted", "need_decompose_overlap_promoted",
        "forbidden_success_proofs", "sorry_warning_successes",
    )):
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
