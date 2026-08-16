#!/usr/bin/env python3
"""Audit disjointness and dataset contracts after a NuminaMath campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["record_id"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--fail", type=Path, required=True)
    parser.add_argument("--invalid", type=Path, required=True)
    parser.add_argument("--need-decompose", type=Path, required=True)
    parser.add_argument("--cloud-frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "success": args.success,
        "fail": args.fail,
        "invalid": args.invalid,
        "need_decompose": args.need_decompose,
        "cloud_frozen": args.cloud_frozen,
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    id_lists = {name: ids(value) for name, value in rows.items()}
    id_sets = {name: set(value) for name, value in id_lists.items()}
    canonical_names = ("success", "fail", "invalid", "need_decompose")
    overlaps: dict[str, int] = {}
    for index, left in enumerate(canonical_names):
        for right in canonical_names[index + 1:]:
            overlaps[f"{left}__{right}"] = len(id_sets[left] & id_sets[right])

    temp_keys = {"temp_record_id", "record_index", "_record_index", "temporary_record_id"}
    success_contract = {
        "pantograph_verified_not_success": sum(row.get("pantograph_verified") != "success" for row in rows["success"]),
        "missing_source": sum(not isinstance(row.get("source"), str) or not row.get("source") for row in rows["success"]),
        "missing_record_hash": sum(not isinstance(row.get("record_hash"), str) or not row.get("record_hash") for row in rows["success"]),
        "temporary_index_fields": sum(bool(set(row) & temp_keys) for row in rows["success"]),
        "proof_contains_sorry_or_admit": sum(
            re.search(r"\b(?:sorry|admit)\b", str(row.get("proof") or row.get("formal_ground_truth") or "")) is not None
            for row in rows["success"]
        ),
    }
    need_contract = {
        "wrong_status": sum((row.get("need_decompose") or {}).get("status") != "need_decompose" for row in rows["need_decompose"]),
        "repair_eligible_not_false": sum((row.get("need_decompose") or {}).get("repair_eligible") is not False for row in rows["need_decompose"]),
        "missing_record_hash": sum(not isinstance(row.get("record_hash"), str) or not row.get("record_hash") for row in rows["need_decompose"]),
        "missing_source": sum(not isinstance(row.get("source"), str) or not row.get("source") for row in rows["need_decompose"]),
        "missing_evidence": sum(not (row.get("need_decompose") or {}).get("evidence") for row in rows["need_decompose"]),
    }
    report = {
        "schema_version": "numinamath_final_state_audit_v1",
        "counts": {name: len(value) for name, value in rows.items()},
        "unique_counts": {name: len(id_sets[name]) for name in rows},
        "duplicate_record_ids": {name: len(id_lists[name]) - len(id_sets[name]) for name in rows},
        "canonical_pairwise_overlaps": overlaps,
        "cloud_frozen_intersection_success": len(id_sets["cloud_frozen"] & id_sets["success"]),
        "cloud_frozen_intersection_need_decompose": len(id_sets["cloud_frozen"] & id_sets["need_decompose"]),
        "cloud_frozen_still_in_fail": len(id_sets["cloud_frozen"] & id_sets["fail"]),
        "success_contract_violations": success_contract,
        "need_decompose_contract_violations": need_contract,
        "file_sha256": {name: digest(path) for name, path in paths.items()},
        "canonical_total": sum(len(rows[name]) for name in canonical_names),
    }
    report["pass"] = (
        all(value == 0 for value in report["duplicate_record_ids"].values())
        and all(value == 0 for value in overlaps.values())
        and report["cloud_frozen_intersection_success"] == 0
        and report["cloud_frozen_intersection_need_decompose"] == 0
        and report["cloud_frozen_still_in_fail"] == len(rows["cloud_frozen"])
        and all(value == 0 for value in success_contract.values())
        and all(value == 0 for value in need_contract.values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
