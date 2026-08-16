#!/usr/bin/env python3
"""Prepare three focused retries after round-8 compilation diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_010000_round8_v1"
SOURCE_BATCH = SOURCE / "batches/numinamath_expand_repair_curated_010000_round8_b00000"
OUT = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_retry_010000_round8b_v1"
BATCH_ID = "numinamath_expand_repair_curated_010000_round8b_b00000"
HEARTBEAT_IDS = {
    "6f8bcb19-7deb-5dcc-8364-f32f8dfc7d3a::le_to_lt_or_eq_43f7f3636b28",
    "aea20295-2894-5c62-9133-b66c4c8254fd::le_to_lt_or_eq_bd5c85f03861",
}
NO_GOALS_ID = "896761ff-6d47-5921-b869-58e66cd43e37::and_right_dbd3308d85bd"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def main() -> None:
    candidates = {
        row["record_id"]: row
        for row in map(json.loads, (SOURCE_BATCH / "candidate_manifest.jsonl").open(encoding="utf-8"))
    }
    target_ids = HEARTBEAT_IDS | {NO_GOALS_ID}
    output_candidates = []
    review_rows = []
    for record_id in sorted(target_ids):
        candidate = dict(candidates[record_id])
        old_proof = candidate["variants"][0]["proof"]
        if record_id in HEARTBEAT_IDS:
            marker = "by\n  have h_parent"
            if not old_proof.startswith(marker):
                raise SystemExit(f"unexpected heartbeat proof header: {record_id}")
            new_proof = old_proof.replace(
                marker,
                "by\n  set_option maxHeartbeats 1000000 in\n  have h_parent",
                1,
            )
            strategy = "repair_indent_with_local_heartbeat_budget_v1"
        else:
            marker = "\n            field_simp \n            ring \n"
            if old_proof.count(marker) < 1:
                raise SystemExit("expected field_simp/ring sequence not found")
            new_proof = old_proof.replace(marker, "\n            field_simp \n", 1)
            strategy = "repair_indent_remove_post_field_simp_no_goals_v1"
        candidate["batch_id"] = BATCH_ID
        candidate["variants"] = [{"strategy": strategy, "proof": new_proof}]
        candidate["candidate_hash"] = sha_text(record_id + "\n" + strategy + "\n" + new_proof)
        candidate["repair_version"] = "numinamath_codex_minimal_repair_v1"
        output_candidates.append(candidate)
        review_rows.append(
            {
                "record_id": record_id,
                "replacement_proof": new_proof,
                "replacement_proof_sha256": sha_text(new_proof),
                "strategy": strategy,
                "semantic_change": False,
            }
        )
    batch_dir = OUT / "batches" / BATCH_ID
    manifest = batch_dir / "candidate_manifest.jsonl"
    atomic_jsonl(manifest, output_candidates)
    atomic_jsonl(OUT / "replacement_review_manifest.jsonl", review_rows)
    report = {
        "batch_id": BATCH_ID,
        "records": len(output_candidates),
        "candidate_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "statement_changes": 0,
        "semantic_changes": 0,
        "external_api_calls": 0,
    }
    atomic_json(batch_dir / "prepare_report.json", report)
    atomic_json(OUT / "prepare_report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
