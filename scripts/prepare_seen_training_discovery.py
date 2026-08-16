#!/usr/bin/env python3
"""Freeze a proof-free EI Discovery batch from data already seen by M0/H0."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    duplicate_audit,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    write_json,
    write_jsonl,
)
from scripts.prepare_expert_iteration_round0 import (
    EXPECTED_H0_MODEL_HASH,
    materialize_ld,
    materialize_wb,
    select_diverse,
)


SELECTION_SEED = 20260821
EXPECTED_CONTRACT_HASH = "26d846afd2d9e45cfe3530c63d464bc674a84026ea0fec9c7c5805ce6e0e0582"
FORBIDDEN_PROOF_FIELDS = {
    "proof",
    "completion",
    "reference_proof",
    "training_proof",
    "text",
    "raw_proof",
}


def stable_rank(value: str, namespace: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/expert_iteration/difficulty_ablation_500rows/seen_discovery/batch1"
        ),
    )
    parser.add_argument("--wb-rows", type=int, default=1000)
    parser.add_argument("--ld-rows", type=int, default=500)
    args = parser.parse_args()
    if args.wb_rows < 0 or args.ld_rows < 0 or args.wb_rows + args.ld_rows <= 0:
        raise ValueError("invalid WB/LD row request")
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    manifest_path = output / "discovery/discovery_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen discovery: {manifest_path}")
    for name in (
        "discovery",
        "success_bank",
        "failure_bank",
        "training_manifest",
        "checkpoint",
        "evaluation",
        "comparisons",
        "audit",
        "runtime/logs",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)

    pool_path = (
        project
        / "outputs/expert_iteration/difficulty_ablation_500rows/seen_training_pool/seen_training_source_pool.jsonl"
    )
    pool_audit_path = pool_path.with_name("seen_training_pool_audit.json")
    ld_full_path = (
        project
        / "outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl"
    )
    old_root = project / "outputs/expert_iteration/round0"
    h0 = (
        project
        / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED"
    )
    for path in (
        pool_path,
        pool_audit_path,
        ld_full_path,
        old_root / "discovery/generation_contract.json",
        old_root / "discovery/runtime_config.json",
        h0 / "model.safetensors",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    pool_audit = read_json(pool_audit_path)
    if pool_audit.get("status") != "SEEN_TRAINING_DISCOVERY_POOL_READY":
        raise RuntimeError("seen-training source pool is not audited")
    if sha256(h0 / "model.safetensors") != EXPECTED_H0_MODEL_HASH:
        raise RuntimeError("H0 checkpoint drifted")

    source_rows = read_jsonl(pool_path)
    wb_rows = [row for row in source_rows if str(row.get("sampling_source") or "").upper() == "WB"]
    ld_seen = [row for row in source_rows if str(row.get("sampling_source") or "").upper() == "LD"]
    if len(wb_rows) < args.wb_rows or len(ld_seen) < args.ld_rows:
        raise RuntimeError(
            f"insufficient seen-training pool: WB {len(wb_rows)}/{args.wb_rows}, "
            f"LD {len(ld_seen)}/{args.ld_rows}"
        )
    full_ld_by_id = {record_id(row): row for row in read_jsonl(ld_full_path)}
    ld_rows: list[dict[str, Any]] = []
    for seen in ld_seen:
        raw = full_ld_by_id.get(record_id(seen))
        if raw is None:
            raise RuntimeError(f"missing source-faithful LD context: {record_id(seen)}")
        if str(raw.get("proof") or "").strip() != str(seen.get("proof") or seen.get("completion") or "").strip():
            raise RuntimeError(f"LD proof identity drift: {record_id(seen)}")
        raw = dict(raw)
        raw["seen_in_checkpoints"] = list(seen.get("seen_in_checkpoints") or [])
        ld_rows.append(raw)

    selected_wb = select_diverse(wb_rows, count=args.wb_rows, source="WB")
    selected_ld = select_diverse(ld_rows, count=args.ld_rows, source="LD-easy")
    selected_wb.sort(key=lambda row: stable_rank(record_id(row), "WB"))
    selected_ld.sort(key=lambda row: stable_rank(record_id(row), "LD"))
    selected_raw = selected_wb + selected_ld
    selected_raw.sort(key=lambda row: stable_rank(record_id(row), "manifest"))
    gate = duplicate_audit(selected_raw)
    if (
        gate["duplicate_rows"]
        or gate["duplicate_record_ids"]
        or gate["theorem_group_duplicates"]
        or gate["max_repeat"] != 1
    ):
        raise RuntimeError("seen Discovery duplicate gate failed")

    wb_ids = {record_id(row) for row in selected_wb}
    materialized = [
        materialize_wb(row) if record_id(row) in wb_ids else materialize_ld(row)
        for row in selected_raw
    ]
    for public, private in zip(materialized, selected_raw):
        public["selection_seed"] = SELECTION_SEED
        public["discovery_source"] = (
            "previously_trained_wb" if public["source"] == "WB" else "previously_trained_ld_easy"
        )
        public["seen_in_checkpoints"] = list(private.get("seen_in_checkpoints") or [])
        if FORBIDDEN_PROOF_FIELDS & set(public):
            raise RuntimeError("proof field leaked into public Discovery manifest")
        prompt_value = str(public.get("prompt") or "")
        marker = "### Lean proof\n"
        if marker not in prompt_value or prompt_value.split(marker, 1)[1].strip():
            raise RuntimeError("reference proof leaked into Discovery prompt")
    if any(
        row["source"] == "LD-easy"
        and (
            not row.get("preassembled_source_template")
            or not row.get("imports")
            or row.get("imports") == ["Mathlib"]
        )
        for row in materialized
    ):
        raise RuntimeError("LD source-faithful template/import gate failed")
    write_jsonl(manifest_path, materialized)

    contract_source = old_root / "discovery/generation_contract.json"
    contract = read_json(contract_source)
    if contract.get("generation_config_sha256") != EXPECTED_CONTRACT_HASH:
        raise RuntimeError("frozen generation contract drifted")
    shutil.copy2(contract_source, output / "discovery/generation_contract.json")
    runtime = read_json(old_root / "discovery/runtime_config.json")
    runtime["run_name"] = "ei_difficulty_ablation_seen_training_discovery_batch1"
    runtime["output_dir"] = str(output / "runtime")
    runtime["data"]["discovery_path"] = str(manifest_path)
    runtime["discovery"]["statements_per_iteration"] = len(materialized)
    runtime["discovery"]["iteration_bucket_counts"] = {
        "0": {"new": len(materialized), "frontier": 0, "unsolved": 0, "audit": 0}
    }
    runtime["discovery"]["iteration_generation_seeds"] = {"0": SELECTION_SEED}
    runtime["discovery"]["generation"]["generation_contract_path"] = str(
        output / "discovery/generation_contract.json"
    )
    runtime["discovery"]["generation"]["generation_contract_sha256"] = EXPECTED_CONTRACT_HASH
    runtime["verification"].update({"num_workers": 1, "queue_maxsize": 8})
    write_json(output / "discovery/runtime_config.json", runtime)

    audit = {
        "status": "SEEN_TRAINING_DISCOVERY_FROZEN",
        "contract": "previously_used_training_data_only_v1",
        "selection_seed": SELECTION_SEED,
        "rows": len(materialized),
        "source_counts": dict(Counter(row["source"] for row in materialized)),
        "private_seen_pool_sha256": sha256(pool_path),
        "public_manifest_sha256": sha256(manifest_path),
        "duplicate_gate": gate,
        "protected_overlap": 0,
        "new_unused_wb_rows": 0,
        "new_ld_three_tier_rows": 0,
        "proof_free": True,
        "reference_proof_in_prompt": False,
        "ld_source_faithful_rows": sum(row["source"] == "LD-easy" for row in materialized),
        "ld_import_groups": len(
            {tuple(row.get("imports") or []) for row in materialized if row["source"] == "LD-easy"}
        ),
        "h0_checkpoint_sha256": EXPECTED_H0_MODEL_HASH,
        "generation_config_sha256": EXPECTED_CONTRACT_HASH,
        "candidates_per_statement": 8,
        "candidate_target": len(materialized) * 8,
        "candidate_seed": SELECTION_SEED,
    }
    write_json(output / "audit/discovery_audit.json", audit)
    write_json(
        output / "status.json",
        {
            "status": "DISCOVERY_MANIFEST_READY",
            "generation_started": False,
            "trainer_started": False,
            "grpo_started": False,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
