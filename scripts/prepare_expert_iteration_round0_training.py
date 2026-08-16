"""Freeze the approved 250-row Expert Iteration Round-0 SFT manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    duplicate_audit,
    identity,
    identity_sets,
    overlap,
    record_id,
    sha256,
    theorem_group,
)
from lean_prover.lean_training.data.contracts import make_attestation_id


SELECTION_SEED = 20260811
EXPECTED_ROWS = 250
EXPECTED_SUCCESS = 200
EXPECTED_FRONTIER = 50
EXPECTED_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
EXPECTED_H0_MODEL_HASH = (
    "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
)
EXPECTED_DISCOVERY_CONTRACT_HASH = (
    "26d846afd2d9e45cfe3530c63d464bc674a84026ea0fec9c7c5805ce6e0e0582"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def stable_rank(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{namespace}:{value}".encode()).hexdigest()


def candidate_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("generation_length") or 10**9),
        int(row.get("candidate_rank") or 10**9),
        str(row.get("candidate_id") or ""),
    )


def protected_sets(project: Path) -> dict[str, list[dict[str, Any]]]:
    index_path = project / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json"
    index = read_json(index_path)
    protected: dict[str, list[dict[str, Any]]] = {}
    for name, entry in index.items():
        if entry.get("role") not in {"true_holdout", "future_true_holdout"}:
            continue
        path = Path(str(entry.get("path") or ""))
        if path.is_file():
            protected[name] = read_jsonl(path)
    return protected


def repair_existing_attestation(root: Path) -> None:
    """Repair only the pre-Trainer manifest alias/ID mapping caught by the SFT gate."""

    manifest_path = root / "training_manifest/ei_round0_train.jsonl"
    gate_path = root / "training_manifest/manifest_gate.json"
    status_path = root / "status.json"
    identity_path = root / "checkpoint/training_identity.json"
    trainer_dir = root / "checkpoint/trainer"
    checkpoint_dir = root / "checkpoint/EI-Round0"
    if not manifest_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("repair requires the already-frozen manifest and gate")
    if trainer_dir.exists() or checkpoint_dir.exists():
        raise RuntimeError("refusing attestation repair after trainer/checkpoint creation")
    gate = read_json(gate_path)
    if sha256(manifest_path) != gate.get("manifest_sha256"):
        raise RuntimeError("manifest hash drifted before repair")
    rows = read_jsonl(manifest_path)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("repair target row count drifted")
    for row in rows:
        row["data_state"] = "verified"
        row["attestation_id"] = make_attestation_id(
            record_id=str(row["record_id"]),
            environment_hash=str(row["environment_hash"]),
            assembler_version=str(row["assembler_version"]),
            normalization_version=str(row["normalization_version"]),
            assembled_source_hash=str(row["assembled_source_hash"]),
        )
        row["ei_round0"]["candidate_attestation_id"] = row["attestation_id"]
    write_jsonl(manifest_path, rows)
    gate["manifest_sha256"] = sha256(manifest_path)
    gate["attestation_contract_repair"] = {
        "reason": "SFT pre-Trainer gate requires data_state and canonical make_attestation_id",
        "rows_repaired": len(rows),
        "proof_content_changed": False,
        "trainer_or_optimizer_steps_before_repair": 0,
        "repaired_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(gate_path, gate)
    status = read_json(status_path)
    status.update(
        {
            "status": "TRAINING_MANIFEST_REPAIRED_PRETRAINER_GATE",
            "trainer_started": False,
            "trainer_completed": False,
            "grpo_started": False,
            "training_manifest_sha256": gate["manifest_sha256"],
            "pretrainer_gate_failures": ["missing data_state", "noncanonical attestation_id"],
        }
    )
    write_json(status_path, status)
    if identity_path.exists():
        identity_path.unlink()
    print(json.dumps(gate["attestation_contract_repair"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--repair-existing-attestation", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/round0"
    if args.repair_existing_attestation:
        repair_existing_attestation(root)
        return
    train_root = root / "training_manifest"
    manifest_path = train_root / "ei_round0_train.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen EI manifest: {manifest_path}")

    paths = {
        "discovery": root / "discovery/discovery_manifest.jsonl",
        "success": root / "success_bank/success_bank.jsonl",
        "results": root / "discovery/candidate_results.jsonl",
        "verifications": root / "discovery/candidate_verifications.jsonl",
        "summary": root / "discovery/discovery_summary.json",
        "generation_contract": root / "discovery/generation_contract.json",
        "feasibility": train_root / "feasibility_report.json",
        "h0_manifest": project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/manifest.jsonl",
        "h0_model": project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED",
        "validation": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if sha256(paths["h0_model"] / "model.safetensors") != EXPECTED_H0_MODEL_HASH:
        raise RuntimeError("H0 frozen model hash drifted")
    contract = read_json(paths["generation_contract"])
    if contract.get("generation_config_sha256") != EXPECTED_DISCOVERY_CONTRACT_HASH:
        raise RuntimeError("Discovery generation contract drifted")
    feasibility = read_json(paths["feasibility"])
    if feasibility.get("recommended_option") != "A_preserve_generated_only_and_80_20":
        raise RuntimeError("feasibility recommendation drifted")

    discovery = read_jsonl(paths["discovery"])
    successes = read_jsonl(paths["success"])
    results = read_jsonl(paths["results"])
    verifications = read_jsonl(paths["verifications"])
    if (len(discovery), len(results), len(verifications), len(successes)) != (500, 4000, 4000, 722):
        raise RuntimeError("Discovery artifact counts drifted")

    discovery_by_record = {record_id(row): row for row in discovery}
    verification_by_id = {str(row["generation_id"]): row for row in verifications}
    success_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        success_by_record[str(row["source_id"])].append(row)
    if len(success_by_record) != 253:
        raise RuntimeError("unique Success Bank theorem count drifted")
    success_by_group: dict[str, list[dict[str, Any]]] = {}
    discovery_by_group: dict[str, dict[str, Any]] = {}
    for source_id, candidates in success_by_record.items():
        source = discovery_by_record.get(source_id)
        if source is None:
            raise RuntimeError(f"success source absent from Discovery manifest: {source_id}")
        group = theorem_group(source)
        if group in success_by_group:
            raise RuntimeError(f"multiple source records map to theorem group: {group}")
        success_by_group[group] = candidates
        discovery_by_group[group] = source

    eligible: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["h0_model"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("H0 tokenizer IDs drifted")
    eos_token = str(tokenizer.eos_token)
    for group, candidates in success_by_group.items():
        candidate = min(candidates, key=candidate_key)
        verification = verification_by_id.get(str(candidate["candidate_id"]))
        if verification is None:
            raise RuntimeError(f"raw verification missing: {candidate['candidate_id']}")
        if not (
            candidate.get("pantograph_verified") is True
            and candidate.get("no_sorry") is True
            and candidate.get("valid_proof_extraction") is True
            and verification.get("verified") is True
            and verification.get("status") == "success"
            and verification.get("environment_hash") == EXPECTED_ENVIRONMENT_HASH
            and not verification.get("contains_sorry")
            and not verification.get("contains_admit")
            and not verification.get("contains_axiom")
        ):
            raise RuntimeError(f"candidate attestation failed: {candidate['candidate_id']}")
        source = discovery_by_group[group]
        proof = str(candidate["proof"])
        prompt = str(source["prompt"])
        completion = proof + eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"prompt prefix mismatch: {group}")
        labels = full_ids[len(prompt_ids) :]
        if not labels or labels[-1] != tokenizer.eos_token_id or len(full_ids) > 1024:
            continue
        eligible[group] = (source, candidate, verification)
    if len(eligible) < 250:
        raise RuntimeError(f"only {len(eligible)} eligible unique generated proofs")

    partial_groups = [group for group, rows in success_by_group.items() if 1 <= len(rows) < 8 and group in eligible]
    easy_groups = [group for group, rows in success_by_group.items() if len(rows) == 8 and group in eligible]
    frontier_groups = sorted(partial_groups, key=lambda group: stable_rank(group, "frontier"))[:EXPECTED_FRONTIER]
    remaining = [group for group in eligible if group not in set(frontier_groups)]
    success_groups = sorted(remaining, key=lambda group: stable_rank(group, "success"))[:EXPECTED_SUCCESS]
    if len(frontier_groups) != EXPECTED_FRONTIER or len(success_groups) != EXPECTED_SUCCESS:
        raise RuntimeError("approved 200/50 selection unavailable")

    def materialize(group: str, role: str) -> dict[str, Any]:
        source, candidate, verification = eligible[group]
        proof = str(candidate["proof"])
        prompt = str(source["prompt"])
        attestation_id = make_attestation_id(
            record_id=str(source.get("record_id") or source.get("id")),
            environment_hash=str(verification["environment_hash"]),
            assembler_version=str(verification["assembler_version"]),
            normalization_version=str(verification["normalization_version"]),
            assembled_source_hash=str(verification["assembled_source_hash"]),
        )
        row = dict(source)
        row.update(
            {
                "id": str(source.get("id") or source.get("record_id")),
                "record_id": str(source.get("record_id") or source.get("id")),
                "statement_id": group,
                "theorem_group_id": group,
                "lean_statement": str(source.get("lean_statement") or source.get("statement")),
                "proof": proof,
                "completion": proof + eos_token,
                "text": prompt + proof + eos_token,
                "sampling_source": str(source.get("source") or source.get("source_name")),
                "pantograph_verified": True,
                "data_state": "verified",
                "proof_verified": True,
                "statement_verified": True,
                "reference_proof_verified": True,
                "verification_status": "verified",
                "environment_hash": verification["environment_hash"],
                "verification_environment_hash": verification["environment_hash"],
                "lean_version": verification["lean_version"],
                "mathlib_commit": verification["mathlib_commit"],
                "assembler_version": verification["assembler_version"],
                "normalization_version": verification["normalization_version"],
                "assembled_source_hash": verification["assembled_source_hash"],
                "attestation_id": attestation_id,
                "attested_at": verification.get("verified_at"),
                "ei_round0": {
                    "round_id": 0,
                    "top_level_role": role,
                    "bucket": role,
                    "target_origin": "pantograph_verified_h0_generation",
                    "selection_seed": SELECTION_SEED,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_rank": candidate["candidate_rank"],
                    "candidate_seed": candidate["candidate_seed"],
                    "statement_successes_at_k8": len(success_by_group[group]),
                    "checkpoint_hash": candidate["checkpoint_hash"],
                    "generation_config_hash": candidate["generation_config_hash"],
                    "candidate_attestation_id": attestation_id,
                },
            }
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        labels = full_ids[len(prompt_ids) :]
        row["input_tokens"] = len(prompt_ids)
        row["label_tokens"] = len(labels)
        row["total_tokens"] = len(full_ids)
        row["zero_label"] = not bool(labels)
        row["truncated"] = len(full_ids) > 1024
        row["ei_round0"].update(
            {
                "valid_label_count_with_eos": len(labels),
                "last_valid_label_id": labels[-1] if labels else None,
            }
        )
        return row

    rows = [materialize(group, "Success Bank") for group in success_groups]
    rows += [materialize(group, "Frontier replay") for group in frontier_groups]
    rows.sort(key=lambda row: stable_rank(theorem_group(row), "manifest-order"))
    roles = Counter(row["ei_round0"]["top_level_role"] for row in rows)
    duplicates = duplicate_audit(rows)
    expected_duplicates = {
        "rows": EXPECTED_ROWS,
        "duplicate_rows": 0,
        "duplicate_record_ids": 0,
        "theorem_group_duplicates": 0,
        "max_repeat": 1,
    }
    if dict(roles) != {"Success Bank": EXPECTED_SUCCESS, "Frontier replay": EXPECTED_FRONTIER}:
        raise RuntimeError(f"composition drifted: {roles}")
    if duplicates != expected_duplicates:
        raise RuntimeError(f"duplicate gate failed: {duplicates}")
    if not all(
        row["pantograph_verified"]
        and row["proof_verified"]
        and row["ei_round0"]["target_origin"] == "pantograph_verified_h0_generation"
        for row in rows
    ):
        raise RuntimeError("unverified or reference target entered EI manifest")

    h0_rows = read_jsonl(paths["h0_manifest"])
    prior_overlap = overlap(rows, h0_rows)
    if not prior_overlap["passed"]:
        raise RuntimeError(f"H0 training overlap: {prior_overlap}")
    protected = protected_sets(project)
    leakage = {name: overlap(rows, values) for name, values in protected.items()}
    if not leakage or not all(item["passed"] for item in leakage.values()):
        raise RuntimeError("protected evaluation leakage gate failed")

    eos = {
        "rows": len(rows),
        "completion_nonempty": sum(bool(row["completion"].removesuffix(eos_token).strip()) for row in rows),
        "valid_label_gt_zero": sum(row["label_tokens"] > 0 for row in rows),
        "last_valid_label_is_eos": sum(row["ei_round0"]["last_valid_label_id"] == tokenizer.eos_token_id for row in rows),
        "eos_token_id": tokenizer.eos_token_id,
        "eos_is_minus_100": tokenizer.eos_token_id == -100,
        "zero_label": sum(row["zero_label"] for row in rows),
        "semantic_truncation": sum(row["truncated"] for row in rows),
        "min_valid_labels": min(row["label_tokens"] for row in rows),
        "max_valid_labels": max(row["label_tokens"] for row in rows),
        "max_total_tokens": max(row["total_tokens"] for row in rows),
    }
    if (
        any(eos[key] != EXPECTED_ROWS for key in ("completion_nonempty", "valid_label_gt_zero", "last_valid_label_is_eos"))
        or eos["eos_is_minus_100"]
        or eos["zero_label"]
        or eos["semantic_truncation"]
    ):
        raise RuntimeError(f"EOS gate failed: {eos}")

    write_jsonl(manifest_path, rows)
    gate = {
        "status": "PASSED_APPROVED_250_ROW_RELAXATION",
        "passed": True,
        "approval": {
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_change": "training row minimum relaxed from 500 to 250",
            "unchanged_contracts": [
                "80/20 source mix",
                "Pantograph-verified generated targets only",
                "theorem-group uniqueness",
                "EOS supervision",
                "data/evaluation isolation",
                "all SFT hyperparameters",
            ],
        },
        "rows": len(rows),
        "role_counts": dict(roles),
        "source_counts": dict(Counter(row["sampling_source"] for row in rows)),
        "selection_seed": SELECTION_SEED,
        "manifest_sha256": sha256(manifest_path),
        "h0_checkpoint_sha256": EXPECTED_H0_MODEL_HASH,
        "discovery_generation_config_sha256": EXPECTED_DISCOVERY_CONTRACT_HASH,
        "duplicates": duplicates,
        "prior_h0_training_overlap": prior_overlap,
        "protected_evaluation_leakage": leakage,
        "eos": eos,
        "eligible_unique_success_groups": len(eligible),
        "partial_success_groups_eligible": len(partial_groups),
        "solved_easy_groups_eligible": len(easy_groups),
        "target_origin_counts": dict(Counter(row["ei_round0"]["target_origin"] for row in rows)),
    }
    write_json(train_root / "manifest_gate.json", gate)
    training_contract = {
        "experiment": "Expert Iteration Round 0",
        "model_name": "EI-Round0",
        "status": "READY_FOR_TRAINING_APPROVED_250_ROWS",
        "starting_checkpoint": "H0-No-Hard merged checkpoint",
        "initialization": "fresh LoRA on frozen merged H0",
        "expected_rows": EXPECTED_ROWS,
        "expected_optimizer_steps": math.ceil(EXPECTED_ROWS / 16),
        "expected_role_counts": {"Success Bank": EXPECTED_SUCCESS, "Frontier replay": EXPECTED_FRONTIER},
        "manifest_gate_path": str(train_root / "manifest_gate.json"),
        "resolved_config": {
            "model_name_or_path": str(paths["h0_model"]),
            "adapter_path": None,
            "train_file": str(manifest_path),
            "validation_file": str(paths["validation"]),
            "output_dir": str(root / "checkpoint/trainer"),
            "prompt_field": "prompt",
            "completion_field": "completion",
            "sample_weight_field": None,
            "sampling_strategy": "fixed_manifest_without_replacement",
            "requested_packing": False,
            "require_pantograph_verified": True,
            "require_supervised_eos": True,
            "max_seq_length": 1024,
            "allow_overlength": False,
            "per_device_train_batch_size": 1,
            "per_device_validation_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "dataloader_drop_last": False,
            "dataloader_num_workers": 0,
            "max_steps": -1,
            "learning_rate": 1e-5,
            "num_train_epochs": 1.0,
            "logging_steps": 4,
            "eval_steps": None,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_steps": math.ceil(EXPECTED_ROWS / 16),
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "warmup_ratio": 0.03,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "device_map": "auto",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "seed": 42,
            "data_seed": 42,
        },
        "canonical_generation_contract": {
            "path": str(paths["generation_contract"]),
            "generation_config_sha256": EXPECTED_DISCOVERY_CONTRACT_HASH,
        },
    }
    write_json(root / "training_manifest/training_contract.json", training_contract)
    status = read_json(root / "status.json")
    status.update(
        {
            "status": "TRAINING_MANIFEST_READY_APPROVED_250_ROWS",
            "trainer_started": False,
            "grpo_started": False,
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": gate["manifest_sha256"],
            "training_rows": EXPECTED_ROWS,
        }
    )
    write_json(root / "status.json", status)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
