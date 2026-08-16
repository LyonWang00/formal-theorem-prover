"""Build and hard-gate the approved 2,000-row Phase C2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    EXPECTED_ENVIRONMENT_HASH,
    EXPECTED_GENERATION_CONFIG_HASH,
    add_identity,
    candidate_is_verified,
    disjoint_from_sets,
    duplicate_audit,
    identity,
    identity_sets,
    materialize_verified_reserve_row,
    overlap,
    proof,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    write_json,
    write_jsonl,
)


NEW_WB_SELECTION_SEED = 20260801  # preserves Phase C1 New-WB1200 as a prefix
PHASE_C2_SELECTION_SEED = 20260802
TRAINING_SEED = 42
EXPECTED_M0_MODEL_HASH = "6d80ac7b0034e46f51554f7ba4b66609e83dc9b9ce9c6c643fcbbcfabc5b6f81"
EXPECTED_C1_NEW_WB_1200_HASH = "b4bef86af5438765309c218740a1ab6eb448398040338bcf875ad5e1ac62ea20"


def stable_rank(value: str, namespace: str, seed: int = PHASE_C2_SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def c1_reserve_rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{NEW_WB_SELECTION_SEED}:{record_id(row)}".encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--phase-root", type=Path, default=Path("outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.phase_root if args.phase_root.is_absolute() else project / args.phase_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase C2 manifest: {manifest_path}")
    paths = {
        "m0_training": project / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "frozen_new_wb1000": project / "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Extra-WB1000.jsonl",
        "verified_reserve": project / "data/processed/lean_workbook_verified_v2/audit/verified_records.jsonl",
        "classification": project / "outputs/task0b_light/classification_manifest.jsonl",
        "hard_reclassification": project / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json",
        "c1_new_wb1200": project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/preflight_new_wb_1200.jsonl",
        "generation_contract": project / "outputs/task0b_light/generation_contract.json",
        "protected_index": project / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "m0_merged": project / "outputs/stage2_sft_incremental_ablation/shared/checkpoints/M0-ADDON-B-FROZEN-MERGED",
        "validation": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase C2 inputs missing: {missing}")
    if sha256(paths["c1_new_wb1200"]) != EXPECTED_C1_NEW_WB_1200_HASH:
        raise RuntimeError("frozen Phase C1 New-WB1200 drifted")
    if sha256(paths["m0_merged"] / "model.safetensors") != EXPECTED_M0_MODEL_HASH:
        raise RuntimeError("frozen M0 checkpoint drifted")

    m0_training = read_jsonl(paths["m0_training"])
    frozen_new_wb = read_jsonl(paths["frozen_new_wb1000"])
    reserve_raw = read_jsonl(paths["verified_reserve"])
    classifications = read_jsonl(paths["classification"])
    hard = read_json(paths["hard_reclassification"])
    c1_new_wb = read_jsonl(paths["c1_new_wb1200"])
    generation_contract = read_json(paths["generation_contract"])
    protected_index = read_json(paths["protected_index"])
    if (len(m0_training), len(frozen_new_wb), len(reserve_raw), len(classifications), len(c1_new_wb)) != (3000, 1000, 7581, 3000, 1200):
        raise RuntimeError("Phase C2 frozen source counts drifted")
    if generation_contract.get("generation_config_sha256") != EXPECTED_GENERATION_CONFIG_HASH:
        raise RuntimeError("canonical generation contract drifted")

    protected_rows: dict[str, list[dict[str, Any]]] = {}
    retention: list[dict[str, Any]] = []
    for name, entry in protected_index.items():
        path_value = entry.get("path")
        if not path_value or not Path(str(path_value)).is_file():
            continue
        rows = read_jsonl(Path(str(path_value)))
        if entry.get("role") == "retention":
            retention.extend(rows)
        elif entry.get("role") in {"true_holdout", "future_true_holdout"}:
            protected_rows[name] = rows
    all_protected = [row for rows in protected_rows.values() for row in rows]
    protected_sets = identity_sets(all_protected)
    first_round_sets = identity_sets(m0_training)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(paths["m0_merged"]), local_files_only=True, trust_remote_code=False)
    eos_token, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
    if eos_token != "<|im_end|>" or eos_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("frozen tokenizer identity drifted")

    # Preserve Phase C1 New-WB1200 exactly, then extend the same deterministic
    # reserve ordering by another 400 protected-clean, first-round-unused rows.
    reserve: list[dict[str, Any]] = []
    reserve_rejections = Counter()
    for raw in reserve_raw:
        row = materialize_verified_reserve_row(raw)
        full_ids = tokenizer(str(row["prompt"]) + str(row["completion"]), add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=True)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            reserve_rejections["prompt_prefix_mismatch"] += 1
            continue
        row["input_tokens"], row["label_tokens"], row["total_tokens"] = len(prompt_ids), len(full_ids) - len(prompt_ids), len(full_ids)
        if row["label_tokens"] <= 0:
            reserve_rejections["zero_label"] += 1
            continue
        if row["total_tokens"] > 1023:
            reserve_rejections["overlength_with_supervised_eos"] += 1
            continue
        if not candidate_is_verified(row):
            reserve_rejections["verification_contract"] += 1
            continue
        if not disjoint_from_sets(row, first_round_sets) or not disjoint_from_sets(row, protected_sets):
            continue
        reserve.append(row)
    selected_new_wb = list(frozen_new_wb)
    new_sets = identity_sets(selected_new_wb)
    for row in sorted(reserve, key=c1_reserve_rank):
        if disjoint_from_sets(row, new_sets):
            selected_new_wb.append(row)
            add_identity(row, new_sets)
        if len(selected_new_wb) == 1600:
            break
    if len(selected_new_wb) != 1600:
        raise RuntimeError(f"New WB 1600 unavailable: {len(selected_new_wb)}")
    if {record_id(row) for row in selected_new_wb[:1200]} != {record_id(row) for row in c1_new_wb}:
        raise RuntimeError("Phase C2 New-WB prefix does not preserve Phase C1 New-WB1200")

    train_by_id = {record_id(row): row for row in m0_training}
    class_by_id = {record_id(row): row for row in classifications}
    if len(train_by_id) != 3000 or set(train_by_id) != set(class_by_id):
        raise RuntimeError("M0 training and Task0B classification mismatch")
    hard_records = list(hard.get("records") or [])
    hard_bucket_by_id = {record_id(row): row.get("hard_bucket") for row in hard_records}
    hard_a_all = sorted((rid for rid, bucket in hard_bucket_by_id.items() if bucket == "Hard-A"), key=lambda rid: stable_rank(rid, "hard-a", NEW_WB_SELECTION_SEED))
    stable_all = sorted((record_id(row) for row in classifications if row.get("final_bucket") == "stable_core"), key=lambda rid: stable_rank(rid, "stable", NEW_WB_SELECTION_SEED))
    frontier_all = sorted((record_id(row) for row in classifications if row.get("final_bucket") == "frontier"), key=lambda rid: stable_rank(rid, "frontier", NEW_WB_SELECTION_SEED))
    def protected_clean(rid: str) -> bool:
        return overlap([train_by_id[rid]], all_protected)["passed"]
    hard_a = [rid for rid in hard_a_all if protected_clean(rid)]
    stable = [rid for rid in stable_all if protected_clean(rid)]
    frontier = [rid for rid in frontier_all if protected_clean(rid)]
    if (len(hard_a), len(stable), len(frontier)) != (397, 12, 47):
        raise RuntimeError(f"protected-clean Task0B counts drifted: {len(hard_a)}/{len(stable)}/{len(frontier)}")
    selected_hard_a = hard_a[:200]

    occupied_ids = set(selected_hard_a) | set(stable) | set(frontier)
    selected_sets = identity_sets(selected_new_wb + [train_by_id[rid] for rid in occupied_ids])
    additional_candidates = []
    for row in m0_training:
        rid = record_id(row)
        classification = class_by_id[rid]
        if row.get("sampling_source") != "WB" or classification.get("classification") != "unclassified_not_sampled":
            continue
        if rid in hard_bucket_by_id or rid in occupied_ids:
            continue
        if not candidate_is_verified(row) or int(row.get("total_tokens") or 0) > 1023:
            continue
        if not disjoint_from_sets(row, protected_sets):
            continue
        additional_candidates.append(row)
    additional_replay: list[dict[str, Any]] = []
    for row in sorted(additional_candidates, key=lambda x: stable_rank(record_id(x), "additional-wb-replay")):
        if disjoint_from_sets(row, selected_sets):
            additional_replay.append(row)
            add_identity(row, selected_sets)
        if len(additional_replay) == 141:
            break
    if len(additional_replay) != 141:
        raise RuntimeError(f"Additional verified WB replay shortage: {len(additional_replay)}/141")

    specs: list[tuple[dict[str, Any], str, str]] = []
    specs += [(row, "New WB", "New WB") for row in selected_new_wb]
    specs += [(train_by_id[rid], "Hard-A", "Hard-A") for rid in selected_hard_a]
    specs += [(train_by_id[rid], "Stable/Core", "Stable/Core") for rid in stable]
    specs += [(train_by_id[rid], "Frontier", "Frontier") for rid in frontier]
    specs += [(row, "Additional verified WB replay", "Additional verified WB replay") for row in additional_replay]
    source_rows: list[dict[str, Any]] = []
    source_proofs: dict[str, str] = {}
    for original, role, bucket in specs:
        row = dict(original)
        row["stage2_phase_c2"] = {
            "top_level_role": role, "bucket": bucket,
            "phase_c2_selection_seed": PHASE_C2_SELECTION_SEED,
            "new_wb_lineage_seed": NEW_WB_SELECTION_SEED,
            "user_approved_composition": True,
        }
        row["text"] = str(row.get("prompt") or "") + proof(row)
        source_rows.append(row)
        source_proofs[record_id(row)] = proof(original)
    source_rows.sort(key=lambda row: stable_rank(record_id(row), "manifest-order"))
    if len(source_rows) != 2000 or len(source_proofs) != 2000:
        raise RuntimeError("Phase C2 source manifest is not 2000 unique rows")

    effective_rows: list[dict[str, Any]] = []
    eos_counts = Counter()
    valid_label_counts: list[int] = []
    for source_row in source_rows:
        rid = record_id(source_row)
        completion = proof(source_row)
        while completion.endswith(eos_token):
            completion = completion[:-len(eos_token)]
        if completion.strip() != source_proofs[rid]:
            raise RuntimeError(f"proof changed before EOS materialization: {rid}")
        prompt = str(source_row.get("prompt") or "")
        if not prompt.strip() or not completion.strip():
            raise RuntimeError(f"empty prompt/completion: {rid}")
        row = dict(source_row)
        row["completion"] = completion + eos_token
        row["text"] = prompt + row["completion"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"prompt prefix mismatch: {rid}")
        labels = full_ids[len(prompt_ids):]
        eos_counts["completion_nonempty"] += bool(completion.strip())
        eos_counts["valid_label_gt_zero"] += bool(labels)
        eos_counts["last_valid_label_is_eos"] += bool(labels and labels[-1] == eos_id)
        eos_counts["eos_supervised"] += bool(labels and eos_id in labels and labels[-1] == eos_id)
        eos_counts["semantic_truncation"] += len(full_ids) > 1024
        eos_counts["zero_label"] += not labels
        valid_label_counts.append(len(labels))
        row["stage2_phase_c2"] = dict(row["stage2_phase_c2"])
        row["stage2_phase_c2"].update({"valid_label_count_with_eos": len(labels), "last_valid_label_id": labels[-1] if labels else None})
        effective_rows.append(row)

    duplicates = duplicate_audit(source_rows)
    expected_duplicates = {"rows": 2000, "duplicate_rows": 0, "duplicate_record_ids": 0, "theorem_group_duplicates": 0, "max_repeat": 1}
    if duplicates != expected_duplicates:
        raise RuntimeError(f"Phase C2 duplicate gate failed: {duplicates}")
    proof_modifications = sum(proof(row).removesuffix(eos_token).strip() != source_proofs[record_id(row)] for row in effective_rows)
    eos_audit = {
        "status": "PASSED", "rows": 2000,
        "completion_nonempty": eos_counts["completion_nonempty"], "valid_label_gt_zero": eos_counts["valid_label_gt_zero"],
        "last_valid_label_is_eos": eos_counts["last_valid_label_is_eos"], "eos_supervised": eos_counts["eos_supervised"],
        "eos_token_id": eos_id, "eos_is_minus_100": False, "zero_label": eos_counts["zero_label"],
        "semantic_truncation": eos_counts["semantic_truncation"], "min_valid_labels": min(valid_label_counts), "max_valid_labels": max(valid_label_counts),
    }
    if any((eos_audit[k] != 2000 for k in ("completion_nonempty", "valid_label_gt_zero", "last_valid_label_is_eos", "eos_supervised"))) or eos_audit["zero_label"] or eos_audit["semantic_truncation"] or proof_modifications:
        raise RuntimeError(f"Phase C2 EOS/proof gate failed: {eos_audit}; proofs={proof_modifications}")
    leakage = {name: overlap(source_rows, rows) for name, rows in protected_rows.items()}
    if not all(result["passed"] for result in leakage.values()):
        write_json(root / "audit/preparation_failure.json", {"status": "BLOCKED_PROTECTED_LEAKAGE", "leakage": leakage, "trainer_started": False})
        raise RuntimeError("Phase C2 protected leakage gate failed")
    leakage_audit = {
        "status": "PASSED", "checks": list(identity({}).keys()), "protected": leakage,
        "wb_train_retention150_allowed_overlap": overlap(source_rows, retention),
        "ld_medium_holdout": {"status": "not_yet_constructed_phase3_only", "phase_c2_contains_ld_medium": False},
    }

    source_path = root / "source_manifest.jsonl"
    write_jsonl(source_path, source_rows)
    write_jsonl(manifest_path, effective_rows)
    role_counts = Counter(row["stage2_phase_c2"]["top_level_role"] for row in source_rows)
    expected_roles = {"New WB": 1600, "Hard-A": 200, "Stable/Core": 12, "Frontier": 47, "Additional verified WB replay": 141}
    if dict(role_counts) != expected_roles:
        raise RuntimeError(f"Phase C2 composition drifted: {role_counts}")
    data_audit = {
        "phase": "C2", "model_name": "S2-Balanced-Hard", "status": "PASSED_READY_FOR_TRAINING",
        "user_approved_composition": expected_roles, "rows": 2000, "role_counts": dict(role_counts), "bucket_counts": dict(role_counts),
        "new_wb": {"rows": 1600, "phase_c1_rows_preserved": 1200, "additional_rows": 400, "phase_c1_id_overlap": len({record_id(x) for x in selected_new_wb} & {record_id(x) for x in c1_new_wb}), "reserve_rejections": dict(reserve_rejections)},
        "additional_verified_wb_replay": {"eligible_candidates": len(additional_candidates), "selected": 141, "source": "first-round WB with Task0B classification=unclassified_not_sampled", "pantograph_verified": all(candidate_is_verified(row) for row in additional_replay)},
        "hard_a": {"protected_clean_pool": len(hard_a), "selected": 200, "failure_contract": "Task0B-Analysis strong near-miss"},
        "duplicates": duplicates, "proof_modifications": proof_modifications,
        "all_pantograph_verified": all(row.get("pantograph_verified") is True for row in source_rows),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "source_manifest": str(source_path), "source_manifest_sha256": sha256(source_path),
        "selection_seeds": {"new_wb_lineage": NEW_WB_SELECTION_SEED, "phase_c2": PHASE_C2_SELECTION_SEED}, "training_seed": TRAINING_SEED,
    }
    write_json(root / "data_audit.json", data_audit)
    write_json(root / "leakage_audit.json", leakage_audit)
    write_json(root / "eos_audit.json", eos_audit)
    gate = {
        "passed": True, "phase": "C2", "rows": 2000, "effective_manifest": str(manifest_path), "effective_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": sha256(source_path), "m0_model_sha256": EXPECTED_M0_MODEL_HASH,
        "generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH, "role_counts": dict(role_counts), "duplicates": duplicates,
        "leakage": leakage_audit, "eos": eos_audit, "proof_modifications": proof_modifications,
        "forbidden_buckets": {"Hard-B": 0, "Hard-C": 0, "unclassified_hard": 0},
    }
    gate_path = root / "audit/phaseC2_manifest_gate.json"
    write_json(gate_path, gate)
    training_contract = {
        "phase": "C2", "model_name": "S2-Balanced-Hard", "status": "READY_FOR_TRAINING",
        "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA", "resume_from_checkpoint": False,
        "optimizer_state_shared": False, "scheduler_state_shared": False, "adapter_state_shared": False,
        "expected_rows": 2000, "expected_optimizer_steps": math.ceil(2000 / 16),
        "resolved_config": {
            "model_name_or_path": str(paths["m0_merged"]), "adapter_path": None, "train_file": str(manifest_path), "validation_file": str(paths["validation"]),
            "output_dir": str(root / "training/trainer"), "prompt_field": "prompt", "completion_field": "completion", "sample_weight_field": None,
            "sampling_strategy": "fixed_manifest_without_replacement", "requested_packing": False, "require_pantograph_verified": True, "require_supervised_eos": True,
            "max_seq_length": 1024, "allow_overlength": False, "per_device_train_batch_size": 1, "per_device_validation_batch_size": 1,
            "gradient_accumulation_steps": 16, "dataloader_drop_last": False, "dataloader_num_workers": 0, "max_steps": -1,
            "learning_rate": 1e-5, "num_train_epochs": 1.0, "logging_steps": 10, "eval_steps": None, "eval_strategy": "epoch", "save_strategy": "epoch",
            "save_steps": 125, "save_total_limit": 2, "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
            "warmup_ratio": 0.03, "warmup_steps": 0, "weight_decay": 0.01, "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
            "gradient_checkpointing": True, "optim": "paged_adamw_8bit", "device_map": "auto", "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
            "seed": TRAINING_SEED, "data_seed": TRAINING_SEED,
        },
        "manifest_gate_sha256": sha256(gate_path),
        "canonical_generation_contract": {"path": str(paths["generation_contract"]), "generation_config_sha256": EXPECTED_GENERATION_CONFIG_HASH},
    }
    write_json(root / "training_contract.json", training_contract)
    write_json(root / "evaluation/status.json", {"status": "NOT_STARTED_WAITING_FOR_TRAINING", "canaries_started": False, "formal_evaluation_started": False, "phase_c3_started": False})
    (root / "report.md").write_text(
        "# Phase C2 — Balanced Hard\n\n**Status: data gates passed; training pending.**\n\n"
        "Approved composition: New WB 1600, Hard-A 200, Stable/Core 12, Frontier 47, and Additional verified WB replay 141. "
        "Hard-B/Hard-C/unclassified hard are absent; duplicate rows and theorem groups are zero; protected leakage is zero; all 2000 final labels supervise EOS.\n",
        encoding="utf-8",
    )
    print(json.dumps({"data_audit": data_audit, "eos": eos_audit, "leakage_passed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
