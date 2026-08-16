"""Freeze H0/H1/H2 manifests for the Hard-A proportion ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
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


NEW_WB_SELECTION_SEED = 20260801
ABLATION_SELECTION_SEED = 20260803
TRAINING_SEED = 42
EXPECTED_M0_MODEL_HASH = "6d80ac7b0034e46f51554f7ba4b66609e83dc9b9ce9c6c643fcbbcfabc5b6f81"
EXPECTED_C2_MANIFEST_HASH = "0883bf536188798c03c87789f34c35cb8367240e14b0f4b5a66c844e649e188a"


def stable_rank(value: str, namespace: str, seed: int = ABLATION_SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def new_wb_rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{NEW_WB_SELECTION_SEED}:{record_id(row)}".encode()).hexdigest()


def materialize_effective(
    rows: list[dict[str, Any]], tokenizer: Any, annotation: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    eos_token, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
    effective: list[dict[str, Any]] = []
    counts = Counter()
    label_lengths: list[int] = []
    modified = 0
    for source in rows:
        original = proof(source)
        completion = original
        while completion.endswith(eos_token):
            completion = completion[:-len(eos_token)]
        if completion.strip() != original.strip():
            modified += 1
        prompt = str(source.get("prompt") or "")
        if not prompt.strip() or not completion.strip():
            raise RuntimeError(f"empty prompt/completion: {record_id(source)}")
        row = dict(source)
        row["hard_a_ablation"] = {**annotation, "top_level_role": source["_ablation_role"], "bucket": source["_ablation_role"]}
        row.pop("_ablation_role", None)
        row["completion"] = completion + eos_token
        row["text"] = prompt + row["completion"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"prompt prefix mismatch: {record_id(source)}")
        labels = full_ids[len(prompt_ids):]
        counts["completion_nonempty"] += bool(completion.strip())
        counts["valid_label_gt_zero"] += bool(labels)
        counts["last_valid_label_is_eos"] += bool(labels and labels[-1] == eos_id)
        counts["eos_supervised"] += bool(labels and eos_id in labels and labels[-1] == eos_id)
        counts["zero_label"] += not labels
        counts["semantic_truncation"] += len(full_ids) > 1024
        label_lengths.append(len(labels))
        row["hard_a_ablation"].update({"valid_label_count_with_eos": len(labels), "last_valid_label_id": labels[-1] if labels else None})
        effective.append(row)
    eos = {
        "status": "PASSED", "rows": len(rows),
        "completion_nonempty": counts["completion_nonempty"], "valid_label_gt_zero": counts["valid_label_gt_zero"],
        "last_valid_label_is_eos": counts["last_valid_label_is_eos"], "eos_supervised": counts["eos_supervised"],
        "eos_token_id": eos_id, "eos_is_minus_100": False, "zero_label": counts["zero_label"],
        "semantic_truncation": counts["semantic_truncation"], "min_valid_labels": min(label_lengths), "max_valid_labels": max(label_lengths),
    }
    if any(eos[key] != len(rows) for key in ("completion_nonempty", "valid_label_gt_zero", "last_valid_label_is_eos", "eos_supervised")) or eos["zero_label"] or eos["semantic_truncation"] or modified:
        raise RuntimeError(f"EOS/proof gate failed: {eos}; modified={modified}")
    return effective, eos, modified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation"
    arms = {"H0": root / "H0_no_hard", "H1": root / "H1_10pct_hard", "H2": root / "H2_20pct_hard"}
    if any((path / "manifest.jsonl").exists() for path in arms.values()):
        raise FileExistsError("refusing to overwrite an existing Hard-A ablation manifest")
    paths = {
        "m0_training": project / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "frozen_new_wb1000": project / "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Extra-WB1000.jsonl",
        "verified_reserve": project / "data/processed/lean_workbook_verified_v2/audit/verified_records.jsonl",
        "classification": project / "outputs/task0b_light/classification_manifest.jsonl",
        "hard_reclassification": project / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json",
        "protected_index": project / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "generation_contract": project / "outputs/task0b_light/generation_contract.json",
        "m0_merged": project / "outputs/stage2_sft_incremental_ablation/shared/checkpoints/M0-ADDON-B-FROZEN-MERGED",
        "validation": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
        "c2_root": project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if sha256(paths["m0_merged"] / "model.safetensors") != EXPECTED_M0_MODEL_HASH:
        raise RuntimeError("M0 model drifted")
    if sha256(paths["c2_root"] / "manifest.jsonl") != EXPECTED_C2_MANIFEST_HASH:
        raise RuntimeError("Phase C2/H1 manifest drifted")
    if read_json(paths["generation_contract"])["generation_config_sha256"] != EXPECTED_GENERATION_CONFIG_HASH:
        raise RuntimeError("generation contract drifted")

    m0_training = read_jsonl(paths["m0_training"])
    classifications = read_jsonl(paths["classification"])
    hard = read_json(paths["hard_reclassification"])
    frozen_new_wb = read_jsonl(paths["frozen_new_wb1000"])
    reserve_raw = read_jsonl(paths["verified_reserve"])
    c2_source = read_jsonl(paths["c2_root"] / "source_manifest.jsonl")
    c2_effective = read_jsonl(paths["c2_root"] / "manifest.jsonl")
    if (len(m0_training), len(classifications), len(frozen_new_wb), len(reserve_raw), len(c2_source), len(c2_effective)) != (3000, 3000, 1000, 7581, 2000, 2000):
        raise RuntimeError("source row counts drifted")

    protected_rows: dict[str, list[dict[str, Any]]] = {}
    retention: list[dict[str, Any]] = []
    for name, entry in read_json(paths["protected_index"]).items():
        p = Path(str(entry.get("path") or ""))
        if not p.is_file():
            continue
        rows = read_jsonl(p)
        if entry.get("role") == "retention":
            retention.extend(rows)
        elif entry.get("role") in {"true_holdout", "future_true_holdout"}:
            protected_rows[name] = rows
    all_protected = [row for values in protected_rows.values() for row in values]
    protected_sets = identity_sets(all_protected)
    first_round_sets = identity_sets(m0_training)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(paths["m0_merged"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("tokenizer drifted")

    # Rebuild a common nested New-WB pool of 1800 rows using the same ordering
    # as C1/C2, then derive H1=first1600 and H2=first1603.
    reserve: list[dict[str, Any]] = []
    for raw in reserve_raw:
        row = materialize_verified_reserve_row(raw)
        full_ids = tokenizer(str(row["prompt"]) + str(row["completion"]), add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=True)["input_ids"]
        row["input_tokens"], row["label_tokens"], row["total_tokens"] = len(prompt_ids), len(full_ids) - len(prompt_ids), len(full_ids)
        if full_ids[:len(prompt_ids)] != prompt_ids or row["label_tokens"] <= 0 or row["total_tokens"] > 1023:
            continue
        if candidate_is_verified(row) and disjoint_from_sets(row, first_round_sets) and disjoint_from_sets(row, protected_sets):
            reserve.append(row)
    common_new_wb = list(frozen_new_wb)
    common_sets = identity_sets(common_new_wb)
    for row in sorted(reserve, key=new_wb_rank):
        if disjoint_from_sets(row, common_sets):
            common_new_wb.append(row)
            add_identity(row, common_sets)
        if len(common_new_wb) == 1800:
            break
    if len(common_new_wb) != 1800:
        raise RuntimeError("New-WB1800 unavailable")
    c2_new_ids = {record_id(row) for row in c2_source if row.get("stage2_phase_c2", {}).get("top_level_role") == "New WB"}
    if {record_id(row) for row in common_new_wb[:1600]} != c2_new_ids:
        raise RuntimeError("H1 New-WB1600 does not match Phase C2")

    train_by_id = {record_id(row): row for row in m0_training}
    hard_records = list(hard["records"])
    hard_a_all = sorted((record_id(row) for row in hard_records if row.get("hard_bucket") == "Hard-A"), key=lambda rid: stable_rank(rid, "hard-a", NEW_WB_SELECTION_SEED))
    hard_a_clean = [rid for rid in hard_a_all if overlap([train_by_id[rid]], all_protected)["passed"]]
    if len(hard_a_all) != 399 or len(hard_a_clean) != 397:
        raise RuntimeError("Hard-A raw/protected-clean pool drifted")
    c2_hard_ids = {record_id(row) for row in c2_source if row.get("stage2_phase_c2", {}).get("top_level_role") == "Hard-A"}
    if set(hard_a_clean[:200]) != c2_hard_ids:
        raise RuntimeError("H1 Hard-A200 does not match Phase C2")
    stable_rows = [row for row in c2_source if row.get("stage2_phase_c2", {}).get("top_level_role") == "Stable/Core"]
    frontier_rows = [row for row in c2_source if row.get("stage2_phase_c2", {}).get("top_level_role") == "Frontier"]
    additional_rows = [row for row in c2_source if row.get("stage2_phase_c2", {}).get("top_level_role") == "Additional verified WB replay"]
    if (len(stable_rows), len(frontier_rows), len(additional_rows)) != (12, 47, 141):
        raise RuntimeError("H0/H1 frozen replay counts drifted")

    def tagged(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        output = []
        for item in rows:
            row = dict(item)
            row["_ablation_role"] = role
            output.append(row)
        return output

    arm_sources = {
        "H0": tagged(common_new_wb[:1800], "New WB") + tagged(stable_rows, "Stable/Core") + tagged(frontier_rows, "Frontier") + tagged(additional_rows, "Additional verified WB replay"),
        "H2": tagged(common_new_wb[:1603], "New WB") + tagged([train_by_id[rid] for rid in hard_a_clean], "Hard-A"),
    }
    expected_counts = {
        "H0": {"New WB": 1800, "Stable/Core": 12, "Frontier": 47, "Additional verified WB replay": 141},
        "H1": {"New WB": 1600, "Hard-A": 200, "Stable/Core": 12, "Frontier": 47, "Additional verified WB replay": 141},
        "H2": {"New WB": 1603, "Hard-A": 397},
    }
    model_names = {"H0": "H0-No-Hard", "H1": "H1-10pct-Hard-A", "H2": "H2-20pct-Hard-A"}
    for arm in ("H0", "H2"):
        source_rows = arm_sources[arm]
        source_rows.sort(key=lambda row: stable_rank(record_id(row), f"{arm}-manifest-order"))
        roles = Counter(row["_ablation_role"] for row in source_rows)
        if len(source_rows) != 2000 or dict(roles) != expected_counts[arm]:
            raise RuntimeError(f"{arm} composition drifted: {roles}")
        effective, eos, modified = materialize_effective(source_rows, tokenizer, {
            "arm": arm, "model": model_names[arm], "selection_seed": ABLATION_SELECTION_SEED,
            "new_wb_lineage_seed": NEW_WB_SELECTION_SEED,
        })
        clean_source = []
        for row in source_rows:
            item = dict(row); item.pop("_ablation_role", None); clean_source.append(item)
        duplicates = duplicate_audit(clean_source)
        if duplicates != {"rows": 2000, "duplicate_rows": 0, "duplicate_record_ids": 0, "theorem_group_duplicates": 0, "max_repeat": 1}:
            raise RuntimeError(f"{arm} duplicate gate failed: {duplicates}")
        leakage = {name: overlap(clean_source, values) for name, values in protected_rows.items()}
        if not all(value["passed"] for value in leakage.values()):
            raise RuntimeError(f"{arm} protected leakage gate failed")
        arm_root = arms[arm]
        source_path, manifest_path = arm_root / "source_manifest.jsonl", arm_root / "manifest.jsonl"
        write_jsonl(source_path, clean_source); write_jsonl(manifest_path, effective)
        leakage_audit = {"status": "PASSED", "checks": list(identity({}).keys()), "protected": leakage, "wb_train_retention150_allowed_overlap": overlap(clean_source, retention)}
        data_audit = {
            "arm": arm, "model_name": model_names[arm], "status": "PASSED_READY_FOR_TRAINING", "rows": 2000,
            "role_counts": dict(roles), "hard_a_rows": roles.get("Hard-A", 0), "hard_a_fraction": roles.get("Hard-A", 0) / 2000,
            "nominal_hard_a_fraction": 0.0 if arm == "H0" else 0.20,
            "h2_protected_gate_adjustment": "399 nominal -> 397 protected-clean Hard-A; New WB 1601 -> 1603" if arm == "H2" else None,
            "duplicates": duplicates, "proof_modifications": modified, "all_pantograph_verified": all(row.get("pantograph_verified") is True for row in clean_source),
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "source_manifest_sha256": sha256(source_path),
        }
        write_json(arm_root / "data_audit.json", data_audit); write_json(arm_root / "leakage_audit.json", leakage_audit); write_json(arm_root / "eos_audit.json", eos)
        gate = {
            "passed": True, "arm": arm, "rows": 2000, "effective_manifest": str(manifest_path), "effective_manifest_sha256": sha256(manifest_path),
            "source_manifest_sha256": sha256(source_path), "m0_model_sha256": EXPECTED_M0_MODEL_HASH, "generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH,
            "role_counts": dict(roles), "duplicates": duplicates, "leakage": leakage_audit, "eos": eos, "proof_modifications": modified,
            "forbidden_buckets": {"Hard-B": 0, "Hard-C": 0, "unclassified_hard": 0},
        }
        gate_path = arm_root / "audit/manifest_gate.json"; write_json(gate_path, gate)
        contract = {
            "phase": arm, "model_name": model_names[arm], "status": "READY_FOR_TRAINING",
            "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA", "resume_from_checkpoint": False,
            "optimizer_state_shared": False, "scheduler_state_shared": False, "adapter_state_shared": False,
            "expected_rows": 2000, "expected_optimizer_steps": math.ceil(2000 / 16),
            "manifest_gate_path": str(gate_path), "annotation_key": "hard_a_ablation", "expected_role_counts": dict(roles), "expected_bucket_counts": dict(roles),
            "resolved_config": {
                "model_name_or_path": str(paths["m0_merged"]), "adapter_path": None, "train_file": str(manifest_path), "validation_file": str(paths["validation"]),
                "output_dir": str(arm_root / "training/trainer"), "prompt_field": "prompt", "completion_field": "completion", "sample_weight_field": None,
                "sampling_strategy": "fixed_manifest_without_replacement", "requested_packing": False, "require_pantograph_verified": True, "require_supervised_eos": True,
                "max_seq_length": 1024, "allow_overlength": False, "per_device_train_batch_size": 1, "per_device_validation_batch_size": 1,
                "gradient_accumulation_steps": 16, "dataloader_drop_last": False, "dataloader_num_workers": 0, "max_steps": -1,
                "learning_rate": 1e-5, "num_train_epochs": 1.0, "logging_steps": 10, "eval_steps": None, "eval_strategy": "epoch", "save_strategy": "epoch",
                "save_steps": 125, "save_total_limit": 2, "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
                "warmup_ratio": 0.03, "warmup_steps": 0, "weight_decay": 0.01, "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
                "gradient_checkpointing": True, "optim": "paged_adamw_8bit", "device_map": "auto", "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
                "seed": TRAINING_SEED, "data_seed": TRAINING_SEED,
            },
            "manifest_gate_sha256": sha256(gate_path), "canonical_generation_contract": {"path": str(paths["generation_contract"]), "generation_config_sha256": EXPECTED_GENERATION_CONFIG_HASH},
        }
        write_json(arm_root / "training_contract.json", contract)
        write_json(arm_root / "evaluation/status.json", {"status": "NOT_STARTED_WAITING_FOR_TRAINING", "arm": arm, "phase_c3_started": False})
        (arm_root / "report.md").write_text(f"# {arm} Hard-A ablation\n\n**Status: data gates passed; training pending.**\n", encoding="utf-8")

    # H1 is the exact independently trained Phase-C2 arm; preserve bytes and
    # provenance instead of re-running an identical deterministic experiment.
    h1_root = arms["H1"]; h1_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["c2_root"] / "manifest.jsonl", h1_root / "manifest.jsonl")
    shutil.copy2(paths["c2_root"] / "source_manifest.jsonl", h1_root / "source_manifest.jsonl")
    c2_data = read_json(paths["c2_root"] / "data_audit.json")
    h1_data = {**c2_data, "arm": "H1", "model_name": model_names["H1"], "status": "PASSED_EQUIVALENT_TO_PHASE_C2", "equivalent_phase_c2_manifest_sha256": EXPECTED_C2_MANIFEST_HASH}
    write_json(h1_root / "data_audit.json", h1_data)
    shutil.copy2(paths["c2_root"] / "leakage_audit.json", h1_root / "leakage_audit.json")
    shutil.copy2(paths["c2_root"] / "eos_audit.json", h1_root / "eos_audit.json")
    write_json(h1_root / "training_contract.json", {
        "arm": "H1", "model_name": model_names["H1"], "status": "COMPLETED_EQUIVALENT_TO_PHASE_C2",
        "source_phase": "C2", "source_root": str(paths["c2_root"]), "manifest_sha256": EXPECTED_C2_MANIFEST_HASH,
        "independent_initialization": "fresh_lora_on_frozen_merged_m0", "starting_checkpoint_model_sha256": EXPECTED_M0_MODEL_HASH,
        "generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH,
    })
    write_json(h1_root / "evaluation/status.json", {"status": "PENDING_ARCHIVE_FROM_EQUIVALENT_PHASE_C2", "arm": "H1", "phase_c3_started": False})
    (h1_root / "report.md").write_text("# H1 — 10% Hard-A\n\nExact manifest/training/evaluation equivalent of completed Phase C2; archive materialization pending.\n", encoding="utf-8")

    # Cross-arm identity audit verifies the intended controlled replacements.
    ids = {
        "H0": {record_id(row) for row in read_jsonl(arms["H0"] / "manifest.jsonl")},
        "H1": {record_id(row) for row in c2_effective},
        "H2": {record_id(row) for row in read_jsonl(arms["H2"] / "manifest.jsonl")},
    }
    design = {
        "status": "PASSED_READY_FOR_H0_H1_H2",
        "arm_manifest_sha256": {arm: sha256(arms[arm] / "manifest.jsonl") for arm in arms},
        "arm_counts": expected_counts,
        "pairwise_identity": {
            "H0_vs_H1": {"shared": len(ids["H0"] & ids["H1"]), "H0_only": len(ids["H0"] - ids["H1"]), "H1_only": len(ids["H1"] - ids["H0"])},
            "H1_vs_H2": {"shared": len(ids["H1"] & ids["H2"]), "H1_only": len(ids["H1"] - ids["H2"]), "H2_only": len(ids["H2"] - ids["H1"])},
        },
        "h1_equivalent_to_phase_c2": sha256(arms["H1"] / "manifest.jsonl") == EXPECTED_C2_MANIFEST_HASH,
        "h2_nominal_rows": {"Hard-A": 399, "New WB": 1601},
        "h2_effective_protected_clean_rows": {"Hard-A": 397, "New WB": 1603},
        "trainer_started": False, "phase_c3_started": False,
    }
    if design["pairwise_identity"] != {"H0_vs_H1": {"shared": 1800, "H0_only": 200, "H1_only": 200}, "H1_vs_H2": {"shared": 1800, "H1_only": 200, "H2_only": 200}}:
        raise RuntimeError(f"cross-arm controlled replacement drifted: {design['pairwise_identity']}")
    write_json(root / "ablation_design_audit.json", design)
    write_json(root / "status.json", {"status": "MANIFESTS_READY_H0_NEXT", "h0_started": False, "h1_archived": False, "h2_started": False, "phase_c3_started": False})
    print(json.dumps(design, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
