"""Build and hard-gate the user-approved 2,000-row Phase-C1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    EXPECTED_GENERATION_CONFIG_HASH,
    duplicate_audit,
    identity,
    overlap,
    proof,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    theorem_group,
    write_json,
    write_jsonl,
)


SELECTION_SEED = 20260801
TRAINING_SEED = 42
EXPECTED_M0_MODEL_HASH = (
    "6d80ac7b0034e46f51554f7ba4b66609e83dc9b9ce9c6c643fcbbcfabc5b6f81"
)
EXPECTED_NEW_WB_HASH = (
    "b4bef86af5438765309c218740a1ab6eb448398040338bcf875ad5e1ac62ea20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("outputs/stage2_data_ratio_ablation/phaseC1_new_wb"),
    )
    return parser.parse_args()


def stable_rank(value: str, namespace: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{namespace}:{value}".encode()
    ).hexdigest()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.phase_root if args.phase_root.is_absolute() else project / args.phase_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase-C1 manifest: {manifest_path}")

    paths = {
        "new_wb": root / "preflight_new_wb_1200.jsonl",
        "m0_training": project
        / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "classification": project / "outputs/task0b_light/classification_manifest.jsonl",
        "hard_reclassification": project
        / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json",
        "generation_contract": project / "outputs/task0b_light/generation_contract.json",
        "protected_index": project
        / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "m0_merged": project
        / "outputs/stage2_sft_incremental_ablation/shared/checkpoints/M0-ADDON-B-FROZEN-MERGED",
        "validation": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase-C1 inputs missing: {missing}")
    if sha256(paths["new_wb"]) != EXPECTED_NEW_WB_HASH:
        raise RuntimeError("frozen New-WB1200 preselection drifted")
    if sha256(paths["m0_merged"] / "model.safetensors") != EXPECTED_M0_MODEL_HASH:
        raise RuntimeError("frozen M0 merged checkpoint drifted")

    new_wb = read_jsonl(paths["new_wb"])
    m0_training = read_jsonl(paths["m0_training"])
    classifications = read_jsonl(paths["classification"])
    hard = read_json(paths["hard_reclassification"])
    generation_contract = read_json(paths["generation_contract"])
    protected_index = read_json(paths["protected_index"])
    if len(new_wb) != 1200 or len(m0_training) != 3000 or len(classifications) != 3000:
        raise RuntimeError("frozen Phase-C1 source counts drifted")
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

    train_by_id = {record_id(row): row for row in m0_training}
    class_by_id = {record_id(row): row for row in classifications}
    if len(train_by_id) != 3000 or set(train_by_id) != set(class_by_id):
        raise RuntimeError("M0 training and Task0B classification identity mismatch")
    hard_records = list(hard.get("records") or [])
    hard_a_ids_all = sorted(
        (
            record_id(row)
            for row in hard_records
            if row.get("hard_bucket") == "Hard-A"
        ),
        key=lambda value: stable_rank(value, "hard-a-split"),
    )
    hard_b_ids_all = sorted(
        (
            record_id(row)
            for row in hard_records
            if row.get("hard_bucket") == "Hard-B"
        ),
        key=lambda value: stable_rank(value, "hard-b-supplement"),
    )
    stable_ids = sorted(
        (
            record_id(row)
            for row in classifications
            if row.get("final_bucket") == "stable_core"
        ),
        key=lambda value: stable_rank(value, "stable"),
    )
    frontier_ids = sorted(
        (
            record_id(row)
            for row in classifications
            if row.get("final_bucket") == "frontier"
        ),
        key=lambda value: stable_rank(value, "frontier"),
    )
    if tuple(map(len, (hard_a_ids_all, hard_b_ids_all, stable_ids, frontier_ids))) != (
        399,
        349,
        12,
        47,
    ):
        raise RuntimeError("approved replay pool counts drifted")

    def protected_clean(rid: str) -> bool:
        return overlap([train_by_id[rid]], all_protected)["passed"]

    hard_a_ids = [rid for rid in hard_a_ids_all if protected_clean(rid)]
    hard_b_ids = [rid for rid in hard_b_ids_all if protected_clean(rid)]
    stable_ids = [rid for rid in stable_ids if protected_clean(rid)]
    frontier_ids = [rid for rid in frontier_ids if protected_clean(rid)]
    excluded_by_protected_gate = {
        "Hard-A": [rid for rid in hard_a_ids_all if rid not in set(hard_a_ids)],
        "Hard-B": [rid for rid in hard_b_ids_all if rid not in set(hard_b_ids)],
    }
    if tuple(map(len, (hard_a_ids, hard_b_ids, stable_ids, frontier_ids))) != (
        397,
        347,
        12,
        47,
    ):
        raise RuntimeError(
            "protected-clean replay counts drifted: "
            f"{len(hard_a_ids)}/{len(hard_b_ids)}/{len(stable_ids)}/{len(frontier_ids)}"
        )

    standalone_hard_a = hard_a_ids[:300]
    replay_hard_a = hard_a_ids[300:]
    replay_hard_b_count = 500 - len(frontier_ids) - len(stable_ids) - len(replay_hard_a)
    replay_hard_b = hard_b_ids[:replay_hard_b_count]
    if replay_hard_b_count != 344:
        raise RuntimeError("expected exactly two leakage-replacement Hard-B rows")
    old_replay_specs = (
        [(rid, "Frontier") for rid in frontier_ids]
        + [(rid, "Stable/Core") for rid in stable_ids]
        + [(rid, "Hard-A") for rid in replay_hard_a]
        + [(rid, "Hard-B") for rid in replay_hard_b]
    )
    if len(old_replay_specs) != 500:
        raise RuntimeError("approved Old replay supplement does not total 500")

    source_rows: list[dict[str, Any]] = []
    source_proofs: dict[str, str] = {}
    for original in new_wb:
        row = dict(original)
        row["stage2_phase_c1"] = {
            "top_level_role": "New WB",
            "bucket": "New WB",
            "selection_seed": SELECTION_SEED,
            "user_approved_hard_b_supplement": 342,
            "leakage_replacement_hard_b": 2,
        }
        source_rows.append(row)
        source_proofs[record_id(row)] = proof(original)
    for rid in standalone_hard_a:
        original = train_by_id[rid]
        row = dict(original)
        row["stage2_phase_c1"] = {
            "top_level_role": "Standalone Hard-A",
            "bucket": "Hard-A",
            "selection_seed": SELECTION_SEED,
            "user_approved_hard_b_supplement": 342,
            "leakage_replacement_hard_b": 2,
        }
        source_rows.append(row)
        source_proofs[rid] = proof(original)
    for rid, bucket in old_replay_specs:
        original = train_by_id[rid]
        row = dict(original)
        row["stage2_phase_c1"] = {
            "top_level_role": "Old replay",
            "bucket": bucket,
            "selection_seed": SELECTION_SEED,
            "priority": "Frontier > Stable/Core > Hard-A; approved Hard-B fills residual",
            "user_approved_hard_b_supplement": 342,
            "leakage_replacement_hard_b": 2,
        }
        source_rows.append(row)
        source_proofs[rid] = proof(original)
    source_rows.sort(key=lambda row: stable_rank(record_id(row), "manifest-order"))
    if len(source_rows) != 2000 or len(source_proofs) != 2000:
        raise RuntimeError("Phase-C1 source manifest is not 2000 unique rows")
    for row in source_rows:
        row["text"] = str(row.get("prompt") or "") + proof(row)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["m0_merged"]), local_files_only=True, trust_remote_code=False
    )
    eos_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    if eos_token != "<|im_end|>" or eos_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("frozen tokenizer EOS/PAD identity drifted")
    effective_rows: list[dict[str, Any]] = []
    eos_counts = Counter()
    valid_label_counts: list[int] = []
    for source_row in source_rows:
        rid = record_id(source_row)
        row = dict(source_row)
        completion = proof(source_row)
        while completion.endswith(eos_token):
            completion = completion[: -len(eos_token)]
        if completion.strip() != source_proofs[rid]:
            raise RuntimeError(f"proof changed before EOS materialization: {rid}")
        prompt = str(row.get("prompt") or "")
        if not prompt.strip() or not completion.strip():
            raise RuntimeError(f"empty prompt/completion: {rid}")
        row["completion"] = completion + eos_token
        row["text"] = prompt + row["completion"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + row["completion"], add_special_tokens=True)[
            "input_ids"
        ]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"prompt prefix mismatch: {rid}")
        labels = full_ids[len(prompt_ids) :]
        if not labels:
            eos_counts["zero_label"] += 1
        if labels and labels[-1] == eos_id:
            eos_counts["last_valid_label_is_eos"] += 1
        if eos_id in labels and labels[-1] == eos_id:
            eos_counts["eos_supervised"] += 1
        if len(full_ids) > 1024:
            eos_counts["semantic_truncation"] += 1
        if completion.strip():
            eos_counts["completion_nonempty"] += 1
        valid_label_counts.append(len(labels))
        row["stage2_phase_c1"] = dict(row["stage2_phase_c1"])
        row["stage2_phase_c1"].update(
            {
                "valid_label_count_with_eos": len(labels),
                "last_valid_label_id": labels[-1] if labels else None,
            }
        )
        effective_rows.append(row)

    duplicates = duplicate_audit(source_rows)
    if duplicates != {
        "rows": 2000,
        "duplicate_rows": 0,
        "duplicate_record_ids": 0,
        "theorem_group_duplicates": 0,
        "max_repeat": 1,
    }:
        raise RuntimeError(f"Phase-C1 duplicate gate failed: {duplicates}")
    proof_modifications = sum(
        proof(row).removesuffix(eos_token).strip() != source_proofs[record_id(row)]
        for row in effective_rows
    )
    if proof_modifications:
        raise RuntimeError("Phase-C1 reference proofs changed")
    eos_audit = {
        "status": "PASSED",
        "rows": len(effective_rows),
        "completion_nonempty": eos_counts["completion_nonempty"],
        "valid_label_gt_zero": sum(value > 0 for value in valid_label_counts),
        "last_valid_label_is_eos": eos_counts["last_valid_label_is_eos"],
        "eos_supervised": eos_counts["eos_supervised"],
        "eos_token_id": eos_id,
        "eos_is_minus_100": False,
        "zero_label": eos_counts["zero_label"],
        "semantic_truncation": eos_counts["semantic_truncation"],
        "min_valid_labels": min(valid_label_counts),
        "max_valid_labels": max(valid_label_counts),
    }
    if (
        eos_audit["completion_nonempty"] != 2000
        or eos_audit["valid_label_gt_zero"] != 2000
        or eos_audit["last_valid_label_is_eos"] != 2000
        or eos_audit["eos_supervised"] != 2000
        or eos_audit["zero_label"] != 0
        or eos_audit["semantic_truncation"] != 0
    ):
        raise RuntimeError(f"Phase-C1 EOS gate failed: {eos_audit}")

    leakage = {name: overlap(source_rows, rows) for name, rows in protected_rows.items()}
    if not all(result["passed"] for result in leakage.values()):
        violations: list[dict[str, Any]] = []
        protected_qualified = {
            name: {identity(row)["qualified_theorem"] for row in rows}
            for name, rows in protected_rows.items()
        }
        for row in source_rows:
            qualified = identity(row)["qualified_theorem"]
            for name, values in protected_qualified.items():
                if qualified and qualified in values:
                    violations.append(
                        {
                            "record_id": record_id(row),
                            "top_level_role": row["stage2_phase_c1"]["top_level_role"],
                            "bucket": row["stage2_phase_c1"]["bucket"],
                            "qualified_theorem": qualified,
                            "protected_dataset": name,
                        }
                    )
        write_json(
            root / "audit/preparation_failure.json",
            {
                "status": "BLOCKED_PROTECTED_LEAKAGE",
                "leakage": leakage,
                "qualified_theorem_violations": violations,
                "trainer_started": False,
            },
        )
        raise RuntimeError(
            f"Phase-C1 protected leakage gate failed: violations={violations}"
        )
    leakage_audit = {
        "status": "PASSED",
        "checks": list(identity({}).keys()),
        "protected": leakage,
        "wb_train_retention150_allowed_overlap": overlap(source_rows, retention),
        "ld_medium_holdout": {
            "status": "not_yet_constructed_phase3_only",
            "phase_c1_contains_ld_medium": False,
        },
    }

    source_path = root / "source_manifest.jsonl"
    write_jsonl(source_path, source_rows)
    write_jsonl(manifest_path, effective_rows)
    manifest_hash = sha256(manifest_path)
    role_counts = Counter(row["stage2_phase_c1"]["top_level_role"] for row in source_rows)
    bucket_counts = Counter(row["stage2_phase_c1"]["bucket"] for row in source_rows)
    data_audit = {
        "phase": "C1",
        "model_name": "S2-New-WB",
        "status": "PASSED_READY_FOR_TRAINING",
        "user_approved_revision": "342 Hard-B rows supplement Old replay",
        "automatic_leakage_gate_replacement": (
            "2 protected-name-conflicting Hard-A rows removed and replaced by "
            "2 additional protected-clean Hard-B rows"
        ),
        "excluded_by_protected_gate": excluded_by_protected_gate,
        "rows": 2000,
        "role_counts": dict(role_counts),
        "bucket_counts": dict(bucket_counts),
        "old_replay_counts": dict(
            Counter(
                row["stage2_phase_c1"]["bucket"]
                for row in source_rows
                if row["stage2_phase_c1"]["top_level_role"] == "Old replay"
            )
        ),
        "duplicates": duplicates,
        "proof_modifications": proof_modifications,
        "all_pantograph_verified": all(
            row.get("pantograph_verified") is True for row in source_rows
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "selection_seed": SELECTION_SEED,
        "training_seed": TRAINING_SEED,
    }
    if dict(role_counts) != {
        "New WB": 1200,
        "Standalone Hard-A": 300,
        "Old replay": 500,
    } or dict(bucket_counts) != {
        "New WB": 1200,
        "Hard-A": 397,
        "Hard-B": 344,
        "Frontier": 47,
        "Stable/Core": 12,
    }:
        raise RuntimeError(f"Phase-C1 composition drifted: {role_counts}; {bucket_counts}")

    gate = {
        "passed": True,
        "phase": "C1",
        "rows": 2000,
        "effective_manifest": str(manifest_path),
        "effective_manifest_sha256": manifest_hash,
        "source_manifest_sha256": sha256(source_path),
        "m0_model_sha256": EXPECTED_M0_MODEL_HASH,
        "generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH,
        "role_counts": dict(role_counts),
        "bucket_counts": dict(bucket_counts),
        "duplicates": duplicates,
        "leakage": leakage_audit,
        "eos": eos_audit,
        "proof_modifications": proof_modifications,
        "user_approved_hard_b_supplement": 342,
        "leakage_replacement_hard_b": 2,
    }
    gate_path = root / "audit/phaseC1_manifest_gate.json"
    write_json(gate_path, gate)
    training_contract = {
        "phase": "C1",
        "model_name": "S2-New-WB",
        "status": "READY_FOR_TRAINING",
        "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA",
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "expected_rows": 2000,
        "expected_optimizer_steps": math.ceil(2000 / 16),
        "resolved_config": {
            "model_name_or_path": str(paths["m0_merged"]),
            "adapter_path": None,
            "train_file": str(manifest_path),
            "validation_file": str(paths["validation"]),
            "output_dir": str(root / "training/trainer"),
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
            "logging_steps": 10,
            "eval_steps": None,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_steps": 125,
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
            "seed": TRAINING_SEED,
            "data_seed": TRAINING_SEED,
        },
        "manifest_gate_sha256": sha256(gate_path),
        "canonical_generation_contract": {
            "path": str(paths["generation_contract"]),
            "generation_config_sha256": EXPECTED_GENERATION_CONFIG_HASH,
        },
    }
    write_json(root / "training_contract.json", training_contract)
    write_json(root / "data_audit.json", data_audit)
    write_json(root / "leakage_audit.json", leakage_audit)
    write_json(root / "eos_audit.json", eos_audit)
    write_json(
        root / "evaluation/status.json",
        {
            "status": "NOT_STARTED_WAITING_FOR_TRAINING",
            "canaries_started": False,
            "formal_evaluation_started": False,
        },
    )
    (root / "report.md").write_text(
        "# Phase C1 — New WB\n\n"
        "**Status: data gates passed; training pending.**\n\n"
        "The approved 2,000-row manifest contains 1,200 New WB, 300 standalone "
        "Hard-A, and 500 Old replay rows. Old replay contains 47 Frontier, 12 "
        "Stable/Core, 97 remaining Hard-A, 342 user-approved Hard-B, and 2 "
        "additional Hard-B rows replacing protected-name-conflicting Hard-A. "
        "Duplicate rows and theorem groups are zero; max repeat is "
        "one; protected leakage is zero; all 2,000 final labels supervise EOS.\n",
        encoding="utf-8",
    )
    print(json.dumps({"data_audit": data_audit, "eos": eos_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
