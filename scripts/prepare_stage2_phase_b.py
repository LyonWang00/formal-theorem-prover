"""Prepare and gate the approved Stage-2 Phase B hard-replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.prepare_stage2_sft_phase0 import (
    duplicate_audit,
    overlap,
    proof,
    record_id,
    sha256,
    source_kind,
    theorem_group,
)


SELECTION_SEED = 20261801
TRAINING_SEED = 42
MODEL_RELATIVE = Path(
    "outputs/stage2_sft_incremental_ablation/shared/checkpoints/"
    "M0-ADDON-B-FROZEN-MERGED"
)
TRAIN_RELATIVE = Path(
    "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/"
    "ADDON-B-WB2000-LD1000.jsonl"
)
CLASSIFICATION_RELATIVE = Path("outputs/task0b_light/classification_manifest.jsonl")
RECLASSIFICATION_RELATIVE = Path(
    "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json"
)
CONTRACT_RELATIVE = Path("outputs/task0b_light/generation_contract.json")
PROTECTED_RELATIVE = Path(
    "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json"
)
VALIDATION_RELATIVE = Path("data/processed/lean_workbook_verified_v2/eval.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage2_data_ablation/phaseB_hard_replay"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{value}".encode()).hexdigest()


def select_frontier(
    rows: list[dict[str, Any]], *, wb: int = 38, ld: int = 2
) -> list[dict[str, Any]]:
    by_source = {
        source: sorted(
            [row for row in rows if str(row.get("source")) == source],
            key=lambda row: stable_rank(record_id(row)),
        )
        for source in ("WB", "LD")
    }
    if len(by_source["WB"]) < wb or len(by_source["LD"]) < ld:
        raise RuntimeError(
            f"Frontier source strata unavailable: "
            f"WB={len(by_source['WB'])}, LD={len(by_source['LD'])}"
        )
    return by_source["WB"][:wb] + by_source["LD"][:ld]


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase B preparation: {output}")

    paths = {
        "training": project / TRAIN_RELATIVE,
        "classification": project / CLASSIFICATION_RELATIVE,
        "reclassification": project / RECLASSIFICATION_RELATIVE,
        "generation_contract": project / CONTRACT_RELATIVE,
        "protected_index": project / PROTECTED_RELATIVE,
        "validation": project / VALIDATION_RELATIVE,
        "model": project / MODEL_RELATIVE,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase B inputs missing: {missing}")

    training = read_jsonl(paths["training"])
    classifications = read_jsonl(paths["classification"])
    reclassification = read_json(paths["reclassification"])
    generation_contract = read_json(paths["generation_contract"])
    protected_index = read_json(paths["protected_index"])
    if len(training) != 3000 or len(classifications) != 3000:
        raise RuntimeError("frozen first-round inputs must each contain 3000 rows")
    if generation_contract.get("generation_config_sha256") != (
        "bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386"
    ):
        raise RuntimeError("Task0B canonical generation contract drifted")

    train_by_id = {record_id(row): row for row in training}
    class_by_id = {record_id(row): row for row in classifications}
    if len(train_by_id) != 3000 or set(train_by_id) != set(class_by_id):
        raise RuntimeError("training/classification identity mismatch")
    phase_a_records = list(reclassification.get("records") or [])
    hard_by_bucket = {
        bucket: [row for row in phase_a_records if row.get("hard_bucket") == bucket]
        for bucket in ("Hard-A", "Hard-B", "Hard-C")
    }
    if {key: len(value) for key, value in hard_by_bucket.items()} != {
        "Hard-A": 399,
        "Hard-B": 349,
        "Hard-C": 85,
    }:
        raise RuntimeError("Phase A hard buckets drifted")

    stable = [
        row for row in classifications if row.get("final_bucket") == "stable_core"
    ]
    frontier_all = [
        row for row in classifications if row.get("final_bucket") == "frontier"
    ]
    if len(stable) != 12 or len(frontier_all) != 47:
        raise RuntimeError("Stable/Frontier counts drifted")
    frontier = select_frontier(frontier_all)

    selected_specs: list[tuple[str, str]] = []
    selected_specs.extend(
        (record_id(row), "Hard-A") for row in hard_by_bucket["Hard-A"]
    )
    selected_specs.extend(
        (record_id(row), "Hard-B") for row in hard_by_bucket["Hard-B"]
    )
    selected_specs.extend((record_id(row), "Stable/Core") for row in stable)
    selected_specs.extend((record_id(row), "Frontier") for row in frontier)
    if len(selected_specs) != 800 or len({rid for rid, _ in selected_specs}) != 800:
        raise RuntimeError("approved Phase B composition is not 800 unique rows")
    if any(rid not in train_by_id for rid, _ in selected_specs):
        raise RuntimeError("selected row missing from frozen first-round manifest")

    # Manifest order is deterministic but has no sampling effect beyond the
    # separately audited fixed-manifest permutation in the trainer.
    selected_specs.sort(key=lambda pair: stable_rank(pair[0]))
    source_rows: list[dict[str, Any]] = []
    for rid, bucket in selected_specs:
        original = train_by_id[rid]
        row = dict(original)
        row["stage2_phase_b"] = {
            "bucket": bucket,
            "selection_seed": SELECTION_SEED,
            "selection": "all Hard-A, all Hard-B, all Stable/Core, stratified 38 WB + 2 LD Frontier",
            "phase_a_analysis_version": reclassification.get("analysis_version"),
            "approved_frontier_supplement": True,
        }
        if proof(row) != proof(original):
            raise RuntimeError("reference proof changed while annotating manifest")
        source_rows.append(row)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["model"]), trust_remote_code=False
    )
    eos_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    if eos_token != "<|im_end|>" or eos_id != 151645:
        raise RuntimeError("frozen tokenizer EOS identity drifted")
    effective_rows: list[dict[str, Any]] = []
    eos_audit = Counter()
    valid_label_counts: list[int] = []
    for source_row in source_rows:
        row = dict(source_row)
        completion = str(row.get("completion") or row.get("proof") or "")
        while completion.endswith(eos_token):
            completion = completion[: -len(eos_token)]
        if completion.strip() != proof(source_row):
            raise RuntimeError("source completion differs from verified proof")
        row["completion"] = completion + eos_token
        prompt = str(row.get("prompt") or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + row["completion"], add_special_tokens=True)[
            "input_ids"
        ]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("prompt is not a token prefix of prompt+completion")
        labels = full_ids[len(prompt_ids) :]
        if not labels:
            eos_audit["zero_label"] += 1
        if labels and labels[-1] == eos_id:
            eos_audit["last_valid_label_is_eos"] += 1
        if labels.count(eos_id) != 1:
            eos_audit["not_exactly_one_eos"] += 1
        if len(full_ids) > 1024:
            eos_audit["semantic_truncation"] += 1
        if completion.strip():
            eos_audit["completion_nonempty"] += 1
        valid_label_counts.append(len(labels))
        row["stage2_phase_b"] = dict(row["stage2_phase_b"])
        row["stage2_phase_b"]["valid_label_count_with_eos"] = len(labels)
        row["stage2_phase_b"]["last_valid_label_id"] = labels[-1] if labels else None
        effective_rows.append(row)

    output.mkdir(parents=True)
    source_path = output / "manifests/source_manifest.jsonl"
    effective_path = output / "manifests/effective_train.jsonl"
    write_jsonl(source_path, source_rows)
    write_jsonl(effective_path, effective_rows)

    protected_rows: dict[str, list[dict[str, Any]]] = {}
    for name, info in protected_index.items():
        raw_path = info.get("path")
        protected_rows[name] = read_jsonl(Path(raw_path)) if raw_path else []
    leakage = {
        name: overlap(source_rows, rows)
        for name, rows in protected_rows.items()
        if rows
    }
    true_holdout_failures = {
        name: result
        for name, result in leakage.items()
        if protected_index[name].get("role") != "retention" and not result["passed"]
    }
    duplicates = duplicate_audit(source_rows)
    bucket_counts = Counter(
        row["stage2_phase_b"]["bucket"] for row in source_rows
    )
    source_counts = Counter(source_kind(row) for row in source_rows)
    expected_buckets = {
        "Hard-A": 399,
        "Hard-B": 349,
        "Stable/Core": 12,
        "Frontier": 40,
    }
    model_hash = sha256(paths["model"] / "model.safetensors")
    if model_hash != generation_contract.get("checkpoint_model_sha256"):
        raise RuntimeError("M0 merged model hash differs from canonical contract")
    gate_passed = bool(
        dict(bucket_counts) == expected_buckets
        and duplicates["duplicate_rows"] == 0
        and duplicates["duplicate_record_ids"] == 0
        and duplicates["theorem_group_duplicates"] == 0
        and duplicates["max_repeat"] == 1
        and not true_holdout_failures
        and eos_audit["completion_nonempty"] == 800
        and eos_audit["last_valid_label_is_eos"] == 800
        and eos_audit["zero_label"] == 0
        and eos_audit["not_exactly_one_eos"] == 0
        and eos_audit["semantic_truncation"] == 0
        and all(
            row.get("pantograph_verified") is True
            and row.get("proof_verified") is True
            and row.get("statement_verified") is True
            for row in source_rows
        )
    )
    gate = {
        "passed": gate_passed,
        "approved_composition_change": "40 Frontier supplement",
        "selection_seed": SELECTION_SEED,
        "rows": len(source_rows),
        "bucket_counts": dict(bucket_counts),
        "source_counts": dict(source_counts),
        "duplicates": duplicates,
        "leakage": leakage,
        "retention_overlap_allowed": True,
        "true_holdout_failures": true_holdout_failures,
        "eos": {
            **dict(eos_audit),
            "eos_token": eos_token,
            "eos_token_id": eos_id,
            "valid_label_count_min": min(valid_label_counts),
            "valid_label_count_max": max(valid_label_counts),
            "valid_label_count_sum": sum(valid_label_counts),
        },
        "proofs_modified": 0,
        "theorems_modified": 0,
        "hard_c_rows": 0,
        "manifest_sha256": sha256(source_path),
        "effective_manifest_sha256": sha256(effective_path),
        "m0_model_sha256": model_hash,
        "generation_contract_sha256": generation_contract[
            "generation_config_sha256"
        ],
    }
    write_json(output / "audit/phaseB_manifest_gate.json", gate)
    if not gate_passed:
        raise RuntimeError(f"Phase B launch gate failed: {json.dumps(gate)}")

    resolved_config = {
        "model_name_or_path": str(paths["model"]),
        "adapter_path": None,
        "train_file": str(effective_path),
        "validation_file": str(paths["validation"]),
        "output_dir": str(output / "training/trainer"),
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
        "save_steps": 50,
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
    }
    training_contract = {
        "phase": "B",
        "model_name": "S2-Hard-Replay",
        "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA",
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "expected_rows": 800,
        "expected_optimizer_steps": math.ceil(800 / 16),
        "resolved_config": resolved_config,
        "manifest_gate_sha256": sha256(output / "audit/phaseB_manifest_gate.json"),
    }
    write_json(output / "training_contract.json", training_contract)
    report = f"""# Phase B launch manifest report

- Composition: 399 Hard-A + 349 Hard-B + 12 Stable/Core + 40 Frontier = 800.
- Frontier supplement: user-approved, fixed seed `{SELECTION_SEED}`, 38 WB + 2 LD.
- Hard-C used: 0.
- Duplicate rows / record IDs / theorem groups: 0 / 0 / 0.
- Maximum repeat: 1.
- True-holdout leakage: 0 across all protected identity dimensions.
- WB retention overlap is reported but allowed because it is an in-distribution retention set.
- EOS: 800/800 completions end in exactly one supervised token `{eos_id}`.
- Semantic truncation: 0.
- Reference proofs and theorem statements modified: 0 / 0.
- Starting checkpoint: frozen merged M0, model hash `{model_hash}`.
- Adapter initialization: fresh; no optimizer/scheduler/adapter state is shared.
- Launch gate: **PASS**.
"""
    (output / "phaseB_manifest_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"phase": "B", "gate": "PASS", **gate}, indent=2))


if __name__ == "__main__":
    main()
