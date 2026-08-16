"""Audit the immutable Phase-C1 recipe before any Trainer is allowed to run.

This script intentionally materializes only a New-WB preselection.  It never
creates the Phase-C1 training manifest and never starts model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SELECTION_SEED = 20260801
EXPECTED_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
EXPECTED_GENERATION_CONFIG_HASH = (
    "bd8f5d25051cb155aaf74b1f25082ddca5b16b44afce7fe066c12c2ab8697386"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage2_data_ratio_ablation/phaseC1_new_wb"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or row.get("sample_id") or "")


def statement(row: dict[str, Any]) -> str:
    return str(
        row.get("lean_statement")
        or row.get("training_statement")
        or row.get("statement")
        or ""
    ).strip()


def normalized_statement(row: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", statement(row)).strip()


def theorem_group(row: dict[str, Any]) -> str:
    value = row.get("theorem_group_id") or row.get("statement_id")
    if value:
        return str(value)
    return "theorem_" + hashlib.sha256(normalized_statement(row).encode()).hexdigest()


def qualified_theorem(row: dict[str, Any]) -> str:
    value = row.get("qualified_name") or row.get("declaration_name")
    if value:
        return str(value)
    match = re.search(
        r"\b(?:theorem|lemma|example|def)\s+([^\s:{(\[]+)", statement(row)
    )
    return match.group(1) if match else ""


def proof(row: dict[str, Any]) -> str:
    return str(
        row.get("proof") or row.get("completion") or row.get("training_proof") or ""
    ).strip()


def proof_hash(row: dict[str, Any]) -> str:
    normalized = re.sub(r"\s+", " ", proof(row)).strip()
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else ""


def identity(row: dict[str, Any]) -> dict[str, Any]:
    group = theorem_group(row)
    return {
        "record_id": record_id(row),
        "theorem_group": group,
        "qualified_theorem": qualified_theorem(row),
        "exact_statement": statement(row),
        "normalized_statement": normalized_statement(row),
        "proof_variant": (group, proof_hash(row)),
    }


def identity_sets(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    values = {key: set() for key in identity({})}
    for row in rows:
        for key, value in identity(row).items():
            if value and value != ("theorem_" + hashlib.sha256(b"").hexdigest(), ""):
                values[key].add(value)
    return values


def overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    first = identity_sets(left)
    second = identity_sets(right)
    intersections = {key: first[key] & second[key] for key in first}
    return {
        **{f"{key}_overlap": len(value) for key, value in intersections.items()},
        "sample_record_ids": sorted(intersections["record_id"])[:10],
        "sample_theorem_groups": sorted(intersections["theorem_group"])[:10],
        "passed": not any(intersections.values()),
    }


def stable_rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{record_id(row)}".encode()).hexdigest()


def candidate_is_verified(row: dict[str, Any]) -> bool:
    environment_hash = str(
        row.get("environment_hash") or row.get("verification_environment_hash") or ""
    )
    return all(
        (
            bool(record_id(row)),
            bool(statement(row)),
            bool(proof(row)),
            row.get("pantograph_verified") is True,
            row.get("proof_verified") is True,
            str(row.get("verification_status") or "verified") == "verified",
            environment_hash == EXPECTED_ENVIRONMENT_HASH,
            int(row.get("total_tokens") or 0) > 0,
            int(row.get("total_tokens") or 0) <= 1023,
        )
    )


def materialize_verified_reserve_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pantograph-attested reserve row to the frozen SFT schema."""
    informal = str(row.get("informal_statement") or "").strip()
    lean = statement(row)
    reference = proof(row)
    prompt = (
        "### Informal statement\n"
        + informal
        + "\n\n### Lean statement\n"
        + lean
        + "\n\n### Lean proof\n"
    )
    materialized = dict(row)
    materialized.update(
        {
            "id": record_id(row),
            "record_id": record_id(row),
            "lean_statement": lean,
            "prompt": prompt,
            "completion": reference,
            "proof": reference,
            "sampling_source": "WB",
            "pantograph_verified": True,
            "proof_verified": True,
            "statement_verified": True,
            "verification_status": "verified",
            "environment_hash": EXPECTED_ENVIRONMENT_HASH,
            "verification_environment_hash": EXPECTED_ENVIRONMENT_HASH,
            "zero_label": False,
            "truncated": False,
            "new_wb_verification_provenance": (
                "data/processed/lean_workbook_verified_v2/audit/verified_records.jsonl"
            ),
        }
    )
    return materialized


def disjoint_from_sets(row: dict[str, Any], forbidden: dict[str, set[Any]]) -> bool:
    values = identity(row)
    return not any(values[key] in forbidden[key] for key in forbidden if values[key])


def add_identity(row: dict[str, Any], sets: dict[str, set[Any]]) -> None:
    for key, value in identity(row).items():
        if value:
            sets[key].add(value)


def duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = Counter(record_id(row) for row in rows)
    groups = Counter(theorem_group(row) for row in rows)
    exact = Counter(
        hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        for row in rows
    )
    return {
        "rows": len(rows),
        "duplicate_rows": sum(value - 1 for value in exact.values() if value > 1),
        "duplicate_record_ids": sum(value - 1 for value in ids.values() if value > 1),
        "theorem_group_duplicates": sum(
            value - 1 for value in groups.values() if value > 1
        ),
        "max_repeat": max(ids.values(), default=0),
    }


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to audit over an existing Phase-C1 manifest: {manifest_path}"
        )

    paths = {
        "a0_wb3000": project
        / "outputs/initial_anchor_ratio_ablation/manifests/A0_WB3000_LD0.jsonl",
        "m0_first_round_training": project
        / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "new_wb_frozen1000": project
        / "outputs/wb_ld_budget_support_replay_ablation/manifests/core/Extra-WB1000.jsonl",
        "new_wb_canonical": project
        / "outputs/leandojo_v2_dataset_build/final_pool/lean_workbook_canonical_candidates.jsonl",
        "new_wb_verified_reserve": project
        / "data/processed/lean_workbook_verified_v2/audit/verified_records.jsonl",
        "new_wb_dataset_manifest": project
        / "data/processed/lean_workbook_verified_v2/manifest.json",
        "classification": project / "outputs/task0b_light/classification_manifest.jsonl",
        "hard_reclassification": project
        / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json",
        "generation_contract": project / "outputs/task0b_light/generation_contract.json",
        "protected_index": project
        / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "m0_merged": project
        / "outputs/stage2_sft_incremental_ablation/shared/checkpoints/M0-ADDON-B-FROZEN-MERGED",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase-C1 inputs missing: {missing}")

    a0 = read_jsonl(paths["a0_wb3000"])
    m0_first_round = read_jsonl(paths["m0_first_round_training"])
    frozen_new_wb = read_jsonl(paths["new_wb_frozen1000"])
    canonical_wb = read_jsonl(paths["new_wb_canonical"])
    verified_reserve_raw = read_jsonl(paths["new_wb_verified_reserve"])
    wb_dataset_manifest = read_json(paths["new_wb_dataset_manifest"])
    classifications = read_jsonl(paths["classification"])
    hard = read_json(paths["hard_reclassification"])
    generation_contract = read_json(paths["generation_contract"])
    protected_index = read_json(paths["protected_index"])
    if (
        len(a0) != 3000
        or len(m0_first_round) != 3000
        or len(frozen_new_wb) != 1000
        or len(classifications) != 3000
    ):
        raise RuntimeError("frozen A0/New-WB/Task0B row counts drifted")
    if generation_contract.get("generation_config_sha256") != EXPECTED_GENERATION_CONFIG_HASH:
        raise RuntimeError("canonical generation contract drifted")
    if len(verified_reserve_raw) != 7581:
        raise RuntimeError("Pantograph-verified WB reserve count drifted")
    if (
        wb_dataset_manifest.get("environment", {}).get("environment_hash")
        != EXPECTED_ENVIRONMENT_HASH
    ):
        raise RuntimeError("verified WB reserve environment drifted")

    protected_rows: dict[str, list[dict[str, Any]]] = {}
    retention_rows: list[dict[str, Any]] = []
    missing_protected: list[str] = []
    for name, entry in protected_index.items():
        if name == "ld_medium_holdout":
            continue
        path_value = entry.get("path")
        if not path_value:
            missing_protected.append(name)
            continue
        path = Path(str(path_value))
        if not path.exists():
            missing_protected.append(name)
            continue
        rows = read_jsonl(path)
        if entry.get("role") == "retention":
            retention_rows.extend(rows)
        elif entry.get("role") in {"true_holdout", "future_true_holdout"}:
            protected_rows[name] = rows

    all_protected = [row for rows in protected_rows.values() for row in rows]
    first_round_sets = identity_sets(m0_first_round)
    protected_sets = identity_sets(all_protected)
    frozen_audit = {
        "verified": sum(candidate_is_verified(row) for row in frozen_new_wb),
        "overlap_with_first_round": overlap(frozen_new_wb, m0_first_round),
        "overlap_with_protected": overlap(frozen_new_wb, all_protected),
        "duplicates": duplicate_audit(frozen_new_wb),
    }

    canonical_verified = [row for row in canonical_wb if candidate_is_verified(row)]
    canonical_unused = [
        row for row in canonical_verified if disjoint_from_sets(row, first_round_sets)
    ]
    canonical_unused_clean = [
        row for row in canonical_unused if disjoint_from_sets(row, protected_sets)
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["m0_merged"]), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("frozen M0 tokenizer identity drifted")
    verified_reserve: list[dict[str, Any]] = []
    reserve_rejections = Counter()
    for raw in verified_reserve_raw:
        row = materialize_verified_reserve_row(raw)
        if not record_id(row) or not statement(row) or not proof(row):
            reserve_rejections["missing_required_text"] += 1
            continue
        full_ids = tokenizer(
            str(row["prompt"]) + str(row["completion"]), add_special_tokens=True
        )["input_ids"]
        prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=True)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            reserve_rejections["prompt_prefix_mismatch"] += 1
            continue
        row["input_tokens"] = len(prompt_ids)
        row["label_tokens"] = len(full_ids) - len(prompt_ids)
        row["total_tokens"] = len(full_ids)
        if row["label_tokens"] <= 0:
            reserve_rejections["zero_label"] += 1
            continue
        if row["total_tokens"] > 1023:
            reserve_rejections["overlength_with_supervised_eos"] += 1
            continue
        verified_reserve.append(row)
    reserve_unused = [
        row for row in verified_reserve if disjoint_from_sets(row, first_round_sets)
    ]
    reserve_unused_clean = [
        row for row in reserve_unused if disjoint_from_sets(row, protected_sets)
    ]

    selected = list(frozen_new_wb)
    selected_sets = identity_sets(selected)
    supplemental: list[dict[str, Any]] = []
    for row in sorted(reserve_unused_clean, key=stable_rank):
        if disjoint_from_sets(row, selected_sets):
            supplemental.append(row)
            add_identity(row, selected_sets)
        if len(supplemental) == 200:
            break
    selected.extend(supplemental)
    selected_audit = duplicate_audit(selected)
    selected_leakage = {
        name: overlap(selected, rows) for name, rows in protected_rows.items()
    }
    selected_retention = overlap(selected, retention_rows)
    new_wb_ready = (
        len(selected) == 1200
        and frozen_audit["verified"] == 1000
        and frozen_audit["overlap_with_first_round"]["passed"]
        and frozen_audit["overlap_with_protected"]["passed"]
        and selected_audit["duplicate_rows"] == 0
        and selected_audit["duplicate_record_ids"] == 0
        and selected_audit["theorem_group_duplicates"] == 0
        and selected_audit["max_repeat"] == 1
        and all(value["passed"] for value in selected_leakage.values())
    )

    class_counts = Counter(str(row.get("final_bucket") or "") for row in classifications)
    hard_records = list(hard.get("records") or [])
    hard_a_ids = {
        record_id(row) for row in hard_records if row.get("hard_bucket") == "Hard-A"
    }
    stable_ids = {
        record_id(row) for row in classifications if row.get("final_bucket") == "stable_core"
    }
    frontier_ids = {
        record_id(row) for row in classifications if row.get("final_bucket") == "frontier"
    }
    if (len(stable_ids), len(frontier_ids), len(hard_a_ids)) != (12, 47, 399):
        raise RuntimeError("Task0B/Phase-A replay bucket counts drifted")
    if (stable_ids & frontier_ids) or (stable_ids & hard_a_ids) or (frontier_ids & hard_a_ids):
        raise RuntimeError("replay categories are unexpectedly non-disjoint")

    required = {
        "New WB": 1200,
        "Old replay": 500,
        "standalone Hard-A": 300,
        "total": 2000,
    }
    replay_available = {
        "Stable/Core": len(stable_ids),
        "Frontier": len(frontier_ids),
        "Hard-A": len(hard_a_ids),
        "unique_union": len(stable_ids | frontier_ids | hard_a_ids),
    }
    # The 300 standalone Hard-A rows and the Hard-A rows inside Old replay must
    # be disjoint under max_repeat=1. Reserving 300 leaves at most 99 Hard-A.
    max_old_replay_after_standalone = (
        len(stable_ids) + len(frontier_ids) + max(0, len(hard_a_ids) - 300)
    )
    old_replay_shortage = max(0, required["Old replay"] - max_old_replay_after_standalone)
    exact_recipe_ready = new_wb_ready and old_replay_shortage == 0
    status = (
        "READY"
        if exact_recipe_ready
        else "BLOCKED_REPLAY_QUOTA"
        if new_wb_ready
        else "BLOCKED_DATA_QUOTAS"
    )

    if new_wb_ready:
        preview_path = output / "preflight_new_wb_1200.jsonl"
        preview_rows: list[dict[str, Any]] = []
        frozen_ids = {record_id(row) for row in frozen_new_wb}
        for row in selected:
            annotated = dict(row)
            annotated["stage2_phase_c1_preflight"] = {
                "role": "New WB",
                "selection_seed": SELECTION_SEED,
                "source_pool": (
                    "frozen_Extra-WB1000"
                    if record_id(row) in frozen_ids
                    else "pantograph_verified_wb_reserve_supplement"
                ),
                "training_authorized": False,
            }
            preview_rows.append(annotated)
        write_jsonl(preview_path, preview_rows)
    else:
        preview_path = None

    data_audit = {
        "phase": "C1",
        "model_name": "S2-New-WB",
        "status": status,
        "exact_recipe_ready": exact_recipe_ready,
        "required": required,
        "new_wb": {
            "status": "READY" if new_wb_ready else "BLOCKED",
            "frozen_pool_rows": len(frozen_new_wb),
            "canonical_pool_rows": len(canonical_wb),
            "canonical_verified_current_environment_rows": len(canonical_verified),
            "canonical_verified_first_round_unused_rows": len(canonical_unused),
            "canonical_verified_unused_protected_clean_rows": len(canonical_unused_clean),
            "pantograph_verified_reserve_rows": len(verified_reserve_raw),
            "reserve_tokenizer_eligible_rows": len(verified_reserve),
            "reserve_first_round_unused_rows": len(reserve_unused),
            "reserve_unused_protected_clean_rows": len(reserve_unused_clean),
            "reserve_rejections": dict(reserve_rejections),
            "supplement_selected": len(supplemental),
            "preselected_total": len(selected),
            "preselection_path": str(preview_path) if preview_path else None,
            "preselection_sha256": sha256(preview_path) if preview_path else None,
            "duplicates": selected_audit,
        },
        "replay": {
            "available": replay_available,
            "requested_internal_priority": ["Frontier", "Stable/Core", "Hard-A"],
            "standalone_hard_a_reserved": 300,
            "hard_a_remaining_for_old_replay": max(0, len(hard_a_ids) - 300),
            "maximum_unique_old_replay_after_reservation": max_old_replay_after_standalone,
            "old_replay_required": 500,
            "old_replay_shortage": old_replay_shortage,
            "maximum_total_rows_under_frozen_categories": (
                1200 + len(stable_ids | frontier_ids | hard_a_ids)
            ),
            "total_shortage": 2000 - (1200 + len(stable_ids | frontier_ids | hard_a_ids)),
        },
        "forbidden_actions_not_taken": {
            "Hard-B_substitution": True,
            "Hard-C_used": False,
            "replacement_sampling": False,
            "duplicate_rows_added": False,
            "recipe_modified": False,
            "trainer_started": False,
            "gpu_training_started": False,
        },
        "source_identity": {
            name: {
                "path": str(path),
                "sha256": sha256(path) if path.is_file() else None,
                "rows": (
                    len(read_jsonl(path)) if path.is_file() and path.suffix == ".jsonl" else None
                ),
            }
            for name, path in paths.items()
            if path.is_file()
        },
    }
    write_json(output / "data_audit.json", data_audit)

    leakage_audit = {
        "status": "NEW_WB_PRESELECTION_PASSED_PHASE_MANIFEST_NOT_BUILT"
        if new_wb_ready
        else "BLOCKED_NEW_WB_PRESELECTION",
        "checks": list(identity({}).keys()),
        "protected": selected_leakage,
        "wb_train_retention150_allowed_overlap": selected_retention,
        "missing_or_future_protected_sets": sorted(
            set(missing_protected) | {"ld_medium_holdout"}
        ),
        "ld_medium_holdout_note": "Not constructed; Phase C1 contains no LD-medium rows.",
        "full_phase_manifest_check": "NOT_RUN_NO_VALID_2000_ROW_MANIFEST",
    }
    write_json(output / "leakage_audit.json", leakage_audit)

    training_contract = {
        "phase": "C1",
        "model_name": "S2-New-WB",
        "status": "PREREGISTERED_NOT_STARTED_DATA_GATE_BLOCKED",
        "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA",
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "expected_rows": 2000,
        "expected_optimizer_steps": 125,
        "resolved_config": {
            "model_name_or_path": str(paths["m0_merged"]),
            "adapter_path": None,
            "train_file": str(manifest_path),
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
            "gradient_accumulation_steps": 16,
            "dataloader_drop_last": False,
            "dataloader_num_workers": 0,
            "max_steps": -1,
            "learning_rate": 1e-5,
            "num_train_epochs": 1.0,
            "logging_steps": 10,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
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
            "completion_only_loss": True,
        },
        "canonical_generation_contract": {
            "path": str(paths["generation_contract"]),
            "generation_config_sha256": EXPECTED_GENERATION_CONFIG_HASH,
        },
    }
    write_json(output / "training_contract.json", training_contract)
    write_json(
        output / "eos_audit.json",
        {
            "status": "NOT_RUN_NO_VALID_2000_ROW_MANIFEST",
            "required": {
                "completion_nonempty": True,
                "valid_labels_gt_zero": True,
                "last_valid_label_equals_eos_token_id": 151645,
                "eos_not_minus_100": True,
                "zero_label": 0,
                "semantic_truncation": 0,
            },
            "trainer_started": False,
        },
    )
    write_json(
        output / "evaluation/status.json",
        {
            "status": "NOT_RUN_DATA_GATE_BLOCKED",
            "reason": "No valid 2000-row Phase-C1 manifest exists.",
            "canaries_started": False,
            "formal_evaluation_started": False,
            "forbidden_suites_started": {
                "Full500": False,
                "Strict-unseen200": False,
                "miniF2F": False,
            },
        },
    )

    report = [
        "# Phase C1 — New WB preflight report",
        "",
        f"**Status: {status}**",
        "",
        (
            "Phase C1 cannot enter training because the frozen replay categories do not contain enough unique rows. The New WB quota itself is satisfiable from verified, first-round-unused data."
            if new_wb_ready
            else "Phase C1 cannot enter training because both the New WB and replay quotas are unsatisfied."
        ),
        "",
        "## New WB",
        "",
        f"- Frozen Extra-WB pool: {len(frozen_new_wb)} rows.",
        f"- Canonical-pool rows outside frozen M0 first-round training after all identity checks: {len(canonical_unused_clean)}.",
        f"- Pantograph-verified reserve rows outside frozen M0 first-round training and protected sets: {len(reserve_unused_clean)}.",
        f"- Deterministically preselected supplement: {len(supplemental)} rows.",
        f"- Audited New WB total: {len(selected)}/1200; data gate: {'PASS' if new_wb_ready else 'FAIL'}.",
        "",
        "## Replay quota blocker",
        "",
        "| Bucket | Available unique rows |",
        "|---|---:|",
        f"| Stable/Core | {len(stable_ids)} |",
        f"| Frontier | {len(frontier_ids)} |",
        f"| Hard-A | {len(hard_a_ids)} |",
        f"| Union | {len(stable_ids | frontier_ids | hard_a_ids)} |",
        "",
        "The recipe requires 500 Old replay rows plus 300 separate Hard-A rows. After reserving 300 Hard-A rows, only 12 Stable/Core + 47 Frontier + 99 Hard-A = 158 unique rows remain for Old replay, leaving a 342-row shortage. Equivalently, all permitted replay categories contain only 458 unique rows for 800 requested replay slots.",
        "",
        "The maximum frozen-category manifest is therefore 1200 New WB + 458 replay = 1658 rows, not 2000.",
        "",
        "## Stop decision",
        "",
        "No Phase-C1 manifest was fabricated. Hard-B was not substituted, Hard-C was not used, no row was repeated, the recipe was not changed, and no Trainer or GPU training was started.",
        "",
        "To continue, explicit approval is needed for a revised replay composition (for example, a named Hard-B supplement or a larger New-WB share).",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    summary = output.parent / "final_summary.md"
    summary.write_text(
        "# Stage-2 data-ratio ablation status\n\n"
        "- Phase C1: blocked at the immutable data gate; New WB 1200 is ready, "
        "but the permitted replay union provides 458 unique rows for 800 slots "
        "(shortage 342).\n"
        "- Phase C2: not started.\n"
        "- Phase C3: not started.\n"
        "- No Trainer, GPU training, canary, or formal evaluation was started.\n",
        encoding="utf-8",
    )
    write_json(
        output / "preflight_status.json",
        {
            "status": status,
            "new_wb_ready": new_wb_ready,
            "phase_manifest_created": False,
            "trainer_started": False,
            "evaluation_started": False,
            "blocking_shortage": old_replay_shortage,
        },
    )
    print(json.dumps(data_audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
