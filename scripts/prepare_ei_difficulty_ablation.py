#!/usr/bin/env python3
"""Build and gate the four EI difficulty-aware ablation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_stage2_ratio_phase_c1_feasibility import duplicate_audit, overlap, record_id, sha256, theorem_group
from scripts.prepare_ei_data_composition_ablation import materialize, protected_sets


SEED = 20260804
H0_HASH = "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
ARMS = {
    "A_medium_hard": {"Medium": 300, "Hard": 100, "Easy": 50, "Repair": 0, "Replay": 50},
    "B_medium_only": {"Medium": 400, "Hard": 50, "Easy": 0, "Repair": 0, "Replay": 50},
    "C_add_repair": {"Medium": 300, "Hard": 100, "Easy": 50, "Repair": 100, "Replay": 50},
    "D_add_replay": {"Medium": 300, "Hard": 100, "Easy": 50, "Repair": 0, "Replay": 150},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{SEED}:{namespace}:{value}".encode()).hexdigest()


def first_tactic(proof: str) -> str:
    body = re.sub(r"^\s*(?:by|term)\b", "", proof).strip()
    match = re.search(r"[A-Za-z_][A-Za-z0-9_!?']*", body)
    return match.group(0) if match else "term"


def pick(rows: list[dict[str, Any]], count: int, namespace: str) -> list[dict[str, Any]]:
    if count > len(rows):
        count = len(rows)
    return sorted(rows, key=lambda row: stable_hash(theorem_group(row), namespace))[:count]


def decorate(row: dict[str, Any], role: str, source_difficulty: str, origin: str) -> dict[str, Any]:
    row["difficulty_ablation"] = {
        "role": role,
        "dynamic_difficulty": source_difficulty,
        "target_origin": origin,
        "selection_seed": SEED,
        "valid_label_count_with_eos": row["ei_ablation"]["valid_label_count_with_eos"],
        "last_valid_label_id": row["ei_ablation"]["last_valid_label_id"],
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/difficulty_ablation"
    h0 = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED"
    if sha256(h0 / "model.safetensors") != H0_HASH:
        raise RuntimeError("frozen H0 model hash drifted")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(h0), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("frozen tokenizer contract drifted")

    dynamic = read_jsonl(project / "outputs/expert_iteration/round0/dynamic_difficulty/dynamic_difficulty_manifest.jsonl")
    discovery = read_jsonl(project / "outputs/expert_iteration/round0/discovery/discovery_manifest.jsonl")
    successes = read_jsonl(project / "outputs/expert_iteration/round0/success_bank/success_bank.jsonl")
    verifications = read_jsonl(project / "outputs/expert_iteration/round0/discovery/candidate_verifications.jsonl")
    repairs = read_jsonl(project / "outputs/expert_iteration/ei_ablation/frontier_repair_v2/repair_verified.jsonl")
    replay_raw = read_jsonl(project / "outputs/initial_anchor_ratio_ablation/manifests/A20_WB2000_LD1000.jsonl")

    discovery_by_id = {record_id(row): row for row in discovery}
    dynamic_by_record = {str(row["record_id"]): row for row in dynamic}
    dynamic_by_theorem = {str(row["theorem_id"]): row for row in dynamic}
    verification_by_id = {str(row["generation_id"]): row for row in verifications}
    success_by_candidate = {str(row["candidate_id"]): row for row in successes}

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dynamic_row in dynamic:
        difficulty = str(dynamic_row["difficulty"]).lower()
        if difficulty not in {"easy", "medium", "hard"}:
            continue
        if difficulty == "hard" and int(dynamic_row.get("success_count") or 0) != 1:
            continue
        candidates = [success_by_candidate[cid] for cid in dynamic_row["successful_candidate_ids"] if cid in success_by_candidate]
        if not candidates:
            raise RuntimeError(f"missing successful target for {dynamic_row['record_id']}")
        candidate = min(candidates, key=lambda row: (int(row.get("generation_length") or 10**9), int(row.get("candidate_rank") or 10**9), str(row["candidate_id"])))
        verification = verification_by_id[str(candidate["candidate_id"])]
        if not (candidate.get("pantograph_verified") is True and candidate.get("no_sorry") is True and verification.get("verified") is True):
            raise RuntimeError(f"invalid generated proof receipt: {candidate['candidate_id']}")
        source = discovery_by_id[str(dynamic_row["record_id"])]
        row = materialize(
            source=source,
            proof=str(candidate["proof"]),
            role=difficulty.title(),
            origin="pantograph_verified_h0_dynamic_success",
            source_verification={
                "kind": "ei_round0_candidate_verification",
                "candidate_id": candidate["candidate_id"],
                "environment_hash": verification["environment_hash"],
                "assembled_source_hash": verification["assembled_source_hash"],
                "checkpoint_hash": candidate["checkpoint_hash"],
                "generation_config_hash": candidate["generation_config_hash"],
            },
            tokenizer=tokenizer,
        )
        if not row["truncated"]:
            pools[difficulty.title()].append(decorate(row, difficulty.title(), difficulty, "dynamic_success"))

    repair_by_group: dict[str, dict[str, Any]] = {}
    for repair in repairs:
        group_id = str(repair.get("aggregation_group_id") or "")
        dyn = dynamic_by_theorem.get(group_id)
        if dyn is None or not repair.get("repair_verified") or not repair.get("selected_verified_proof"):
            continue
        if dyn.get("zero_success_subtype") == "data_environment_hard":
            continue
        if repair.get("target_failure_class") in {"syntax_error", "unknown_identifier", "environment_error", "data_error"}:
            continue
        proof = str(repair["selected_verified_proof"])
        if any(token in proof.lower() for token in ("sorry", "admit", "axiom")):
            continue
        selected = repair["minimal_repair"] if repair["selected_strategy"] == "minimal_repair" else repair["clean_repair"]
        receipt = selected["verification"]
        if not receipt.get("verified"):
            continue
        source = discovery_by_id[str(dyn["record_id"])]
        row = materialize(
            source=source,
            proof=proof,
            role="Repair",
            origin="pantograph_verified_deepseek_repair_v2",
            source_verification={
                "kind": "repair_data_v2",
                "target_attempt_id": repair["target_attempt_id"],
                "environment_hash": receipt["environment_hash"],
                "assembled_source_hash": receipt["assembled_source_sha256"],
                "prompt_sha256": repair["prompt_sha256"],
                "selected_strategy": repair["selected_strategy"],
            },
            tokenizer=tokenizer,
        )
        if row["truncated"]:
            continue
        row = decorate(row, "Repair", str(dyn["difficulty"]), "verified_repair")
        previous = repair_by_group.get(theorem_group(row))
        if previous is None or (row["label_tokens"], row["proof"]) < (previous["label_tokens"], previous["proof"]):
            repair_by_group[theorem_group(row)] = row
    pools["Repair"] = list(repair_by_group.values())

    protected = protected_sets(project)
    all_protected = [row for rows in protected.values() for row in rows]
    all_dynamic_groups = {theorem_group(row) for role in ("Easy", "Medium", "Hard") for row in pools[role]}
    replay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in replay_raw:
        sampling_source = str(raw.get("sampling_source") or "")
        if sampling_source not in {"WB", "LD"} or theorem_group(raw) in all_dynamic_groups:
            continue
        if raw.get("pantograph_verified") is not True or raw.get("proof_verified") is not True:
            continue
        proof = str(raw.get("proof") or "")
        if not proof or any(token in proof.lower() for token in ("sorry", "admit", "axiom")):
            continue
        row = materialize(
            source=raw,
            proof=proof,
            role="Replay",
            origin="verified_inherited_h0_foundation_replay",
            source_verification={
                "kind": "inherited_m0_foundation_attestation",
                "environment_hash": raw.get("verification_environment_hash") or raw["environment_hash"],
                "assembled_source_hash": raw["assembled_source_hash"],
                "attestation_id": raw.get("attestation_id"),
                "lineage": "H0 starts from M0-ADDON-B-FROZEN; A20_WB2000_LD1000 is the frozen M0 foundation manifest",
            },
            tokenizer=tokenizer,
        )
        if not row["truncated"] and overlap([row], all_protected)["passed"]:
            replay[sampling_source].append(decorate(row, "Replay", "foundation", "foundation_replay"))

    selected_a_discovery = (
        pick(pools["Medium"], ARMS["A_medium_hard"]["Medium"], "A:medium")
        + pick(pools["Hard"], ARMS["A_medium_hard"]["Hard"], "A:hard")
        + pick(pools["Easy"], ARMS["A_medium_hard"]["Easy"], "A:easy")
    )
    a_discovery_groups = {theorem_group(row) for row in selected_a_discovery}

    manifests: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for arm, requested in ARMS.items():
        if arm in {"A_medium_hard", "C_add_repair", "D_add_replay"}:
            discovery_rows = list(selected_a_discovery)
        else:
            discovery_rows = (
                pick(pools["Medium"], requested["Medium"], arm + ":medium")
                + pick(pools["Hard"], requested["Hard"], arm + ":hard")
            )
        blocked = {theorem_group(row) for row in discovery_rows}
        repair_rows = pick([row for row in pools["Repair"] if theorem_group(row) not in blocked], requested["Repair"], arm + ":repair")
        blocked.update(theorem_group(row) for row in repair_rows)
        replay_count = requested["Replay"]
        ld_count = round(replay_count / 3)
        wb_count = replay_count - ld_count
        replay_rows = (
            pick([row for row in replay["WB"] if theorem_group(row) not in blocked], wb_count, arm + ":replay:wb")
            + pick([row for row in replay["LD"] if theorem_group(row) not in blocked], ld_count, arm + ":replay:ld")
        )
        rows = discovery_rows + repair_rows + replay_rows
        rows.sort(key=lambda row: stable_hash(theorem_group(row), arm + ":order"))
        manifests[arm] = rows
        output = root / arm
        manifest_path = output / "manifest.jsonl"
        write_jsonl(manifest_path, rows)

        roles = Counter(row["difficulty_ablation"]["role"] for row in rows)
        domains = Counter("LD-easy" if row["sampling_source"] == "LD" else "WB" for row in rows)
        tactics = Counter(first_tactic(str(row["proof"])) for row in rows)
        lengths = sorted(int(row["label_tokens"]) for row in rows)
        duplicates = duplicate_audit(rows)
        leakage = {name: overlap(rows, values) for name, values in protected.items()}
        eos = {
            "rows": len(rows),
            "completion_nonempty": sum(bool(str(row["completion"]).removesuffix(str(tokenizer.eos_token)).strip()) for row in rows),
            "valid_label_gt_zero": sum(int(row["label_tokens"]) > 0 for row in rows),
            "last_valid_label_is_eos": sum(row["difficulty_ablation"]["last_valid_label_id"] == tokenizer.eos_token_id for row in rows),
            "zero_label": sum(bool(row["zero_label"]) for row in rows),
            "semantic_truncation": sum(bool(row["truncated"]) for row in rows),
            "eos_token_id": tokenizer.eos_token_id,
            "eos_is_minus_100": tokenizer.eos_token_id == -100,
        }
        replay_roles = [row for row in rows if row["difficulty_ablation"]["role"] == "Replay"]
        replay_domains = Counter(row["sampling_source"] for row in replay_roles)
        requested_realized = {role: {"requested": requested[role], "realized": roles.get(role, 0)} for role in requested}
        audit = {
            "status": "PASSED",
            "passed": True,
            "requested_rows": sum(requested.values()),
            "realized_rows": len(rows),
            "requested_vs_realized": requested_realized,
            "shortfall_policy": "category shortfalls retained as actual counts; no repetition and no cross-role backfill",
            "role_counts": dict(roles),
            "difficulty_counts": dict(roles),
            "domain_counts": dict(domains),
            "domain_limitation": "Frozen Dynamic/Repair pools are WB-only; Replay is balanced at approximately WB:LD=2:1 but whole-arm balance is unattainable without changing prescribed sources or roles.",
            "replay_domain_counts": dict(replay_domains),
            "replay_wb_ld_ratio": (replay_domains["WB"] / replay_domains["LD"]) if replay_domains["LD"] else None,
            "tactic_distribution": dict(tactics),
            "proof_token_length": {
                "min": min(lengths), "median": lengths[len(lengths) // 2], "p95": lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)], "max": max(lengths),
            },
            "manifest_sha256": sha256(manifest_path),
            "duplicates": duplicates,
            "protected_evaluation_leakage": leakage,
            "eos": eos,
            "all_targets_pantograph_verified": all(row.get("pantograph_verified") is True for row in rows),
            "dynamic_targets_are_generated_not_reference": all(row["difficulty_ablation"]["target_origin"] == "dynamic_success" for row in rows if row["difficulty_ablation"]["role"] in {"Easy", "Medium", "Hard"}),
            "repair_targets_are_verified": all(row["difficulty_ablation"]["target_origin"] == "verified_repair" for row in repair_rows),
            "repair_excludes_data_environment_impossible": all(dynamic_by_record[row["record_id"]].get("zero_success_subtype") != "data_environment_hard" for row in repair_rows),
            "replay_target_exception": "The global no-reference-target rule is applied to Dynamic and Repair roles. Replay necessarily uses frozen supervised foundation targets because the task explicitly requires H0 foundation Replay.",
            "same_discovery_as_A": arm not in {"C_add_repair", "D_add_replay"} or {theorem_group(row) for row in discovery_rows} == a_discovery_groups,
        }
        expected_duplicates = {"rows": len(rows), "duplicate_rows": 0, "duplicate_record_ids": 0, "theorem_group_duplicates": 0, "max_repeat": 1}
        audit["passed"] = (
            duplicates == expected_duplicates
            and all(value["passed"] for value in leakage.values())
            and eos["completion_nonempty"] == len(rows)
            and eos["valid_label_gt_zero"] == len(rows)
            and eos["last_valid_label_is_eos"] == len(rows)
            and eos["zero_label"] == 0 and eos["semantic_truncation"] == 0 and not eos["eos_is_minus_100"]
            and audit["all_targets_pantograph_verified"] and audit["dynamic_targets_are_generated_not_reference"]
            and audit["repair_targets_are_verified"] and audit["repair_excludes_data_environment_impossible"]
            and abs((audit["replay_wb_ld_ratio"] or 0) - 2.0) <= 0.1
            and audit["same_discovery_as_A"]
        )
        audit["status"] = "PASSED" if audit["passed"] else "FAILED"
        write_json(output / "data_audit.json", audit)
        if not audit["passed"]:
            raise RuntimeError(f"data audit failed for {arm}")

        contract = {
            "experiment": arm,
            "status": "READY_FOR_TRAINING",
            "starting_checkpoint": str(h0),
            "starting_checkpoint_model_sha256": H0_HASH,
            "initialization": "fresh_lora_on_frozen_merged_H0",
            "expected_rows": len(rows),
            "expected_optimizer_steps": math.ceil(len(rows) / 16),
            "expected_role_counts": dict(roles),
            "manifest_sha256": audit["manifest_sha256"],
            "resolved_config": {
                "model_name_or_path": str(h0), "adapter_path": None,
                "train_file": str(manifest_path),
                "validation_file": str(project / "data/processed/lean_workbook_verified_v2/eval.jsonl"),
                "output_dir": str(output / "checkpoint/trainer"),
                "prompt_field": "prompt", "completion_field": "completion", "sample_weight_field": None,
                "sampling_strategy": "fixed_manifest_without_replacement", "requested_packing": False,
                "require_pantograph_verified": True, "require_supervised_eos": True,
                "max_seq_length": 1024, "allow_overlength": False,
                "per_device_train_batch_size": 1, "per_device_validation_batch_size": 1,
                "gradient_accumulation_steps": 16, "dataloader_drop_last": False, "dataloader_num_workers": 0,
                "max_steps": -1, "learning_rate": 1e-5, "num_train_epochs": 1.0,
                "logging_steps": 4, "eval_steps": None, "eval_strategy": "epoch", "save_strategy": "epoch",
                "save_steps": math.ceil(len(rows) / 16), "save_total_limit": 2,
                "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
                "warmup_ratio": 0.03, "warmup_steps": 0, "weight_decay": 0.01,
                "lr_scheduler_type": "linear", "max_grad_norm": 1.0, "gradient_checkpointing": True,
                "optim": "paged_adamw_8bit", "device_map": "auto",
                "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "seed": 42, "data_seed": 42,
            },
            "frozen_generation_contract": str(project / "outputs/task0b_light/generation_contract.json"),
        }
        write_json(output / "training_contract.json", contract)
        summary[arm] = {"rows": len(rows), "roles": dict(roles), "domains": dict(domains), "manifest_sha256": audit["manifest_sha256"]}
        (output / "report.md").write_text(
            f"# {arm} pre-training report\n\n- Requested/realized rows: {sum(requested.values())}/{len(rows)}\n"
            f"- Roles: {dict(roles)}\n- Domains: {dict(domains)}\n- Replay domains: {dict(replay_domains)}\n"
            f"- Data audit: PASS\n- Manifest SHA256: `{audit['manifest_sha256']}`\n",
            encoding="utf-8",
        )

    payload = {
        "status": "READY_FOR_TRAINING", "selection_seed": SEED,
        "pool_counts": {role: len(values) for role, values in pools.items()},
        "repair_eligible_after_exclusions": len(pools["Repair"]),
        "replay_eligible": {source: len(values) for source, values in replay.items()},
        "arms": summary, "trainer_started": False, "grpo_started": False,
    }
    write_json(root / "data_preparation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
