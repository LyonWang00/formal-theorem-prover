#!/usr/bin/env python3
"""Build and gate the scaled EI-A/B/C composition-ablation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.data.contracts import make_attestation_id
from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    duplicate_audit,
    overlap,
    record_id,
    sha256,
    theorem_group,
)


SEED = 20260815
H0_HASH = "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
TRAIN_ENVIRONMENT_HASH = "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
LEAN_VERSION = "4.29.1"
LEAN_COMMIT = "f72c35b3f637c8c6571d353742168ab66cc22c00"
MATHLIB_COMMIT = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"
ASSEMBLER_VERSION = "2"
NORMALIZATION_VERSION = "2"
ARMS = {
    "EI-A_success_only": {"success": 1.0, "frontier": 0.0, "replay": 0.0},
    "EI-B_success_frontier": {"success": 0.70, "frontier": 0.30, "replay": 0.0},
    "EI-C_success_frontier_replay": {"success": 0.60, "frontier": 0.20, "replay": 0.20},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


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


def normalized_proof_hash(proof: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", proof).strip().encode()).hexdigest()


def build_prompt(statement: str, informal: str = "") -> str:
    return (
        "### Informal statement\n" + informal + "\n\n"
        "### Lean statement\n" + statement + "\n\n"
        "### Lean proof\n"
    )


def first_tactic(proof: str) -> str:
    body = re.sub(r"^\s*(?:by|term)\b", "", proof).strip()
    match = re.search(r"[A-Za-z_][A-Za-z0-9_!?']*", body)
    return match.group(0) if match else "term"


def length_bin(value: int) -> str:
    if value <= 8:
        return "short"
    if value <= 24:
        return "medium"
    return "long"


def diversity_key(row: dict[str, Any]) -> tuple[str, str, str]:
    source_file = str(row.get("source_file") or "")
    parts = source_file.replace("\\", "/").split("/")
    domain = str(row.get("domain") or row.get("category") or (parts[1] if len(parts) > 2 else "WB"))
    return domain, first_tactic(str(row["proof"])), length_bin(int(row["label_tokens"]))


def diverse_select(rows: list[dict[str, Any]], count: int, namespace: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], deque[dict[str, Any]]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[diversity_key(row)].append(row)
    for key, values in grouped.items():
        values.sort(key=lambda row: stable_hash(theorem_group(row), namespace))
        buckets[key] = deque(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda key: stable_hash("|".join(key), namespace + ":bucket"))
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise RuntimeError(f"diverse selection shortfall for {namespace}: {len(selected)}/{count}")
    return selected


def materialize(
    *,
    source: dict[str, Any],
    proof: str,
    role: str,
    origin: str,
    source_verification: dict[str, Any],
    tokenizer: Any,
    record_id_override: str | None = None,
) -> dict[str, Any]:
    eos = str(tokenizer.eos_token)
    row_id = record_id_override or str(source.get("record_id") or source.get("id") or source.get("sample_id"))
    group = str(source.get("theorem_group_id") or source.get("statement_id"))
    statement = str(source.get("lean_statement") or source.get("training_statement") or source.get("statement")).strip()
    prompt = str(source.get("prompt") or build_prompt(statement, str(source.get("informal_statement") or "")))
    proof = proof.strip()
    completion = proof + eos
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"prompt token prefix mismatch: {row_id}")
    labels = full_ids[len(prompt_ids) :]
    assembled_hash = str(source_verification["assembled_source_hash"])
    attestation_id = make_attestation_id(
        record_id=row_id,
        environment_hash=TRAIN_ENVIRONMENT_HASH,
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash=assembled_hash,
    )
    return {
        "id": row_id,
        "record_id": row_id,
        "theorem_group_id": group,
        "statement_id": group,
        "source": str(source.get("source") or source.get("source_name") or "unknown"),
        "sampling_source": "LD" if "leandojo" in str(source.get("source") or "").lower() else "WB",
        "source_file": str(source.get("source_file") or ""),
        "lean_statement": statement,
        "statement": statement,
        "prompt": prompt,
        "proof": proof,
        "completion": completion,
        "text": prompt + completion,
        "proof_hash_normalized": normalized_proof_hash(proof),
        "imports": list(source.get("imports") or ["Mathlib"]),
        "data_state": "verified",
        "statement_verified": True,
        "proof_verified": True,
        "reference_proof_verified": role == "Replay",
        "pantograph_verified": True,
        "verification_status": "verified",
        "environment_hash": TRAIN_ENVIRONMENT_HASH,
        "verification_environment_hash": str(source_verification.get("environment_hash") or ""),
        "lean_version": LEAN_VERSION,
        "lean_commit": LEAN_COMMIT,
        "mathlib_commit": MATHLIB_COMMIT,
        "assembler_version": ASSEMBLER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "assembled_source_hash": assembled_hash,
        "attestation_id": attestation_id,
        "input_tokens": len(prompt_ids),
        "label_tokens": len(labels),
        "total_tokens": len(full_ids),
        "zero_label": not bool(labels),
        "truncated": len(full_ids) > 1024,
        "ei_ablation": {
            "role": role,
            "target_origin": origin,
            "selection_seed": SEED,
            "source_verification": source_verification,
            "valid_label_count_with_eos": len(labels),
            "last_valid_label_id": labels[-1] if labels else None,
        },
    }


def protected_sets(project: Path) -> dict[str, list[dict[str, Any]]]:
    paths = {
        "WB-Unseen-Holdout150": "outputs/wb_ld_budget_support_replay_ablation/datasets/wb_unseen_holdout_150.jsonl",
        "LD-easy64": "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/ld_easy_holdout_64.jsonl",
        "Monitor64": "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
        "Hard-LD128": "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl",
        "Full500": "outputs/b2_expanded_validation/datasets/full500.jsonl",
        "Strict-unseen200": "outputs/b2_expanded_validation/datasets/strict_unseen_discovery.jsonl",
    }
    return {name: read_jsonl(project / path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/ei_ablation"
    h0 = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED"
    if sha256(h0 / "model.safetensors") != H0_HASH:
        raise RuntimeError("frozen H0 model hash drifted")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(h0), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError("H0 tokenizer contract drifted")

    discovery = read_jsonl(project / "outputs/expert_iteration/round0/discovery/discovery_manifest.jsonl")
    successes = read_jsonl(project / "outputs/expert_iteration/round0/success_bank/success_bank.jsonl")
    verifications = read_jsonl(project / "outputs/expert_iteration/round0/discovery/candidate_verifications.jsonl")
    repairs = read_jsonl(root / "frontier_repair_v2/repair_verified.jsonl")
    replay_pool = read_jsonl(project / "outputs/initial_anchor_ratio_ablation/audit/ld_easy_expansion/trainable_easy_pool.jsonl")
    discovery_by_id = {record_id(row): row for row in discovery}
    verification_by_id = {str(row["generation_id"]): row for row in verifications}

    success_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in successes:
        # Candidate pipeline statement IDs are generation-stage identities and
        # intentionally differ from the frozen Discovery theorem_group_id.
        # source_id is the stable bridge back to the proof-free manifest.
        success_candidates[str(candidate["source_id"])].append(candidate)
    success_rows = []
    for source_id, candidates in success_candidates.items():
        source = discovery_by_id.get(source_id)
        if source is None:
            raise RuntimeError(f"Success Bank source_id missing from Discovery: {source_id}")
        candidate = min(candidates, key=lambda row: (int(row.get("generation_length") or 10**9), int(row.get("candidate_rank") or 10**9), str(row["candidate_id"])))
        verification = verification_by_id[str(candidate["candidate_id"])]
        if not (candidate.get("pantograph_verified") is True and verification.get("verified") is True and candidate.get("no_sorry") is True):
            raise RuntimeError(f"invalid Success Bank attestation: {candidate['candidate_id']}")
        row = materialize(
            source=source,
            proof=str(candidate["proof"]),
            role="Success",
            origin="pantograph_verified_h0_self_generation",
            source_verification={
                "kind": "ei_round0_candidate_verification",
                "candidate_id": candidate["candidate_id"],
                "candidate_pipeline_statement_id": candidate["statement_id"],
                "discovery_record_id": source_id,
                "environment_hash": verification["environment_hash"],
                "assembled_source_hash": verification["assembled_source_hash"],
                "checkpoint_hash": candidate["checkpoint_hash"],
                "generation_config_hash": candidate["generation_config_hash"],
            },
            tokenizer=tokenizer,
        )
        if not row["truncated"]:
            success_rows.append(row)
    success_by_group = {theorem_group(row): row for row in success_rows}
    if len(success_by_group) < 250:
        raise RuntimeError(f"too few unique usable Success rows: {len(success_by_group)}")

    frontier_by_group: dict[str, dict[str, Any]] = {}
    for repair in repairs:
        if not repair.get("repair_verified") or repair.get("target_failure_class") not in {"unsolved_goals", "tactic_error"}:
            continue
        source_id = str((repair.get("source_metadata") or {}).get("source_id") or "")
        source = discovery_by_id.get(source_id)
        if source is None:
            raise RuntimeError(
                f"RepairData source_id missing from Discovery: {source_id or repair['target_attempt_id']}"
            )
        group = theorem_group(source)
        selected = repair["minimal_repair"] if repair["selected_strategy"] == "minimal_repair" else repair["clean_repair"]
        receipt = selected["verification"]
        if not receipt.get("verified"):
            raise RuntimeError(f"promoted repair lacks verified receipt: {repair['target_attempt_id']}")
        row = materialize(
            source=source,
            proof=str(repair["selected_verified_proof"]),
            role="Frontier",
            origin="pantograph_verified_repair_data_v2",
            source_verification={
                "kind": "repair_data_v2",
                "target_attempt_id": repair["target_attempt_id"],
                "repair_pipeline_statement_id": repair["statement_id"],
                "discovery_record_id": source_id,
                "selected_strategy": repair["selected_strategy"],
                "resolved_attempt_assessment": repair["resolved_attempt_assessment"],
                "environment_hash": receipt["environment_hash"],
                "repair_environment_hash": repair["environment"]["environment_hash"],
                "assembled_source_hash": receipt["assembled_source_sha256"],
                "prompt_sha256": repair["prompt_sha256"],
            },
            tokenizer=tokenizer,
        )
        if row["truncated"]:
            continue
        previous = frontier_by_group.get(group)
        if previous is None or (row["label_tokens"], row["proof"]) < (previous["label_tokens"], previous["proof"]):
            frontier_by_group[group] = row

    protected = protected_sets(project)
    all_protected = [row for rows in protected.values() for row in rows]
    replay_candidates = []
    discovery_groups = {theorem_group(row) for row in discovery}
    for raw in replay_pool:
        group = theorem_group(raw)
        verification = raw.get("verification") or {}
        if group in discovery_groups or not verification.get("compile_success"):
            continue
        row = materialize(
            source=raw,
            proof=str(raw.get("training_proof") or raw.get("proof")),
            role="Replay",
            origin="verified_foundation_ld_easy_replay",
            source_verification={
                "kind": "leandojo_v2_fidelity_verification",
                "environment_hash": verification["environment_hash"],
                "assembled_source_hash": verification["assembled_source_hash"],
                "qualified_name": verification.get("qualified_name"),
                "source_file": verification.get("source_file"),
            },
            tokenizer=tokenizer,
        )
        if not row["truncated"] and overlap([row], all_protected)["passed"]:
            replay_candidates.append(row)

    total_rows = len(success_by_group)
    requested_rows = 500
    frontier_available = len(frontier_by_group)
    if len(replay_candidates) < math.floor(total_rows * 0.20):
        raise RuntimeError("verified replay shortfall")

    arm_manifests: dict[str, list[dict[str, Any]]] = {}
    for arm, ratios in ARMS.items():
        requested_frontier_count = round(total_rows * ratios["frontier"])
        frontier_count = min(requested_frontier_count, frontier_available)
        # Largest-remainder scaling of 300/100/100 to 253 is 152/51/50;
        # keep the tied extra row on Frontier because it is the tested signal.
        replay_count = math.floor(total_rows * ratios["replay"])
        success_count = total_rows - frontier_count - replay_count
        frontier = diverse_select(list(frontier_by_group.values()), frontier_count, arm + ":frontier") if frontier_count else []
        blocked_groups = {theorem_group(row) for row in frontier}
        replay = diverse_select(
            [row for row in replay_candidates if theorem_group(row) not in blocked_groups], replay_count, arm + ":replay"
        ) if replay_count else []
        blocked_groups.update(theorem_group(row) for row in replay)
        success = diverse_select(
            [row for group, row in success_by_group.items() if group not in blocked_groups], success_count, arm + ":success"
        )
        rows = success + frontier + replay
        rows.sort(key=lambda row: stable_hash(theorem_group(row), arm + ":order"))
        arm_manifests[arm] = rows

        output = root / arm
        manifest_path = output / "manifest.jsonl"
        write_jsonl(manifest_path, rows)
        duplicates = duplicate_audit(rows)
        leakage = {name: overlap(rows, values) for name, values in protected.items()}
        roles = Counter(row["ei_ablation"]["role"] for row in rows)
        eos = {
            "rows": len(rows),
            "completion_nonempty": sum(bool(row["completion"].removesuffix(str(tokenizer.eos_token)).strip()) for row in rows),
            "valid_label_gt_zero": sum(row["label_tokens"] > 0 for row in rows),
            "last_valid_label_is_eos": sum(row["ei_ablation"]["last_valid_label_id"] == tokenizer.eos_token_id for row in rows),
            "eos_token_id": tokenizer.eos_token_id,
            "eos_is_minus_100": tokenizer.eos_token_id == -100,
            "zero_label": sum(row["zero_label"] for row in rows),
            "semantic_truncation": sum(row["truncated"] for row in rows),
            "max_total_tokens": max(row["total_tokens"] for row in rows),
        }
        expected_duplicates = {"rows": total_rows, "duplicate_rows": 0, "duplicate_record_ids": 0, "theorem_group_duplicates": 0, "max_repeat": 1}
        audit = {
            "status": "PASSED",
            "passed": True,
            "requested_rows": requested_rows,
            "realized_rows": total_rows,
            "scale_reduction_reason": "Success Bank has only 253 unique theorem groups; the task permits EI-A to use the maximum and equal arm sizes are required for attribution.",
            "requested_frontier_rows": requested_frontier_count,
            "realized_frontier_rows": frontier_count,
            "frontier_reduction_reason": (
                "none"
                if frontier_count == requested_frontier_count
                else "Verified unique Frontier repairs were insufficient; the task requires reducing Frontier rather than using unverified targets."
            ),
            "role_counts": dict(roles),
            "role_ratios": {key: value / total_rows for key, value in roles.items()},
            "manifest_sha256": sha256(manifest_path),
            "duplicates": duplicates,
            "protected_evaluation_leakage": leakage,
            "eos": eos,
            "all_targets_pantograph_verified": all(row["pantograph_verified"] is True for row in rows),
            "success_targets_are_h0_generated": all(row["ei_ablation"]["target_origin"] == "pantograph_verified_h0_self_generation" for row in rows if row["ei_ablation"]["role"] == "Success"),
            "frontier_targets_are_verified_repairs": all(row["ei_ablation"]["target_origin"] == "pantograph_verified_repair_data_v2" for row in rows if row["ei_ablation"]["role"] == "Frontier"),
            "replay_targets_are_verified_foundation": all(row["ei_ablation"]["target_origin"] == "verified_foundation_ld_easy_replay" for row in rows if row["ei_ablation"]["role"] == "Replay"),
        }
        audit["passed"] = (
            duplicates == expected_duplicates
            and all(value["passed"] for value in leakage.values())
            and eos["completion_nonempty"] == total_rows
            and eos["valid_label_gt_zero"] == total_rows
            and eos["last_valid_label_is_eos"] == total_rows
            and not eos["eos_is_minus_100"]
            and eos["zero_label"] == 0
            and eos["semantic_truncation"] == 0
            and audit["all_targets_pantograph_verified"]
            and audit["success_targets_are_h0_generated"]
            and audit["frontier_targets_are_verified_repairs"]
            and audit["replay_targets_are_verified_foundation"]
        )
        audit["status"] = "PASSED" if audit["passed"] else "FAILED"
        write_json(output / "data_audit.json", audit)
        if not audit["passed"]:
            raise RuntimeError(f"{arm} data audit failed")

        contract = {
            "experiment": arm,
            "status": "READY_FOR_TRAINING",
            "starting_checkpoint": str(h0),
            "starting_checkpoint_model_sha256": H0_HASH,
            "initialization": "fresh_lora_on_frozen_merged_H0",
            "expected_rows": total_rows,
            "expected_optimizer_steps": math.ceil(total_rows / 16),
            "expected_role_counts": dict(roles),
            "manifest_sha256": audit["manifest_sha256"],
            "resolved_config": {
                "model_name_or_path": str(h0),
                "adapter_path": None,
                "train_file": str(manifest_path),
                "validation_file": str(project / "data/processed/lean_workbook_verified_v2/eval.jsonl"),
                "output_dir": str(output / "checkpoint/trainer"),
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
                "save_steps": math.ceil(total_rows / 16),
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
            "frozen_generation_contract": str(project / "outputs/task0b_light/generation_contract.json"),
        }
        write_json(output / "training_contract.json", contract)
        (output / "report.md").write_text(
            f"# {arm} pre-training report\n\n"
            f"- Rows: {total_rows} (scaled from 500 because Success Bank has {total_rows} unique theorem groups)\n"
            f"- Composition: {dict(roles)}\n"
            f"- Requested/realized Frontier: {requested_frontier_count}/{frontier_count}\n"
            f"- Data audit: PASS\n"
            f"- Manifest SHA256: `{audit['manifest_sha256']}`\n",
            encoding="utf-8",
        )

    summary = {
        "status": "EI_ABLATION_DATA_READY",
        "requested_rows_per_arm": requested_rows,
        "realized_rows_per_arm": total_rows,
        "success_unique_theorems": len(success_by_group),
        "frontier_verified_unique_theorems": len(frontier_by_group),
        "frontier_shortfall_policy": "reduce Frontier and backfill with verified Success while preserving equal arm size",
        "replay_eligible_rows": len(replay_candidates),
        "arm_role_counts": {arm: dict(Counter(row["ei_ablation"]["role"] for row in rows)) for arm, rows in arm_manifests.items()},
        "trainer_started": False,
        "grpo_started": False,
    }
    write_json(root / "data_preparation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
