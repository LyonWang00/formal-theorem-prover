"""Build and hard-gate the 2,000-row Phase C3 LD-medium curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    EXPECTED_ENVIRONMENT_HASH,
    EXPECTED_GENERATION_CONFIG_HASH,
    add_identity,
    disjoint_from_sets,
    duplicate_audit,
    identity,
    identity_sets,
    overlap,
    proof,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    write_json,
    write_jsonl,
)
from scripts.prepare_ld_easy_pilot import add_source_template
from scripts.prepare_wb_ld_small_sft_ablation import format_row


SELECTION_SEED = 20260805
TRAINING_SEED = 42
EXPECTED_M0_MODEL_HASH = (
    "6d80ac7b0034e46f51554f7ba4b66609e83dc9b9ce9c6c643fcbbcfabc5b6f81"
)
EXPECTED_COUNTS = {
    "New WB": 1700,
    "LD-medium": 200,
    "Replay": 100,
}
EXPECTED_BUCKETS = {
    "New WB": 1700,
    "LD-medium": 200,
    "Stable/Core": 12,
    "Frontier": 47,
    "Additional verified WB replay": 41,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("outputs/stage2_data_ratio_ablation/phaseC3_ld_medium"),
    )
    return parser.parse_args()


def stable_rank(value: str, namespace: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def source_file(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return str(row.get("source_file") or metrics.get("source_file") or "")


def domain(row: dict[str, Any]) -> str:
    parts = source_file(row).split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Mathlib" else "unknown"


def proof_style(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return str(row.get("proof_style") or metrics.get("proof_style") or "unknown")


def proof_tokens(row: dict[str, Any]) -> int:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return int(metrics.get("proof_tokens") or row.get("label_tokens") or 0)


def read_jsonl_utf8(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verified_ld_source(row: dict[str, Any]) -> bool:
    verification = row.get("verification")
    if not isinstance(verification, dict):
        return False
    status = str(row.get("verification_status") or "")
    environment = str(
        row.get("environment_hash")
        or row.get("verification_environment_hash")
        or verification.get("environment_hash")
        or ""
    )
    return all(
        (
            status in {"verified", "verified_default_timeout"},
            verification.get("compile_success") is True,
            verification.get("timed_out") is False,
            not verification.get("error_category"),
            environment == EXPECTED_ENVIRONMENT_HASH,
            bool(row.get("training_statement") or row.get("statement")),
            bool(row.get("training_proof") or row.get("proof")),
            bool(source_file(row)),
        )
    )


def select_diverse(
    rows: list[dict[str, Any]], *, count: int, namespace: str
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"{namespace}: need {count} rows, only {len(rows)} available")
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        length = proof_tokens(row)
        length_bin = "short" if length <= 16 else "medium" if length <= 48 else "long"
        buckets[(domain(row), source_file(row), proof_style(row), length_bin)].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: stable_rank(record_id(row), f"{namespace}:{key}"))
    keys = sorted(buckets, key=lambda key: stable_rank(repr(key), f"{namespace}:bucket"))
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[str, str, str, str]] = []
        for key in keys:
            if len(selected) >= count:
                break
            selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise RuntimeError(f"{namespace}: diverse selection did not fill target")
    return selected


def choose_source_disjoint_holdout(
    rows: list[dict[str, Any]], *, holdout_rows: int, retain_train_rows: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[source_file(row)].append(row)
    files = sorted(
        by_file,
        key=lambda value: (
            len(by_file[value]),
            stable_rank(value, "ld-medium-holdout-file"),
        ),
    )
    chosen: list[str] = []
    candidate_count = 0
    for value in files:
        proposed = chosen + [value]
        excluded = sum(len(by_file[item]) for item in proposed)
        if len(rows) - excluded < retain_train_rows:
            continue
        chosen.append(value)
        candidate_count += len(by_file[value])
        if candidate_count >= holdout_rows:
            candidates = [row for item in chosen for row in by_file[item]]
            holdout = select_diverse(
                candidates, count=holdout_rows, namespace="ld-medium-holdout"
            )
            remaining = [row for row in rows if source_file(row) not in set(chosen)]
            return holdout, remaining, sorted(chosen)
    raise RuntimeError("unable to construct source-file-disjoint LD-medium holdout")


def histogram(values: Iterable[int]) -> dict[str, int]:
    result = Counter()
    for value in values:
        if value <= 8:
            result["1-8"] += 1
        elif value <= 16:
            result["9-16"] += 1
        elif value <= 32:
            result["17-32"] += 1
        elif value <= 64:
            result["33-64"] += 1
        else:
            result["65+"] += 1
    return dict(result)


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.phase_root if args.phase_root.is_absolute() else project / args.phase_root
    if (root / "manifest.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite C3 manifest: {root / 'manifest.jsonl'}")
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "h0_source": project
        / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/manifest.jsonl",
        "labels": project
        / "outputs/ld_manual_classification_freeze/two_round_v1/combined_manual_labels_1725.jsonl",
        "freeze": project
        / "outputs/ld_manual_classification_freeze/two_round_v1/freeze_manifest.json",
        "ld_pool": project
        / "outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl",
        "generation_contract": project / "outputs/task0b_light/generation_contract.json",
        "protected_index": project
        / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "m0_merged": project
        / "outputs/stage2_sft_incremental_ablation/shared/checkpoints/M0-ADDON-B-FROZEN-MERGED",
        "validation": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"C3 inputs missing: {missing}")
    if sha256(paths["m0_merged"] / "model.safetensors") != EXPECTED_M0_MODEL_HASH:
        raise RuntimeError("frozen M0 merged checkpoint drifted")
    generation = read_json(paths["generation_contract"])
    if generation.get("generation_config_sha256") != EXPECTED_GENERATION_CONFIG_HASH:
        raise RuntimeError("canonical generation contract drifted")
    freeze = read_json(paths["freeze"])
    if freeze.get("status") != "FROZEN_READ_ONLY" or (freeze.get("combined") or {}).get("difficulty_counts") != {
        "difficult": 108,
        "easy": 1068,
        "exclude": 34,
        "medium": 515,
    }:
        raise RuntimeError("two-round manual LD classification freeze drifted")

    labels = read_jsonl_utf8(paths["labels"])
    if len(labels) != 1725:
        raise RuntimeError("manual LD label count drifted")
    medium_labels = [
        row
        for row in labels
        if row.get("difficulty") == "medium" and row.get("reviewed_by") == "codex"
    ]
    if len(medium_labels) != 515:
        raise RuntimeError("reviewed LD-medium count drifted")
    if any(
        not isinstance(row.get("evidence"), dict)
        or not str(row.get("reason_summary") or "").strip()
        for row in medium_labels
    ):
        raise RuntimeError("LD-medium includes a non-semantic or incomplete review")

    ld_pool = read_jsonl_utf8(paths["ld_pool"])
    pool_by_id = {record_id(row): row for row in ld_pool}
    if len(pool_by_id) != len(ld_pool):
        raise RuntimeError("LeanDojo-v2 final pool contains duplicate record IDs")
    missing_medium = [row["sample_id"] for row in medium_labels if row["sample_id"] not in pool_by_id]
    if missing_medium:
        raise RuntimeError(f"manual LD-medium labels missing from final pool: {missing_medium[:10]}")
    annotations = {str(row["sample_id"]): row for row in medium_labels}
    medium_raw: list[dict[str, Any]] = []
    for rid, annotation in annotations.items():
        raw = dict(pool_by_id[rid])
        if not verified_ld_source(raw):
            continue
        if str(annotation.get("qualified_name") or "") != str(raw.get("qualified_name") or ""):
            raise RuntimeError(f"LD classification/source qualified-name mismatch: {rid}")
        raw["manual_difficulty_annotation"] = annotation
        medium_raw.append(raw)

    protected_index = read_json(paths["protected_index"])
    required_protected = {
        "wb_unseen_holdout150",
        "ld_easy64",
        "monitor64",
        "hard_ld128",
        "full500",
        "strict_unseen200",
    }
    if not required_protected.issubset(protected_index):
        raise RuntimeError("shared protected evaluation index is incomplete")
    protected_rows: dict[str, list[dict[str, Any]]] = {}
    retention: list[dict[str, Any]] = []
    for name, entry in protected_index.items():
        value = entry.get("path")
        if not value:
            continue
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"protected dataset missing: {name}: {path}")
        rows = read_jsonl(path)
        if entry.get("role") == "retention":
            retention.extend(rows)
        elif entry.get("role") in {"true_holdout", "future_true_holdout"}:
            protected_rows[name] = rows
    all_existing_protected = [row for rows in protected_rows.values() for row in rows]
    protected_sets = identity_sets(all_existing_protected)
    protected_clean_medium = [
        row for row in medium_raw if disjoint_from_sets(row, protected_sets)
    ]
    if len(protected_clean_medium) < 264:
        raise RuntimeError(
            f"only {len(protected_clean_medium)} protected-clean verified LD-medium rows"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["m0_merged"]), local_files_only=True, trust_remote_code=False
    )
    if (
        tokenizer.eos_token != "<|im_end|>"
        or tokenizer.eos_token_id != 151645
        or tokenizer.pad_token_id != 151643
    ):
        raise RuntimeError("frozen tokenizer EOS/PAD identity drifted")

    # Tokenize all reviewed, verified candidates first; overlength rows are not eligible.
    tokenized_medium: list[dict[str, Any]] = []
    raw_by_id: dict[str, dict[str, Any]] = {}
    for raw in protected_clean_medium:
        formatted = format_row(raw, tokenizer)
        if formatted.get("truncated") or int(formatted.get("total_tokens") or 0) > 1024:
            continue
        rid = record_id(formatted)
        formatted["verification_status"] = "verified"
        formatted["environment_hash"] = EXPECTED_ENVIRONMENT_HASH
        formatted["verification_environment_hash"] = EXPECTED_ENVIRONMENT_HASH
        formatted["pantograph_verified"] = True
        formatted["proof_verified"] = True
        formatted["statement_verified"] = True
        formatted["qualified_name"] = str(raw.get("qualified_name") or "")
        formatted["manual_difficulty_annotation"] = annotations[rid]
        tokenized_medium.append(formatted)
        raw_by_id[rid] = raw

    holdout_training_schema, train_candidates, holdout_files = choose_source_disjoint_holdout(
        tokenized_medium, holdout_rows=64, retain_train_rows=200
    )
    ld_train = select_diverse(
        train_candidates, count=200, namespace="ld-medium-train"
    )
    if set(map(source_file, holdout_training_schema)) & set(map(source_file, ld_train)):
        raise RuntimeError("LD-medium source-file-disjoint split failed")
    if not overlap(ld_train, holdout_training_schema)["passed"]:
        raise RuntimeError("LD-medium six-axis train/holdout disjointness failed")

    holdout: list[dict[str, Any]] = []
    for formatted in holdout_training_schema:
        rid = record_id(formatted)
        raw = raw_by_id[rid]
        row = dict(formatted)
        add_source_template(row, raw)
        row["difficulty"] = "medium"
        row["category"] = "LD-medium-holdout64"
        row["reference_proof_sha256"] = hashlib.sha256(
            proof(formatted).encode("utf-8")
        ).hexdigest()
        row["reference_verification"] = {
            "compile_success": True,
            "timed_out": False,
            "environment_hash": EXPECTED_ENVIRONMENT_HASH,
            "verification_status": "verified_default_timeout",
        }
        for key in ("proof", "completion", "reference_proof", "text"):
            row.pop(key, None)
        holdout.append(row)
    if any(key in row for row in holdout for key in ("proof", "completion", "reference_proof", "text")):
        raise RuntimeError("LD-medium holdout is not proof-free")

    h0_rows = read_jsonl(paths["h0_source"])
    if len(h0_rows) != 2000:
        raise RuntimeError("H0 audited source manifest drifted")
    by_h0_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in h0_rows:
        annotation = row.get("hard_a_ablation") or {}
        by_h0_role[str(annotation.get("top_level_role") or "")].append(row)
    if {key: len(value) for key, value in by_h0_role.items()} != {
        "New WB": 1800,
        "Frontier": 47,
        "Additional verified WB replay": 141,
        "Stable/Core": 12,
    }:
        raise RuntimeError("H0 role composition drifted")

    new_holdout_sets = identity_sets(holdout_training_schema)
    forbidden_train = identity_sets(all_existing_protected + holdout_training_schema)
    stable = by_h0_role["Stable/Core"]
    frontier = by_h0_role["Frontier"]
    if any(not disjoint_from_sets(row, forbidden_train) for row in stable + frontier):
        raise RuntimeError("mandatory Stable/Core or Frontier replay overlaps a protected set")
    additional = sorted(
        (
            row
            for row in by_h0_role["Additional verified WB replay"]
            if disjoint_from_sets(row, forbidden_train)
        ),
        key=lambda row: stable_rank(record_id(row), "additional-replay"),
    )[:41]
    new_wb = sorted(
        (
            row
            for row in by_h0_role["New WB"]
            if disjoint_from_sets(row, forbidden_train)
        ),
        key=lambda row: stable_rank(record_id(row), "new-wb"),
    )[:1700]
    if len(new_wb) != 1700 or len(additional) != 41:
        raise RuntimeError("C3 WB/replay pool cannot fill the approved composition")

    source_rows: list[dict[str, Any]] = []
    source_proofs: dict[str, str] = {}

    def append_rows(rows: Iterable[dict[str, Any]], role: str, bucket: str) -> None:
        for original in rows:
            row = dict(original)
            row.pop("hard_a_ablation", None)
            row["stage2_phase_c3"] = {
                "top_level_role": role,
                "bucket": bucket,
                "selection_seed": SELECTION_SEED,
                "foundation_dominant": True,
                "contains_hard_a_b_c": False,
            }
            row["completion"] = proof(original)
            row["text"] = str(row.get("prompt") or "") + row["completion"]
            source_rows.append(row)
            source_proofs[record_id(row)] = proof(original)

    append_rows(new_wb, "New WB", "New WB")
    append_rows(ld_train, "LD-medium", "LD-medium")
    append_rows(stable, "Replay", "Stable/Core")
    append_rows(frontier, "Replay", "Frontier")
    append_rows(additional, "Replay", "Additional verified WB replay")
    source_rows.sort(key=lambda row: stable_rank(record_id(row), "manifest-order"))
    if len(source_rows) != 2000 or len(source_proofs) != 2000:
        raise RuntimeError("C3 source manifest is not 2,000 unique rows")

    role_counts = Counter(row["stage2_phase_c3"]["top_level_role"] for row in source_rows)
    bucket_counts = Counter(row["stage2_phase_c3"]["bucket"] for row in source_rows)
    if dict(role_counts) != EXPECTED_COUNTS or dict(bucket_counts) != EXPECTED_BUCKETS:
        raise RuntimeError(f"C3 composition drifted: {role_counts}; {bucket_counts}")
    duplicates = duplicate_audit(source_rows)
    if duplicates != {
        "rows": 2000,
        "duplicate_rows": 0,
        "duplicate_record_ids": 0,
        "theorem_group_duplicates": 0,
        "max_repeat": 1,
    }:
        raise RuntimeError(f"C3 duplicate gate failed: {duplicates}")

    effective_rows: list[dict[str, Any]] = []
    eos_counts = Counter()
    valid_label_counts: list[int] = []
    eos_token = str(tokenizer.eos_token)
    eos_id = int(tokenizer.eos_token_id)
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
        full_ids = tokenizer(prompt + row["completion"], add_special_tokens=True)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"prompt prefix mismatch: {rid}")
        labels_for_loss = full_ids[len(prompt_ids) :]
        if not labels_for_loss:
            eos_counts["zero_label"] += 1
        if labels_for_loss and labels_for_loss[-1] == eos_id:
            eos_counts["last_valid_label_is_eos"] += 1
            eos_counts["eos_supervised"] += 1
        if len(full_ids) > 1024:
            eos_counts["semantic_truncation"] += 1
        if completion.strip():
            eos_counts["completion_nonempty"] += 1
        valid_label_counts.append(len(labels_for_loss))
        row["stage2_phase_c3"] = dict(row["stage2_phase_c3"])
        row["stage2_phase_c3"].update(
            {
                "valid_label_count_with_eos": len(labels_for_loss),
                "last_valid_label_id": labels_for_loss[-1] if labels_for_loss else None,
            }
        )
        effective_rows.append(row)

    proof_modifications = sum(
        proof(row).removesuffix(eos_token).strip() != source_proofs[record_id(row)]
        for row in effective_rows
    )
    eos_audit = {
        "status": "PASSED",
        "rows": 2000,
        "completion_nonempty": eos_counts["completion_nonempty"],
        "valid_label_gt_zero": sum(value > 0 for value in valid_label_counts),
        "last_valid_label_is_eos": eos_counts["last_valid_label_is_eos"],
        "eos_supervised": eos_counts["eos_supervised"],
        "eos_token_id": eos_id,
        "eos_is_minus_100": eos_id == -100,
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
        or eos_audit["eos_is_minus_100"]
    ):
        raise RuntimeError(f"C3 EOS gate failed: {eos_audit}")
    if proof_modifications:
        raise RuntimeError("C3 reference proofs changed")

    leakage = {name: overlap(source_rows, rows) for name, rows in protected_rows.items()}
    leakage["ld_medium_holdout64"] = overlap(source_rows, holdout_training_schema)
    if not all(result["passed"] for result in leakage.values()):
        raise RuntimeError(f"C3 protected leakage gate failed: {leakage}")
    retention_overlap = overlap(source_rows, retention)

    holdout_path = root / "evaluation/datasets/ld_medium_holdout64.jsonl"
    source_path = root / "source_manifest.jsonl"
    manifest_path = root / "manifest.jsonl"
    write_jsonl(holdout_path, holdout)
    write_jsonl(source_path, source_rows)
    write_jsonl(manifest_path, effective_rows)

    ld_split_audit = {
        "status": "PASSED",
        "manual_freeze_status": freeze["status"],
        "manual_labels_total": len(labels),
        "manual_medium_total": len(medium_labels),
        "verified_medium": len(medium_raw),
        "protected_clean_medium": len(protected_clean_medium),
        "token_length_eligible_medium": len(tokenized_medium),
        "train_rows": len(ld_train),
        "holdout_rows": len(holdout),
        "holdout_proof_free": True,
        "source_file_disjoint": True,
        "holdout_source_files": holdout_files,
        "train_source_files": sorted(set(map(source_file, ld_train))),
        "train_holdout_six_axis": overlap(ld_train, holdout_training_schema),
        "train_domains": dict(Counter(map(domain, ld_train))),
        "holdout_domains": dict(Counter(map(domain, holdout_training_schema))),
        "train_proof_token_histogram": histogram(map(proof_tokens, ld_train)),
        "holdout_proof_token_histogram": histogram(map(proof_tokens, holdout_training_schema)),
        "train_proof_styles": dict(Counter(map(proof_style, ld_train))),
        "holdout_proof_styles": dict(Counter(map(proof_style, holdout_training_schema))),
        "manual_labels_sha256": sha256(paths["labels"]),
        "manual_freeze_sha256": sha256(paths["freeze"]),
        "ld_pool_sha256": sha256(paths["ld_pool"]),
        "holdout_sha256": sha256(holdout_path),
    }
    leakage_audit = {
        "status": "PASSED",
        "checks": list(identity({}).keys()),
        "protected": leakage,
        "wb_train_retention150_allowed_overlap": retention_overlap,
    }
    data_audit = {
        "phase": "C3",
        "model_name": "S2-LD-Medium-Conservative",
        "status": "PASSED_READY_FOR_TRAINING",
        "rows": 2000,
        "role_counts": dict(role_counts),
        "bucket_counts": dict(bucket_counts),
        "duplicates": duplicates,
        "proof_modifications": proof_modifications,
        "all_pantograph_verified": all(
            row.get("pantograph_verified") is True and row.get("proof_verified") is True
            for row in source_rows
        ),
        "contains_hard_a_b_c": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "selection_seed": SELECTION_SEED,
        "training_seed": TRAINING_SEED,
        "ld_medium": ld_split_audit,
    }
    write_json(root / "data_audit.json", data_audit)
    write_json(root / "leakage_audit.json", leakage_audit)
    write_json(root / "eos_audit.json", eos_audit)
    write_json(root / "audit/ld_medium_split_audit.json", ld_split_audit)

    gate = {
        "passed": True,
        "phase": "C3",
        "rows": 2000,
        "effective_manifest": str(manifest_path),
        "effective_manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": sha256(source_path),
        "m0_model_sha256": EXPECTED_M0_MODEL_HASH,
        "generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH,
        "role_counts": dict(role_counts),
        "bucket_counts": dict(bucket_counts),
        "duplicates": duplicates,
        "leakage": leakage_audit,
        "eos": eos_audit,
        "proof_modifications": proof_modifications,
        "ld_medium_split": ld_split_audit,
    }
    gate_path = root / "audit/phaseC3_manifest_gate.json"
    write_json(gate_path, gate)

    training_contract = {
        "phase": "C3",
        "model_name": "S2-LD-Medium-Conservative",
        "status": "READY_FOR_TRAINING",
        "initialization": "M0-ADDON-B-FROZEN merged checkpoint + fresh LoRA",
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "expected_rows": 2000,
        "expected_optimizer_steps": math.ceil(2000 / 16),
        "manifest_gate_path": str(gate_path),
        "annotation_key": "stage2_phase_c3",
        "expected_role_counts": EXPECTED_COUNTS,
        "expected_bucket_counts": EXPECTED_BUCKETS,
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
    print(json.dumps(data_audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
