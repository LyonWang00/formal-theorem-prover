"""Prepare and hard-gate Expert Iteration Round 0 discovery data.

This stage is deliberately generation-free and trainer-free.  It selects 500
Pantograph-attested statements that are disjoint from H0/M0 training and every
protected evaluation set, removes all reference proofs from the public
discovery manifest, and resolves the frozen Task0 generation contract onto the
H0 merged checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.audit_stage2_ratio_phase_c1_feasibility import (
    EXPECTED_ENVIRONMENT_HASH,
    EXPECTED_GENERATION_CONFIG_HASH,
    add_identity,
    disjoint_from_sets,
    duplicate_audit,
    identity_sets,
    overlap,
    proof,
    read_json,
    read_jsonl,
    record_id,
    sha256,
    statement,
    write_json,
    write_jsonl,
)
from scripts.prepare_ld_easy_pilot import add_source_template


SELECTION_SEED = 20260810
DISCOVERY_ROWS = 500
LD_EASY_TARGET = 100
EXPECTED_H0_MODEL_HASH = (
    "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
)
PARAMETER_KEYS = (
    "backend",
    "checkpoint_type",
    "dtype",
    "load_in_4bit",
    "prompt_format",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "repetition_penalty",
    "max_new_tokens",
    "eos_token_id",
    "pad_token_id",
    "stop_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/expert_iteration/round0"),
    )
    parser.add_argument(
        "--discovery-rows",
        type=int,
        default=DISCOVERY_ROWS,
        help="Number of frozen discovery statements to select.",
    )
    parser.add_argument(
        "--ld-easy-target",
        type=int,
        default=LD_EASY_TARGET,
        help="Maximum number of legal unused LD-easy statements in this batch.",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Additional JSONL manifest whose identities must be excluded.",
    )
    return parser.parse_args()


def stable_rank(value: str, namespace: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{namespace}:{value}".encode("utf-8")
    ).hexdigest()


def contract_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("generation_config_sha256", None)
    raw = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prompt(informal: str, lean_statement: str) -> str:
    return (
        "### Informal statement\n"
        + informal.strip()
        + "\n\n### Lean statement\n"
        + lean_statement.strip()
        + "\n\n### Lean proof\n"
    )


def proof_tokens(row: dict[str, Any]) -> int:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    explicit = (
        row.get("label_tokens")
        or row.get("proof_token_estimate")
        or metrics.get("proof_tokens")
    )
    if explicit:
        return int(explicit)
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_']*|\S", proof(row)))


def length_bin(row: dict[str, Any]) -> str:
    tokens = proof_tokens(row)
    if tokens <= 8:
        return "1-8"
    if tokens <= 16:
        return "9-16"
    if tokens <= 32:
        return "17-32"
    if tokens <= 64:
        return "33-64"
    return "65+"


def wb_domain(row: dict[str, Any]) -> str:
    text = " ".join(
        (str(row.get("informal_statement") or ""), statement(row))
    ).lower()
    rules = (
        ("geometry", ("triangle", "circle", "angle", "area", "distance")),
        ("number_theory", ("prime", "divis", "mod ", "nat", "integer")),
        ("calculus", ("deriv", "integral", "continuous", "limit")),
        ("combinatorics", ("card", "finite", "choose", "permutation")),
        ("inequality", ("inequality", "≤", "≥", "<", ">", "nlinarith")),
        ("algebra", ("equation", "polynomial", "ring", "field", "solve")),
    )
    for name, needles in rules:
        if any(needle in text for needle in needles):
            return name
    return "general"


def ld_domain(row: dict[str, Any]) -> str:
    source_file = str(row.get("source_file") or "")
    parts = source_file.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Mathlib" else "unknown"


def tactic_style(row: dict[str, Any]) -> str:
    explicit = str(row.get("proof_style") or "")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    explicit = explicit or str(metrics.get("proof_style") or "")
    if explicit:
        return explicit
    body = proof(row).lower()
    for name in (
        "simp",
        "norm_num",
        "linarith",
        "nlinarith",
        "ring",
        "omega",
        "aesop",
        "exact",
        "rfl",
    ):
        if re.search(rf"\b{re.escape(name)}\b", body):
            return name
    return "other"


def verified_ld(row: dict[str, Any]) -> bool:
    verification = row.get("verification")
    return bool(
        isinstance(verification, dict)
        and str(row.get("verification_status") or "")
        in {"verified", "verified_default_timeout"}
        and verification.get("compile_success") is True
        and verification.get("timed_out") is False
        and not verification.get("error_category")
        and str(
            row.get("environment_hash")
            or verification.get("environment_hash")
            or ""
        )
        == EXPECTED_ENVIRONMENT_HASH
        and statement(row)
        and proof(row)
    )


def verified_wb(row: dict[str, Any]) -> bool:
    return bool(
        record_id(row)
        and statement(row)
        and proof(row)
        and row.get("pantograph_verified") is True
        and row.get("proof_verified") is True
        and str(row.get("verification_status") or "") == "verified"
        and str(row.get("environment_hash") or "") == EXPECTED_ENVIRONMENT_HASH
    )


def select_diverse(
    rows: list[dict[str, Any]], *, count: int, source: str
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"{source}: need {count}, only {len(rows)} legal rows")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = wb_domain(row) if source == "WB" else ld_domain(row)
        buckets[(domain, length_bin(row), tactic_style(row))].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: stable_rank(record_id(row), f"{source}:{key}"))
    keys = sorted(buckets, key=lambda key: stable_rank(repr(key), f"{source}:bucket"))
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        remaining: list[tuple[str, str, str]] = []
        for key in keys:
            if len(selected) >= count:
                break
            selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    if len(selected) != count:
        raise RuntimeError(f"{source}: diverse selection failed")
    return selected


def materialize_wb(row: dict[str, Any]) -> dict[str, Any]:
    lean_statement = statement(row)
    output = {
        "id": record_id(row),
        "record_id": record_id(row),
        "source": "WB",
        "source_name": "WB",
        "data_role": "discovery",
        "category": wb_domain(row),
        "informal_statement": str(row.get("informal_statement") or ""),
        "lean_statement": lean_statement,
        "statement": lean_statement,
        "imports": ["Mathlib"],
        "prompt": prompt(str(row.get("informal_statement") or ""), lean_statement),
        "theorem_group_id": str(row.get("statement_id") or ""),
        "qualified_name": str(
            row.get("source_declaration") or row.get("qualified_name") or ""
        ),
        "discovery_source": "verified_unused_wb",
        "proof_free": True,
        "reference_attestation_id": str(row.get("attestation_id") or ""),
        "reference_assembled_source_hash": str(
            row.get("assembled_source_hash") or ""
        ),
        "reference_proof_hash": hashlib.sha256(
            re.sub(r"\s+", " ", proof(row)).strip().encode("utf-8")
        ).hexdigest(),
        "proof_token_length": proof_tokens(row),
        "proof_length_bin": length_bin(row),
        "tactic_style": tactic_style(row),
        "selection_seed": SELECTION_SEED,
    }
    return output


def materialize_ld(row: dict[str, Any]) -> dict[str, Any]:
    lean_statement = str(row.get("training_statement") or statement(row)).strip()
    output = {
        "id": record_id(row),
        "record_id": record_id(row),
        "source": "LD-easy",
        "source_name": "LD-easy",
        "data_role": "discovery",
        "category": ld_domain(row),
        "lean_statement": lean_statement,
        "statement": lean_statement,
        # Source-faithful LD verification must preload exactly the record's
        # imports.  Preloading all of Mathlib and then replaying the source
        # template causes systematic duplicate-declaration errors.
        "imports": list(row.get("imports") or []),
        "prompt": prompt("", lean_statement),
        "theorem_group_id": str(row.get("theorem_group_id") or ""),
        "qualified_name": str(row.get("qualified_name") or ""),
        "source_file": str(row.get("source_file") or ""),
        "discovery_source": "verified_unused_ld_easy",
        "proof_free": True,
        "reference_attestation_id": str(row.get("attestation_id") or ""),
        "reference_assembled_source_hash": str(
            row.get("assembled_source_hash") or ""
        ),
        "reference_proof_hash": hashlib.sha256(
            re.sub(r"\s+", " ", proof(row)).strip().encode("utf-8")
        ).hexdigest(),
        "proof_token_length": proof_tokens(row),
        "proof_length_bin": length_bin(row),
        "tactic_style": tactic_style(row),
        "selection_seed": SELECTION_SEED,
    }
    add_source_template(output, row)
    return output


def histogram(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "unknown") for row in rows))


def main() -> None:
    args = parse_args()
    if args.discovery_rows <= 0:
        raise ValueError("--discovery-rows must be positive")
    if args.ld_easy_target < 0 or args.ld_easy_target > args.discovery_rows:
        raise ValueError("--ld-easy-target must be within [0, discovery-rows]")
    project = args.project.resolve()
    root = args.output if args.output.is_absolute() else project / args.output
    manifest_path = root / "discovery/discovery_manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen discovery: {manifest_path}")
    for directory in (
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
        (root / directory).mkdir(parents=True, exist_ok=True)

    paths = {
        "wb": project
        / "data/processed/lean_workbook_verified_v2/audit/verified_records.jsonl",
        "ld": project
        / "outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl",
        "labels": project
        / "outputs/ld_manual_classification_freeze/two_round_v1/combined_manual_labels_1725.jsonl",
        "m0_train": project
        / "outputs/wb_ld_budget_support_replay_ablation/manifests/fixed_wb/ADDON-B-WB2000-LD1000.jsonl",
        "h0_train": project
        / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/manifest.jsonl",
        "protected": project
        / "outputs/stage2_sft_incremental_ablation/shared/protected_eval_index.json",
        "ld_medium_holdout": project
        / "outputs/stage2_data_ratio_ablation/phaseC3_ld_medium/evaluation/datasets/ld_medium_holdout64.jsonl",
        "h0": project
        / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED",
        "parent_contract": project / "outputs/task0b_light/generation_contract.json",
        "parent_runtime": project
        / "outputs/task0b_light/task0b_light_runtime_config.json",
        "eval160": project / "data/processed/lean_workbook_verified_v2/eval.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"EI Round 0 inputs missing: {missing}")
    if sha256(paths["h0"] / "model.safetensors") != EXPECTED_H0_MODEL_HASH:
        raise RuntimeError("H0 merged checkpoint drifted")

    protected_index = read_json(paths["protected"])
    protected_rows: dict[str, list[dict[str, Any]]] = {}
    for name, entry in protected_index.items():
        value = entry.get("path")
        if value and Path(str(value)).is_file():
            protected_rows[name] = read_jsonl(Path(str(value)))
    protected_rows["ld_medium_holdout64"] = read_jsonl(paths["ld_medium_holdout"])
    all_eval = [row for rows in protected_rows.values() for row in rows]
    additional_exclusions: list[dict[str, Any]] = []
    resolved_exclude_manifests: list[Path] = []
    for value in args.exclude_manifest:
        path = Path(value)
        if not path.is_absolute():
            path = project / path
        if not path.is_file():
            raise FileNotFoundError(f"additional exclusion manifest missing: {path}")
        resolved_exclude_manifests.append(path)
        additional_exclusions.extend(read_jsonl(path))
    prior_training = read_jsonl(paths["m0_train"]) + read_jsonl(paths["h0_train"])
    prior_training.extend(additional_exclusions)
    forbidden_sets = identity_sets(prior_training + all_eval)

    wb_all = [row for row in read_jsonl(paths["wb"]) if verified_wb(row)]
    wb_legal = [row for row in wb_all if disjoint_from_sets(row, forbidden_sets)]

    labels = read_jsonl(paths["labels"])
    easy_ids = {
        str(row.get("sample_id") or "")
        for row in labels
        if row.get("difficulty") == "easy"
        and row.get("reviewed_by") == "codex"
        and row.get("trainable_for_short_whole_proof") is True
    }
    ld_all = [
        row
        for row in read_jsonl(paths["ld"])
        if record_id(row) in easy_ids and verified_ld(row)
    ]
    ld_legal = [row for row in ld_all if disjoint_from_sets(row, forbidden_sets)]
    if args.ld_easy_target and not ld_legal:
        raise RuntimeError(
            "no legal unused LD-easy row remains after training/evaluation exclusion"
        )
    ld_count = min(args.ld_easy_target, len(ld_legal))
    wb_count = args.discovery_rows - ld_count
    selected_raw = select_diverse(wb_legal, count=wb_count, source="WB")
    if ld_count:
        selected_raw += select_diverse(ld_legal, count=ld_count, source="LD-easy")
    selected_raw.sort(key=lambda row: stable_rank(record_id(row), "manifest-order"))
    raw_gate = duplicate_audit(selected_raw)
    raw_leakage = {
        "prior_training": overlap(selected_raw, prior_training),
        **{name: overlap(selected_raw, rows) for name, rows in protected_rows.items()},
    }
    if (
        len(selected_raw) != args.discovery_rows
        or raw_gate["duplicate_rows"]
        or raw_gate["duplicate_record_ids"]
        or raw_gate["theorem_group_duplicates"]
        or raw_gate["max_repeat"] != 1
        or not all(value["passed"] for value in raw_leakage.values())
    ):
        raise RuntimeError("EI Round 0 discovery isolation/duplicate gate failed")

    raw_by_id = {record_id(row): row for row in selected_raw}
    wb_ids = {record_id(row) for row in wb_legal}
    materialized = [
        materialize_wb(raw_by_id[record_id(raw)])
        if record_id(raw) in wb_ids
        else materialize_ld(raw_by_id[record_id(raw)])
        for raw in selected_raw
    ]
    forbidden_proof_fields = {
        "proof",
        "completion",
        "reference_proof",
        "training_proof",
        "text",
        "raw_proof",
    }
    if any(forbidden_proof_fields & set(row) for row in materialized):
        raise RuntimeError("public discovery manifest contains a proof field")
    if any(
        not row.get("prompt")
        or str(raw_by_id[record_id(row)].get("proof") or "").strip()
        in str(row.get("prompt"))
        for row in materialized
    ):
        raise RuntimeError("reference proof leaked into a discovery prompt")
    write_jsonl(manifest_path, materialized)

    parent = read_json(paths["parent_contract"])
    if parent.get("generation_config_sha256") != EXPECTED_GENERATION_CONFIG_HASH:
        raise RuntimeError("Task0 canonical generation contract drifted")
    for filename, key in (
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("chat_template.jinja", "chat_template_sha256"),
    ):
        if sha256(paths["h0"] / filename) != parent[key]:
            raise RuntimeError(f"H0 {filename} differs from canonical tokenizer asset")
    resolved = dict(parent)
    resolved.update(
        {
            "contract_name": "expert_iteration_round0_h0_merged_transformers_v1",
            "frozen_model": "H0-No-Hard",
            "checkpoint_path": str(paths["h0"]),
            "checkpoint_model_sha256": EXPECTED_H0_MODEL_HASH,
            "tokenizer_path": str(paths["h0"]),
            "parent_generation_contract_path": str(paths["parent_contract"]),
            "parent_generation_contract_sha256": EXPECTED_GENERATION_CONFIG_HASH,
            "candidates_per_statement": 8,
            "candidate_seed_note": (
                "Eight sampled sequences share the fixed per-prompt generator seed "
                "and are distinguished by sample_index 0..7."
            ),
        }
    )
    # candidates_per_statement is run shape, not a decoding parameter in the
    # parent task.  All frozen decoding parameters remain byte-for-byte equal.
    resolved["generation_config_sha256"] = contract_hash(resolved)
    for key in PARAMETER_KEYS:
        if resolved.get(key) != parent.get(key):
            raise RuntimeError(f"generation parameter drift at {key}")
    contract_path = root / "discovery/generation_contract.json"
    write_json(contract_path, resolved)

    runtime = read_json(paths["parent_runtime"])
    runtime["run_name"] = "expert_iteration_round0"
    runtime["output_dir"] = str(root / "runtime")
    runtime["data"].update(
        {
            "train_path": str(paths["h0_train"]),
            "eval_path": str(paths["eval160"]),
            "discovery_path": str(manifest_path),
            "monitor_path": None,
            "benchmark_path": None,
            "benchmark_dev_path": None,
            "benchmark_test_path": None,
            "overlap_policy": "error",
        }
    )
    runtime["initial_model"].update(
        {
            "base_model": str(paths["h0"]),
            "sft_adapter": None,
            "tokenizer": str(paths["h0"]),
            "skip_initial_sft": True,
            "checkpoint_strategy": "fixed_anchor",
            "merged_sft0_path": str(paths["h0"]),
        }
    )
    generation = runtime["discovery"]["generation"]
    generation.update(
        {
            "backend": "transformers",
            "samples_per_statement": 8,
            "temperature": parent["temperature"],
            "top_p": parent["top_p"],
            "max_new_tokens": parent["max_new_tokens"],
            "batch_size": 1,
            "load_in_4bit": False,
            "generation_contract_path": str(contract_path),
            "generation_contract_sha256": resolved["generation_config_sha256"],
        }
    )
    runtime["discovery"]["statements_per_iteration"] = args.discovery_rows
    runtime["discovery"]["sampling_budget"] = {
        "new": 8,
        "frontier": 8,
        "unsolved": 8,
        "audit": 8,
    }
    runtime["discovery"]["iteration_bucket_counts"] = {
        "0": {"new": args.discovery_rows, "frontier": 0, "unsolved": 0, "audit": 0}
    }
    runtime["discovery"]["iteration_sampling_budget"] = {
        "0": {"new": 8, "frontier": 8, "unsolved": 8, "audit": 8}
    }
    runtime["discovery"]["iteration_generation_seeds"] = {"0": SELECTION_SEED}
    runtime["verification"].update(
        {"imports": ["Mathlib"], "timeout_seconds": 60, "num_workers": 2}
    )
    runtime["training"].update(
        {
            "use_qlora": True,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "learning_rate": 1e-5,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 1024,
            "packing": False,
            "eval_strategy": "epoch",
            "eval_steps": None,
            "save_strategy": "epoch",
        }
    )
    runtime["monitor"]["enabled"] = False
    runtime["benchmark"]["enabled"] = False
    runtime["execution"]["generation_timeout_seconds"] = 43200
    runtime_path = root / "discovery/runtime_config.json"
    write_json(runtime_path, runtime)

    source_counts = Counter(row["source"] for row in materialized)
    audit = {
        "status": "EI_ROUND0_DISCOVERY_FROZEN",
        "selection_seed": SELECTION_SEED,
        "rows": len(materialized),
        "source_counts": dict(source_counts),
        "available_after_gates": {"WB": len(wb_legal), "LD-easy": len(ld_legal)},
        "requested_ld_easy": args.ld_easy_target,
        "selected_ld_easy": ld_count,
        "ld_shortfall_filled_by_verified_wb": args.ld_easy_target - ld_count,
        "additional_exclusion_manifests": [
            str(path) for path in resolved_exclude_manifests
        ],
        "domains": histogram(materialized, "category"),
        "proof_length_bins": histogram(materialized, "proof_length_bin"),
        "tactic_styles": histogram(materialized, "tactic_style"),
        "duplicate_gate": raw_gate,
        "leakage": raw_leakage,
        "proof_free": True,
        "reference_proof_in_prompt": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "h0_checkpoint": str(paths["h0"]),
        "h0_checkpoint_sha256": EXPECTED_H0_MODEL_HASH,
        "parent_generation_config_sha256": EXPECTED_GENERATION_CONFIG_HASH,
        "resolved_generation_config_sha256": resolved["generation_config_sha256"],
        "generation_parameters_unchanged": {
            key: resolved.get(key) for key in PARAMETER_KEYS
        },
        "candidate_seed": SELECTION_SEED,
        "candidates_per_statement": 8,
        "candidate_target": args.discovery_rows * 8,
    }
    write_json(root / "audit/discovery_audit.json", audit)
    write_json(
        root / "status.json",
        {
            "status": "DISCOVERY_MANIFEST_READY",
            "generation_started": False,
            "trainer_started": False,
            "grpo_started": False,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
