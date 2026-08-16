"""Build and audit the fixed datasets for B2 expanded validation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

from lean_prover.lean_training.evaluation.expanded_validation import (
    PRIMARY_SEED,
    REPLICATION_SEED,
    SEED_DERIVATION_VERSION,
    assert_disjoint,
    normalized_statement_hash,
    original_success_counts,
    proof_free_prompt,
    read_jsonl,
    sha256_file,
    restore_full500_rows,
    statement_id,
    write_json_atomic,
    write_jsonl_atomic,
)


def directory_identity(path: Path, patterns: tuple[str, ...]) -> dict[str, Any]:
    files = [candidate for pattern in patterns for candidate in sorted(path.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no identity files in {path}")
    return {
        "absolute_path": str(path.resolve()),
        "files": {
            file.name: {"size": file.stat().st_size, "sha256": sha256_file(file)}
            for file in files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/b2_expanded_validation"))
    parser.add_argument(
        "--verified-root", type=Path, default=Path("data/processed/lean_workbook_verified_v2")
    )
    parser.add_argument("--round1", type=Path, default=Path("outputs/expert_iteration_round1"))
    parser.add_argument(
        "--ablation-root", type=Path, default=Path("outputs/expert_sft_no_replacement_ablation")
    )
    parser.add_argument(
        "--m0", type=Path,
        default=Path("outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"),
    )
    args = parser.parse_args()
    datasets = args.root / "datasets"

    train = read_jsonl(args.verified_root / "train.jsonl")
    evaluation = read_jsonl(args.verified_root / "eval.jsonl")
    discovery700 = read_jsonl(args.verified_root / "discovery.jsonl")
    monitor = read_jsonl(args.round1 / "inputs/verified/monitor.jsonl")
    benchmark = read_jsonl(args.round1 / "inputs/verified/benchmark.jsonl")
    generations = read_jsonl(args.round1 / "iteration_000/discovery/generations.jsonl")
    verifications = read_jsonl(args.round1 / "iteration_000/discovery/verifications.jsonl")
    b2_manifest_path = args.ablation_root / "manifests/B2_train_1000.jsonl"
    b2_manifest = read_jsonl(b2_manifest_path)

    full500, legacy_mapping = restore_full500_rows(discovery700, generations)
    full_ids = [statement_id(row) for row in full500]
    expert_ids = {
        statement_id(row) for row in b2_manifest if row.get("sampling_source") == "expert"
    }
    if len(expert_ids) != 153:
        raise ValueError(f"expected 153 B2 expert statements, found {len(expert_ids)}")
    full_set = set(full_ids)
    if not expert_ids <= full_set:
        raise ValueError("B2 expert statements are not a subset of first-round Full500")
    expert_seen = [row for row in full500 if statement_id(row) in expert_ids]
    non_expert = [row for row in full500 if statement_id(row) not in expert_ids]
    if len(expert_seen) != 153 or len(non_expert) != 347:
        raise ValueError("Full500 expert/non-expert partition has unexpected sizes")

    full_hashes = {normalized_statement_hash(row) for row in full500}
    strict_unseen = [
        row
        for row in discovery700
        if statement_id(row) not in full_set
        and normalized_statement_hash(row) not in full_hashes
    ]
    protected = {
        "train": train,
        "eval": evaluation,
        "full500": full500,
        "b2_training_manifest": b2_manifest,
        "monitor": monitor,
        "benchmark": benchmark,
    }
    overlap_reports = [
        assert_disjoint("strict_unseen", strict_unseen, name, rows)
        for name, rows in protected.items()
    ]
    invalid = [row for row in overlap_reports if not row["valid"]]
    if invalid:
        raise ValueError(f"strict unseen role/hash overlap: {invalid}")
    if len(strict_unseen) > 200:
        raise ValueError(f"strict unseen unexpectedly exceeds 200: {len(strict_unseen)}")

    prompt_audit = {}
    for name, rows in {
        "full500": full500,
        "strict_unseen": strict_unseen,
        "monitor": monitor,
        "benchmark": benchmark,
    }.items():
        prompt_audit[name] = {
            "rows": len(rows),
            "proof_free_prompts": sum(bool(proof_free_prompt(row)) for row in rows),
        }

    counts = original_success_counts(generations, verifications)
    buckets = {
        index: [row for row in full500 if counts[statement_id(row)] == index]
        for index in range(5)
    }

    outputs = {
        "full500.jsonl": full500,
        "full500_expert_seen.jsonl": expert_seen,
        "full500_non_expert.jsonl": non_expert,
        "strict_unseen_discovery.jsonl": strict_unseen,
        "monitor_minif2f_valid_64.jsonl": monitor,
        "benchmark_minif2f_test_96.jsonl": benchmark,
        **{f"full500_m0_bucket_{index}_of_4.jsonl": rows for index, rows in buckets.items()},
    }
    for filename, rows in outputs.items():
        write_jsonl_atomic(datasets / filename, rows)
    write_json_atomic(datasets / "full500_legacy_id_mapping.json", legacy_mapping)

    b2_checkpoint = args.ablation_root / "checkpoints/B2_anchor_expert_1000/step_63"
    b2_training_metrics = json.loads(
        (args.ablation_root / "checkpoints/B2_anchor_expert_1000/training_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint_contract = json.loads(
        (b2_checkpoint / "checkpoint_contract.json").read_text(encoding="utf-8")
    )
    run_manifest = {
        "experiment": "b2_expanded_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "models": {
            "M0": {
                **directory_identity(args.m0, ("model*.safetensors", "config.json", "tokenizer.json")),
                "role": "clean_M0_merged_anchor",
            },
            "B2": {
                **directory_identity(
                    b2_checkpoint,
                    ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json"),
                ),
                "role": "B2_step_63_candidate",
                "training_step": 63,
                "parent_m0_hash": b2_training_metrics["m0_model_hash"],
                "training_manifest_path": str(b2_manifest_path.resolve()),
                "training_manifest_sha256": sha256_file(b2_manifest_path),
                "lora": b2_training_metrics["lora"],
                "checkpoint_contract": checkpoint_contract,
                "merge_manifest": None,
                "merge_status": "not_merged; evaluated as M0 plus fixed LoRA adapter",
            },
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "samples_per_statement": 4,
            "statement_batch_size": 100,
            "primary_seed": PRIMARY_SEED,
            "replication_seed": REPLICATION_SEED,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
        },
        "datasets": {
            name: {
                "path": str((datasets / name).resolve()),
                "rows": len(rows),
                "sha256": sha256_file(datasets / name),
            }
            for name, rows in outputs.items()
        },
        "partition": {
            "full500": len(full500),
            "expert_seen": len(expert_seen),
            "non_expert": len(non_expert),
            "strict_unseen": len(strict_unseen),
            "expert_seen_intersection_non_expert": sorted(
                {statement_id(row) for row in expert_seen}
                & {statement_id(row) for row in non_expert}
            ),
            "expert_seen_union_non_expert_equals_full500": (
                {statement_id(row) for row in expert_seen}
                | {statement_id(row) for row in non_expert}
            ) == full_set,
            "original_m0_bucket_counts": {
                f"{index}_of_4": len(rows) for index, rows in buckets.items()
            },
            "legacy_id_mapping_rows": len(legacy_mapping),
            "legacy_id_mapping_sha256": sha256_file(
                datasets / "full500_legacy_id_mapping.json"
            ),
        },
        "role_overlap_audit": overlap_reports,
        "prompt_audit": prompt_audit,
        "source_counts": {
            "train": len(train),
            "eval": len(evaluation),
            "verified_discovery": len(discovery700),
            "round1_generation_rows": len(generations),
            "round1_verification_rows": len(verifications),
            "monitor": len(monitor),
            "benchmark": len(benchmark),
            "b2_training_manifest": len(b2_manifest),
            "b2_training_sources": dict(
                Counter(str(row.get("sampling_source")) for row in b2_manifest)
            ),
        },
        "forbidden_actions": {
            "training": False,
            "proof_bank_updates": False,
            "round2": False,
            "m2": False,
        },
    }
    write_json_atomic(args.root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
