#!/usr/bin/env python3
"""Build all fixed, unique B/C ablation manifests from one master permutation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from lean_prover.lean_training.data.random_manifest import (
    canonical_sha256,
    deterministic_permutation,
    file_sha256,
    master_permutation_metadata,
    proof_homogeneity,
    read_jsonl,
    record_id,
    statement_id,
    validate_manifest,
    validate_verified_row,
    write_json,
    write_jsonl,
)


ANCHOR_SELECTION_SEED = 20260720
TRAIN_SEED = 20260721
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "generation_seed": 4401,
    "prompt_template": "plain_text_lean_sections_v1",
}


def directory_contract(path: Path) -> dict[str, Any]:
    required = ("model.safetensors", "config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"M0 checkpoint is missing {missing}: {path}")
    tokenizer_files = sorted(
        file
        for file in path.iterdir()
        if file.is_file()
        and (
            "token" in file.name.lower()
            or file.name in {"special_tokens_map.json", "chat_template.jinja"}
        )
    )
    tokenizer_hashes = {file.name: file_sha256(file) for file in tokenizer_files}
    return {
        "absolute_path": str(path.resolve()),
        "model_hash": file_sha256(path / "model.safetensors"),
        "config_hash": file_sha256(path / "config.json"),
        "tokenizer_hash": canonical_sha256(tokenizer_hashes),
        "tokenizer_files": tokenizer_hashes,
        "generation_config": GENERATION_CONFIG,
        "generation_config_hash": canonical_sha256(GENERATION_CONFIG),
    }


def validate_no_split_conflicts(
    rows: list[dict[str, Any]], forbidden: set[str], *, name: str
) -> None:
    overlap = sorted({statement_id(row) for row in rows} & forbidden)
    if overlap:
        raise ValueError(f"{name} overlaps eval/monitor/benchmark: {overlap[:5]}")


def source_statistics(rows: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    sources = Counter(str(row["sampling_source"]) for row in rows)
    source_label_tokens: Counter[str] = Counter()
    source_proof_tokens: dict[str, list[int]] = {}
    overlength = []
    for row in rows:
        source = str(row["sampling_source"])
        completion = str(row.get("completion") or row.get("proof") or "")
        label_tokens = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        total_tokens = len(
            tokenizer(
                str(row.get("prompt") or "") + completion,
                add_special_tokens=True,
            )["input_ids"]
        )
        if total_tokens > 1024:
            overlength.append((record_id(row), total_tokens))
        source_label_tokens[source] += label_tokens
        source_proof_tokens.setdefault(source, []).append(label_tokens)
    if overlength:
        raise ValueError(f"manifest contains overlength rows: {overlength[:5]}")
    total_label_tokens = sum(source_label_tokens.values())
    return {
        "source_rows": dict(sources),
        "label_token_count": total_label_tokens,
        "source_label_tokens": dict(source_label_tokens),
        "source_label_token_share": {
            source: count / max(1, total_label_tokens)
            for source, count in source_label_tokens.items()
        },
        "source_mean_proof_tokens": {
            source: sum(values) / len(values)
            for source, values in source_proof_tokens.items()
        },
        "overlength_rows": 0,
    }


def build_frontier(
    *, expert_pool: Path, discovery_verifications: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    for row in read_jsonl(discovery_verifications):
        sid = str(row["statement_id"])
        attempts[sid] += 1
        success[sid] += int(bool(row.get("verified")))
    if set(attempts.values()) != {4}:
        raise ValueError("Discovery source must contain exactly four attempts/statement")
    frontier_ids = sorted(sid for sid in attempts if success[sid] in (1, 2))
    pool = read_jsonl(expert_pool)
    by_statement = {statement_id(row): row for row in pool}
    if len(by_statement) != len(pool):
        raise ValueError("fixed primary expert pool repeats a statement")
    if set(by_statement) != set(frontier_ids):
        raise ValueError(
            "fixed primary expert pool does not exactly cover 1/4 and 2/4 frontier"
        )
    rows = []
    for sid in frontier_ids:
        row = dict(by_statement[sid])
        validate_verified_row(row, allow_origin_discovery=True)
        row.pop("sample_weight", None)
        # TRL selects language-modeling mode whenever a ``text`` column exists.
        # Mixed Anchor/Expert manifests must expose only prompt/completion so
        # completion-only masking cannot be bypassed by a nullable text column.
        row.pop("text", None)
        row["statement_id"] = sid
        row["sampling_source"] = "expert"
        row["sampling_experiment"] = "frontier_primary_fixed"
        row["frontier_success_count_at_4"] = success[sid]
        row["truncated"] = False
        rows.append(row)
    counts = Counter(success[sid] for sid in frontier_ids)
    return rows, {
        "source_path": str(expert_pool.resolve()),
        "source_sha256": file_sha256(expert_pool),
        "verification_path": str(discovery_verifications.resolve()),
        "verification_sha256": file_sha256(discovery_verifications),
        "frontier_rows": len(rows),
        "success_count_at_4": {str(key): value for key, value in sorted(counts.items())},
        "one_primary_proof_per_statement": True,
        "selection": "reuse_existing_fixed_primary_training_proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--monitor", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--frontier-expert", required=True, type=Path)
    parser.add_argument("--discovery-verifications", required=True, type=Path)
    parser.add_argument("--discovery-gate", required=True, type=Path)
    parser.add_argument("--m0", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-manifests", action="store_true")
    args = parser.parse_args()

    output = args.output_root
    manifests = output / "manifests"
    anchor_source = args.anchor.expanduser().resolve()
    anchor_rows = read_jsonl(anchor_source)
    eligible = []
    for row in anchor_rows:
        validate_verified_row(row, allow_origin_discovery=False)
        eligible.append(row)
    if len(eligible) != 3000:
        raise ValueError(f"expected 3000 eligible Anchor rows, got {len(eligible)}")
    permutation = deterministic_permutation(eligible, seed=ANCHOR_SELECTION_SEED)
    permutation = [
        {
            **row,
            "sampling_source": "anchor",
            "sampling_experiment": "anchor_master_permutation",
            "truncated": False,
        }
        for row in permutation
    ]
    forbidden = {
        statement_id(row)
        for path in (args.eval, args.monitor, args.benchmark)
        for row in read_jsonl(path)
    }
    validate_no_split_conflicts(permutation, forbidden, name="anchor master")
    frontier, frontier_metadata = build_frontier(
        expert_pool=args.frontier_expert,
        discovery_verifications=args.discovery_verifications,
    )
    validate_no_split_conflicts(frontier, forbidden, name="frontier expert")
    if {statement_id(row) for row in permutation} & {
        statement_id(row) for row in frontier
    }:
        raise ValueError("Anchor and Frontier Expert statements overlap")

    expert_count = len(frontier)
    b2_anchor_count = 1000 - expert_count
    c3_anchor_count = 39 if expert_count == 153 else max(1, round(expert_count * 0.25))
    def arm_rows(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**row, "sampling_experiment": name} for row in rows]

    arms = {
        "B1": arm_rows("B1", permutation[:1000]),
        "B2": arm_rows("B2", permutation[:b2_anchor_count] + frontier),
        "C1": arm_rows("C1", permutation[:160]),
        "C2": arm_rows("C2", frontier),
        "C3": arm_rows("C3", permutation[:c3_anchor_count] + frontier),
    }
    if [record_id(row) for row in arms["B2"][:b2_anchor_count]] != [
        record_id(row) for row in arms["B1"][:b2_anchor_count]
    ]:
        raise ValueError("B2 Anchor is not the required B1 prefix")

    tokenizer = AutoTokenizer.from_pretrained(str(args.m0), trust_remote_code=True)
    arm_paths = {
        "B1": manifests / "B1_anchor_1000.jsonl",
        "B2_anchor": manifests / f"B2_anchor_{b2_anchor_count}.jsonl",
        "frontier": manifests / "frontier_expert.jsonl",
        "B2": manifests / "B2_train_1000.jsonl",
        "C1": manifests / "C1_anchor_160.jsonl",
        "C2": manifests / "C2_frontier_expert.jsonl",
        "C3_anchor": manifests / f"C3_anchor_{c3_anchor_count}.jsonl",
        "C3": manifests / f"C3_train_{len(arms['C3'])}.jsonl",
    }
    validation = {
        name: {
            **validate_manifest(rows, allow_origin_discovery=name in {"B2", "C2", "C3"}),
            **source_statistics(rows, tokenizer),
        }
        for name, rows in arms.items()
    }
    if args.dry_run:
        print(json.dumps({"validation": validation, "frontier": frontier_metadata}, indent=2))
        return

    if output.exists():
        if not args.refresh_manifests:
            raise FileExistsError(f"output already exists: {output}")
        if not manifests.is_dir():
            raise FileNotFoundError(
                f"cannot refresh missing experiment manifest directory: {manifests}"
            )
    else:
        output.mkdir(parents=True)
        manifests.mkdir(parents=True)
    master_hash = write_jsonl(
        manifests / "anchor_master_permutation.jsonl", permutation
    )
    write_json(
        manifests / "anchor_master_permutation_ids.json",
        {
            "ordered_record_ids": [record_id(row) for row in permutation],
            "ordered_statement_ids": [statement_id(row) for row in permutation],
        },
    )
    write_json(
        manifests / "anchor_master_permutation_metadata.json",
        master_permutation_metadata(
            source_path=anchor_source,
            rows=permutation,
            seed=ANCHOR_SELECTION_SEED,
            manifest_sha256=master_hash,
        ),
    )
    hashes = {
        "B1": write_jsonl(arm_paths["B1"], arms["B1"]),
        "B2_anchor": write_jsonl(
            arm_paths["B2_anchor"], arms["B2"][:b2_anchor_count]
        ),
        "frontier": write_jsonl(arm_paths["frontier"], frontier),
        "B2": write_jsonl(arm_paths["B2"], arms["B2"]),
        "C1": write_jsonl(arm_paths["C1"], arms["C1"]),
        "C2": write_jsonl(arm_paths["C2"], arms["C2"]),
        "C3_anchor": write_jsonl(
            arm_paths["C3_anchor"], arms["C3"][:c3_anchor_count]
        ),
        "C3": write_jsonl(arm_paths["C3"], arms["C3"]),
    }

    gate_rows = sorted(
        read_jsonl(args.discovery_gate), key=lambda row: statement_id(row)
    )
    if len(gate_rows) != 150:
        raise ValueError("fixed Discovery Gate must contain 150 rows")
    random.Random(ANCHOR_SELECTION_SEED).shuffle(gate_rows)
    early_gate = gate_rows[:50]
    gate_path = manifests / "early_retention_gate_50.jsonl"
    gate_hash = write_jsonl(gate_path, early_gate)
    write_json(
        manifests / "early_retention_gate_metadata.json",
        {
            "source_path": str(args.discovery_gate.resolve()),
            "source_sha256": file_sha256(args.discovery_gate),
            "selection_seed": ANCHOR_SELECTION_SEED,
            "selection": "sort statement_id then deterministic shuffle and take first 50",
            "statement_ids": [statement_id(row) for row in early_gate],
            "manifest_sha256": gate_hash,
        },
    )
    experiment_manifest = {
        "anchor_selection_seed": ANCHOR_SELECTION_SEED,
        "training_seed": TRAIN_SEED,
        "sampling_strategy": "fixed_manifest_without_replacement",
        "m0": directory_contract(args.m0.resolve()),
        "anchor_source": str(anchor_source),
        "anchor_source_sha256": file_sha256(anchor_source),
        "eligible_anchor_rows": len(permutation),
        "frontier": frontier_metadata,
        "composition": {
            "B1": {"anchor": 1000, "expert": 0, "total": 1000},
            "B2": {
                "anchor": b2_anchor_count,
                "expert": expert_count,
                "total": 1000,
            },
            "C1": {"anchor": 160, "expert": 0, "total": 160},
            "C2": {"anchor": 0, "expert": expert_count, "total": expert_count},
            "C3": {
                "anchor": c3_anchor_count,
                "expert": expert_count,
                "total": c3_anchor_count + expert_count,
            },
        },
        "paths": {key: str(path.resolve()) for key, path in arm_paths.items()},
        "sha256": hashes,
        "validation": validation,
        "proof_homogeneity": {
            name: proof_homogeneity(rows) for name, rows in arms.items()
        },
        "early_retention_gate": str(gate_path.resolve()),
        "early_retention_gate_sha256": gate_hash,
    }
    write_json(manifests / "experiment_manifest.json", experiment_manifest)
    print(json.dumps(experiment_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
