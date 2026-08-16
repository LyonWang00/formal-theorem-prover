#!/usr/bin/env python3
"""Freeze checkpoint identities, canary datasets, and length-specific configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.expert_iteration.config import (
    load_expert_iteration_config,
)
from lean_prover.lean_training.expert_iteration.schemas import (
    stable_statement_id,
)


MODEL_PATHS = {
    "M0-ZERO": None,
    "MIX-A-WB100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-A-WB100/final",
    "MIX-B-LD25": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-B-LD25/final",
    "MIX-C-LD50": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-C-LD50/final",
    "MIX-E-LD100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-E-LD100/final",
}
LENGTHS = (256, 512, 1024)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def checkpoint_inventory(root: Path, output: Path, clean_m0: Path) -> None:
    clean_model = clean_m0 / "model.safetensors"
    tokenizer = clean_m0 / "tokenizer.json"
    if not clean_model.is_file() or not tokenizer.is_file():
        raise FileNotFoundError(f"incomplete clean M0 checkpoint: {clean_m0}")
    code_commit = git_head(root)
    rows: list[dict[str, Any]] = []
    for model_name, relative_adapter in MODEL_PATHS.items():
        adapter_path = root / relative_adapter if relative_adapter else None
        checkpoint_path = adapter_path or clean_m0
        weight_path = (
            adapter_path / "adapter_model.safetensors"
            if adapter_path
            else clean_model
        )
        if not weight_path.is_file():
            raise FileNotFoundError(f"missing checkpoint weight: {weight_path}")
        contract_path = (
            adapter_path / "checkpoint_contract.json" if adapter_path else None
        )
        contract = (
            json.loads(contract_path.read_text(encoding="utf-8"))
            if contract_path and contract_path.is_file()
            else {}
        )
        base_checkpoint = str(
            contract.get("base_checkpoint")
            or contract.get("fixed_anchor_checkpoint")
            or clean_m0
        )
        if adapter_path and Path(base_checkpoint).name != clean_m0.name:
            resolved = (root / base_checkpoint).resolve()
            if resolved != clean_m0.resolve():
                raise ValueError(
                    f"{model_name} does not identify clean M0 as its base: "
                    f"{base_checkpoint}"
                )
        row = {
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": sha256(weight_path),
            "checkpoint_weight_file": str(weight_path),
            "base_checkpoint": str(clean_m0),
            "adapter_identity": (
                {
                    "adapter_path": str(adapter_path),
                    "adapter_config_hash": sha256(
                        adapter_path / "adapter_config.json"
                    ),
                    "contract_hash": (
                        sha256(contract_path)
                        if contract_path and contract_path.is_file()
                        else None
                    ),
                }
                if adapter_path
                else None
            ),
            "tokenizer_hash": sha256(tokenizer),
            "training_code_commit": str(
                contract.get("training_code_commit") or code_commit
            ),
        }
        rows.append(row)
    if len({row["checkpoint_hash"] for row in rows}) != len(rows):
        raise ValueError("checkpoint weight identities are not unique")
    write_json(output / "audit/checkpoints.json", rows)


def quantile_bin(value: float, boundaries: tuple[float, float]) -> str:
    if value <= boundaries[0]:
        return "short"
    if value <= boundaries[1]:
        return "medium"
    return "long"


def _terciles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0)
    return (
        ordered[round((len(ordered) - 1) / 3)],
        ordered[round(2 * (len(ordered) - 1) / 3)],
    )


def round_robin_strata(
    items: list[dict[str, Any]],
    *,
    count: int,
    signature,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[tuple(signature(item))].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets, key=lambda key: repr(key))
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        next_keys: list[tuple[Any, ...]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise ValueError(f"requested {count} rows, selected {len(selected)}")
    return selected


def freeze_wb_canary(
    *,
    gate_path: Path,
    attempts_path: Path,
    generations_path: Path,
    output_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(gate_path)
    attempts = read_jsonl(attempts_path)
    generations = read_jsonl(generations_path)
    solved = defaultdict(bool)
    for row in attempts:
        solved[str(row["problem_id"])] |= bool(row.get("success"))
    lengths: dict[str, list[int]] = defaultdict(list)
    for row in generations:
        lengths[str(row["statement_id"])].append(
            int((row.get("metadata") or {}).get("completion_tokens") or 0)
        )
    candidates: list[dict[str, Any]] = []
    for source_row in rows:
        row = json.loads(json.dumps(source_row))
        row["data_role"] = "benchmark"
        source = str(row.get("source_name") or row.get("source") or "unknown")
        statement = str(row.get("lean_statement") or row.get("statement") or "")
        statement_id = stable_statement_id(source, str(row.get("id") or ""), statement)
        mean_length = (
            sum(lengths[statement_id]) / len(lengths[statement_id])
            if lengths[statement_id]
            else 0.0
        )
        row.setdefault("metadata", {})["length_ablation"] = {
            "dataset": "WB_LENGTH_CANARY_50",
            "parent_manifest": str(gate_path),
            "m0_solved_at_4": bool(solved[statement_id]),
            "m0_mean_output_tokens": mean_length,
        }
        candidates.append(row)
    boundaries = _terciles(
        [
            float(row["metadata"]["length_ablation"]["m0_mean_output_tokens"])
            for row in candidates
        ]
    )
    for row in candidates:
        row["metadata"]["length_ablation"]["m0_output_length_bin"] = quantile_bin(
            float(row["metadata"]["length_ablation"]["m0_mean_output_tokens"]),
            boundaries,
        )
    selected = round_robin_strata(
        candidates,
        count=50,
        seed=seed,
        signature=lambda row: (
            row["metadata"]["length_ablation"]["m0_solved_at_4"],
            row["metadata"]["length_ablation"]["m0_output_length_bin"],
            str(row.get("category") or "unknown"),
        ),
    )
    write_jsonl(output_path, selected)
    return selected


def freeze_ld_canary(
    *,
    holdout_path: Path,
    full_pool_path: Path,
    output_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    holdout = read_jsonl(holdout_path)
    full_by_id = {str(row["id"]): row for row in read_jsonl(full_pool_path)}
    enriched: list[dict[str, Any]] = []
    proof_lengths: list[float] = []
    premise_counts: list[float] = []
    for source_row in holdout:
        row = json.loads(json.dumps(source_row))
        original = full_by_id.get(str(row.get("id"))) or {}
        premises = list(original.get("premises") or [])
        proof_tokens = int(
            original.get("label_tokens")
            or row.get("label_tokens")
            or len(str(row.get("proof") or "").split())
        )
        premise_count = len(premises)
        same_file_count = sum(bool(item.get("is_same_file")) for item in premises)
        source_file = str(row.get("source_file") or original.get("source_file") or "")
        domain = (
            source_file.split("/")[1]
            if source_file.startswith("Mathlib/") and "/" in source_file[8:]
            else "unknown"
        )
        row["data_role"] = "benchmark"
        row.setdefault("metadata", {})["length_ablation"] = {
            "dataset": "LD_LENGTH_CANARY_64",
            "parent_manifest": str(holdout_path),
            "reference_proof_tokens": proof_tokens,
            "proof_style": str(
                row.get("proof_style") or original.get("proof_style") or "unknown"
            ),
            "premise_count": premise_count,
            "same_file_premise_count": same_file_count,
            "mathlib_domain": domain,
        }
        proof_lengths.append(float(proof_tokens))
        premise_counts.append(float(premise_count))
        enriched.append(row)
    proof_boundaries = _terciles(proof_lengths)
    premise_boundaries = _terciles(premise_counts)
    for row in enriched:
        audit = row["metadata"]["length_ablation"]
        audit["reference_proof_length_bin"] = quantile_bin(
            float(audit["reference_proof_tokens"]), proof_boundaries
        )
        audit["premise_count_bin"] = quantile_bin(
            float(audit["premise_count"]), premise_boundaries
        )
        audit["has_same_file_premise"] = bool(audit["same_file_premise_count"])
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_file[str(row.get("source_file") or "")].append(row)
    if len(by_file) != 8 or any(len(rows) < 8 for rows in by_file.values()):
        raise ValueError(
            f"expected eight holdout source groups with >=8 rows: "
            f"{ {key: len(value) for key, value in by_file.items()} }"
        )
    selected: list[dict[str, Any]] = []
    for index, (source_file, rows) in enumerate(sorted(by_file.items())):
        selected.extend(
            round_robin_strata(
                rows,
                count=8,
                seed=seed + index,
                signature=lambda row: (
                    row["metadata"]["length_ablation"]["proof_style"],
                    row["metadata"]["length_ablation"][
                        "reference_proof_length_bin"
                    ],
                    row["metadata"]["length_ablation"]["premise_count_bin"],
                    row["metadata"]["length_ablation"][
                        "has_same_file_premise"
                    ],
                ),
            )
        )
    write_jsonl(output_path, selected)
    return selected


def freeze_configs(base_config: Path, output: Path) -> None:
    config = load_expert_iteration_config(base_config)
    payload = config.model_dump(mode="json")
    payload["discovery"]["generation"]["max_model_len"] = 2048
    for length in LENGTHS:
        length_payload = json.loads(json.dumps(payload))
        length_payload["discovery"]["generation"]["max_new_tokens"] = length
        path = output / f"manifests/config_L{length}.json"
        write_json(path, length_payload)


def summarize_canaries(
    output: Path,
    wb_rows: list[dict[str, Any]],
    ld_rows: list[dict[str, Any]],
) -> None:
    combined = []
    for row in wb_rows:
        copy = json.loads(json.dumps(row))
        copy.setdefault("metadata", {})["length_ablation_dataset"] = "wb"
        combined.append(copy)
    for row in ld_rows:
        copy = json.loads(json.dumps(row))
        copy.setdefault("metadata", {})["length_ablation_dataset"] = "ld"
        combined.append(copy)
    combined_path = output / "manifests/combined_length_canary_114.jsonl"
    write_jsonl(combined_path, combined)
    paths = {
        "WB_LENGTH_CANARY_50": output / "manifests/wb_length_canary_50.jsonl",
        "LD_LENGTH_CANARY_64": output / "manifests/ld_length_canary_64.jsonl",
        "COMBINED_LENGTH_CANARY_114": combined_path,
    }
    audit = {
        name: {"path": str(path), "rows": len(read_jsonl(path)), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    audit["paired_seed"] = 20260901
    audit["max_new_tokens"] = list(LENGTHS)
    audit["cache_isolation"] = (
        "generation_id includes max_new_tokens and each length has a distinct output/cache directory"
    )
    write_json(output / "audit/frozen_canaries.json", audit)
    report = [
        "# Frozen length-ablation canaries",
        "",
        "The parent evaluation manifests are unchanged. These derived manifests are "
        "frozen before any length result is observed.",
        "",
        "| Dataset | Rows | SHA-256 |",
        "|---|---:|---|",
    ]
    for name, value in audit.items():
        if not isinstance(value, dict):
            continue
        report.append(f"| {name} | {value['rows']} | `{value['sha256']}` |")
    (output / "audit/frozen_canaries.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    for path in paths.values():
        path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/expert_iteration.round1.yaml"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    clean_m0 = (
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_inventory(root, output, clean_m0)
    wb_rows = freeze_wb_canary(
        gate_path=root / "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
        attempts_path=(
            root
            / "outputs/wb_ld_small_sft_ablation/evaluation/zero_step/wb_gate150/attempts.jsonl"
        ),
        generations_path=(
            root
            / "outputs/wb_ld_small_sft_ablation/evaluation/zero_step/wb_gate150/generations.jsonl"
        ),
        output_path=output / "manifests/wb_length_canary_50.jsonl",
        seed=20260801,
    )
    ld_rows = freeze_ld_canary(
        holdout_path=(
            root
            / "outputs/wb_ld_small_sft_ablation/evaluation/ld_holdout_manifest.jsonl"
        ),
        full_pool_path=(
            root
            / "outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl"
        ),
        output_path=output / "manifests/ld_length_canary_64.jsonl",
        seed=20260801,
    )
    summarize_canaries(output, wb_rows, ld_rows)
    freeze_configs(root / args.base_config, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoints": len(MODEL_PATHS),
                "wb_canary": len(wb_rows),
                "ld_canary": len(ld_rows),
                "lengths": list(LENGTHS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
