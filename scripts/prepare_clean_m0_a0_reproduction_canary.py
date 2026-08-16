#!/usr/bin/env python3
"""Freeze the low-variance canary inputs for the CLEAN-M0/A0 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    project = Path(".").resolve()
    output = (
        project
        / "outputs/clean_m0_a0_reproduction_audit/evaluation/canary"
    )
    output.mkdir(parents=True, exist_ok=True)
    source_dataset = (
        project / "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl"
    )
    source_config = (
        project
        / "outputs/ld_length_difficulty_pipeline/length_ablation/manifests/"
        "config_L256.json"
    )
    rows = [
        json.loads(line)
        for line in source_dataset.open(encoding="utf-8-sig")
        if line.strip()
    ][:24]
    if len(rows) != 24 or len({row["id"] for row in rows}) != 24:
        raise ValueError("canary must contain 24 unique frozen WB statements")
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["length_ablation_dataset"] = "wb"
        metadata["reproduction_canary_source"] = str(source_dataset)
        row["metadata"] = metadata
        row["proof"] = None
        row["completion"] = None
        row["reference_proof"] = None
        row["has_reference_proof"] = False
        row["data_role"] = "benchmark"
    canary_dataset = output / "wb_canary_24.jsonl"
    with canary_dataset.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = json.loads(source_config.read_text(encoding="utf-8-sig"))
    generation = config["discovery"]["generation"]
    generation["temperature"] = 0.0
    generation["top_p"] = 1.0
    generation["samples_per_statement"] = 1
    canary_config = output / "config_greedy.json"
    write_json(canary_config, config)
    write_json(
        output / "canary_identity.json",
        {
            "dataset_path": str(canary_dataset),
            "dataset_sha256": sha256(canary_dataset),
            "source_dataset_path": str(source_dataset),
            "source_dataset_sha256": sha256(source_dataset),
            "selection": "first_24_before_any_reproduction_evaluation",
            "statement_ids": [row["id"] for row in rows],
            "reference_proofs_removed": True,
            "config_path": str(canary_config),
            "config_sha256": sha256(canary_config),
            "source_config_path": str(source_config),
            "source_config_sha256": sha256(source_config),
            "generation_override": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 256,
                "samples_per_statement": 1,
            },
        },
    )
    print(output / "canary_identity.json")


if __name__ == "__main__":
    main()
