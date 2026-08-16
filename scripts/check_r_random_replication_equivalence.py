#!/usr/bin/env python3
"""Check 24-prompt greedy merged/unmerged equivalence for S2 and S3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.prepare_r_random_replication import BASE, NEW_MODELS, OUTPUT
from scripts.run_r_random_replication_evaluation import (
    CORE,
    checkpoint_paths,
    evaluation_config,
    read_json,
    read_jsonl,
    run_logged,
    sha256,
    write_json,
    write_jsonl,
)


def prepare(project: Path, root: Path) -> tuple[Path, Path]:
    source = project / CORE["wb_unseen150"][0]
    manifest = root / "evaluation/checkpoint_equivalence/greedy24_manifest.jsonl"
    write_jsonl(manifest, read_jsonl(source)[:24])
    config = read_json(evaluation_config(project))
    generation = config["discovery"]["generation"]
    generation["samples_per_statement"] = 1
    generation["temperature"] = 0.0
    generation["top_p"] = 0.95
    # A merge-equivalence check must compare the same numerical model path:
    # BF16(base + dynamic LoRA) versus BF16(merged weights). Quantizing both
    # inputs would instead compare quantize(base) + LoRA with
    # quantize(base + LoRA), which is not an equivalence-preserving operation.
    generation["load_in_4bit"] = False
    for key in config["discovery"]["sampling_budget"]:
        config["discovery"]["sampling_budget"][key] = 1
    config_path = (
        root / "evaluation/checkpoint_equivalence/greedy_generation_config.json"
    )
    write_json(config_path, config)
    write_json(
        root / "evaluation/checkpoint_equivalence/canary_identity.json",
        {
            "source_path": str(source),
            "source_sha256": sha256(source),
            "selection": "first_24_in_frozen_order",
            "manifest_path": str(manifest),
            "manifest_sha256": sha256(manifest),
            "rows": 24,
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 256,
            "seed": 20261701,
            "load_in_4bit": False,
            "numerical_comparison": (
                "bf16_dynamic_lora_vs_bf16_merged; formal evaluations retain "
                "the frozen 4-bit generation contract"
            ),
            "pantograph_outcome_definition": "verified_success_boolean",
            "pantograph_status_taxonomy": "reported separately as a diagnostic",
        },
    )
    return manifest, config_path


def run_one(
    *,
    project: Path,
    root: Path,
    label: str,
    model: Path,
    adapter: Path | None,
    manifest: Path,
    config: Path,
    output: Path,
    suffix: str,
) -> None:
    if (output / "benchmark_summary.json").is_file():
        return
    command = [
        sys.executable,
        "-m",
        "scripts.run_wb_ld_ablation_evaluation",
        "--config",
        str(config),
        "--role",
        "benchmark",
        "--model",
        str(model),
        "--dataset",
        str(manifest),
        "--output",
        str(output),
        "--seed",
        "20261701",
        "--samples-per-statement",
        "1",
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    run_logged(
        command,
        cwd=project,
        log=root / "runtime/logs" / f"equivalence_{suffix}_{label}.log",
    )


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def compare(project: Path, root: Path, label: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        project / BASE, local_files_only=True, trust_remote_code=False
    )
    stage = root / "evaluation/checkpoint_equivalence" / label
    left_generations = {
        (row["statement_id"], int(row["sample_index"])): row
        for row in read_jsonl(stage / "unmerged/generations.jsonl")
    }
    right_generations = {
        (row["statement_id"], int(row["sample_index"])): row
        for row in read_jsonl(stage / "merged/generations.jsonl")
    }
    left_attempts = {
        (row["problem_id"], int(row["attempt_index"])): row
        for row in read_jsonl(stage / "unmerged/attempts.jsonl")
    }
    right_attempts = {
        (row["problem_id"], int(row["attempt_index"])): row
        for row in read_jsonl(stage / "merged/attempts.jsonl")
    }
    keys = sorted(set(left_generations) & set(right_generations))
    if len(keys) != 24:
        raise RuntimeError(f"{label} equivalence alignment is {len(keys)} != 24")
    details = []
    for key in keys:
        left = left_generations[key]
        right = right_generations[key]
        left_tokens = tokenizer.encode(
            str(left["raw_output"]), add_special_tokens=False
        )
        right_tokens = tokenizer.encode(
            str(right["raw_output"]), add_special_tokens=False
        )
        details.append(
            {
                "statement_id": key[0],
                "exact_match": left_tokens == right_tokens,
                "first_token_match": left_tokens[:1] == right_tokens[:1],
                "common_prefix_tokens": common_prefix(left_tokens, right_tokens),
                "unmerged_tokens": len(left_tokens),
                "merged_tokens": len(right_tokens),
                "proof_extraction_match": (
                    left.get("extracted_proof") == right.get("extracted_proof")
                ),
                "pantograph_outcome_match": (
                    bool(left_attempts[key].get("success"))
                    == bool(right_attempts[key].get("success"))
                ),
                "pantograph_status_match": (
                    left_attempts[key].get("status")
                    == right_attempts[key].get("status")
                ),
                "unmerged_status": left_attempts[key].get("status"),
                "merged_status": right_attempts[key].get("status"),
            }
        )
    mismatches = sum(not row["pantograph_outcome_match"] for row in details)
    report = {
        "status": (
            "CHECKPOINT_EQUIVALENCE_PASSED"
            if mismatches == 0
            else "CHECKPOINT_EQUIVALENCE_FAILED"
        ),
        "passed": mismatches == 0,
        "prompts": 24,
        "exact_matches": sum(row["exact_match"] for row in details),
        "first_token_matches": sum(row["first_token_match"] for row in details),
        "mean_common_prefix_tokens": sum(
            row["common_prefix_tokens"] for row in details
        )
        / 24,
        "proof_extraction_matches": sum(
            row["proof_extraction_match"] for row in details
        ),
        "pantograph_outcome_matches": 24 - mismatches,
        "pantograph_outcome_mismatches": mismatches,
        "pantograph_status_matches": sum(
            row["pantograph_status_match"] for row in details
        ),
        "pantograph_status_mismatches": sum(
            not row["pantograph_status_match"] for row in details
        ),
        "details": details,
    }
    write_json(stage / "equivalence_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    manifest, config = prepare(project, root)
    reports: dict[str, Any] = {}
    for label in NEW_MODELS:
        base, adapter = checkpoint_paths(project, root, label)
        merged = root / "checkpoints" / label / "merged"
        if not merged.exists():
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "scripts.merge_lora_adapter",
                    "--base-model",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(merged),
                ],
                cwd=project,
                log=root / "runtime/logs" / f"merge_{label}.log",
            )
        stage = root / "evaluation/checkpoint_equivalence" / label
        run_one(
            project=project,
            root=root,
            label=label,
            model=base,
            adapter=adapter,
            manifest=manifest,
            config=config,
            output=stage / "unmerged",
            suffix="unmerged",
        )
        run_one(
            project=project,
            root=root,
            label=label,
            model=merged,
            adapter=None,
            manifest=manifest,
            config=config,
            output=stage / "merged",
            suffix="merged",
        )
        reports[label] = compare(project, root, label)
    write_json(
        root / "evaluation/checkpoint_equivalence/equivalence_summary.json",
        reports,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if not all(row["passed"] for row in reports.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
