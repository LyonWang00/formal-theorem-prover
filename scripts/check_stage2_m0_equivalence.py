#!/usr/bin/env python3
"""Build and verify the read-only merged M0 required by Stage-2 Phase 0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.run_r_random_replication_evaluation import (
    read_json,
    read_jsonl,
    run_logged,
    sha256,
    write_json,
    write_jsonl,
)


FROZEN_NAME = "M0-ADDON-B-FROZEN"
SOURCE = Path(
    "outputs/wb_ld_budget_support_replay_ablation/datasets/"
    "wb_unseen_holdout_150.jsonl"
)
BASE = Path("models/Qwen2.5-1.5B-Instruct")
ADAPTER = Path(
    "outputs/wb_ld_budget_support_replay_ablation/checkpoints/"
    "ADDON-B-WB2000-LD1000/best"
)
ROOT = Path("outputs/stage2_sft_incremental_ablation")


def run_evaluation(
    *,
    project: Path,
    config: Path,
    manifest: Path,
    model: Path,
    adapter: Path | None,
    output: Path,
    log: Path,
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
    run_logged(command, cwd=project, log=log)


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def compare(project: Path, stage: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        project / BASE, local_files_only=True, trust_remote_code=False
    )
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
        raise RuntimeError(f"M0 equivalence alignment is {len(keys)} != 24")
    details = []
    for key in keys:
        left = left_generations[key]
        right = right_generations[key]
        left_tokens = tokenizer.encode(str(left["raw_output"]), add_special_tokens=False)
        right_tokens = tokenizer.encode(str(right["raw_output"]), add_special_tokens=False)
        prefix = common_prefix(left_tokens, right_tokens)
        details.append(
            {
                "statement_id": key[0],
                "exact_generation_match": left_tokens == right_tokens,
                "first_token_match": left_tokens[:1] == right_tokens[:1],
                "common_prefix_tokens": prefix,
                "major_common_prefix_match": prefix >= min(
                    8, len(left_tokens), len(right_tokens)
                ),
                "unmerged_tokens": len(left_tokens),
                "merged_tokens": len(right_tokens),
                "proof_extraction_outcome_match": bool(left.get("extracted_proof"))
                == bool(right.get("extracted_proof")),
                "exact_extracted_proof_match": left.get("extracted_proof")
                == right.get("extracted_proof"),
                "pantograph_outcome_match": bool(left_attempts[key].get("success"))
                == bool(right_attempts[key].get("success")),
                "pantograph_status_match": left_attempts[key].get("status")
                == right_attempts[key].get("status"),
                "unmerged_status": left_attempts[key].get("status"),
                "merged_status": right_attempts[key].get("status"),
            }
        )
    first_token_matches = sum(row["first_token_match"] for row in details)
    prefix_matches = sum(row["major_common_prefix_match"] for row in details)
    extraction_outcome_matches = sum(
        row["proof_extraction_outcome_match"] for row in details
    )
    pantograph_matches = sum(row["pantograph_outcome_match"] for row in details)
    passed = (
        first_token_matches == 24
        and prefix_matches == 24
        and extraction_outcome_matches == 24
        and pantograph_matches == 24
    )
    return {
        "status": (
            "CHECKPOINT_EQUIVALENCE_PASSED"
            if passed
            else "CHECKPOINT_EQUIVALENCE_FAILED"
        ),
        "passed": passed,
        "frozen_model": FROZEN_NAME,
        "prompts": 24,
        "first_token_matches": first_token_matches,
        "major_common_prefix_matches": prefix_matches,
        "mean_common_prefix_tokens": sum(
            row["common_prefix_tokens"] for row in details
        )
        / 24,
        "proof_extraction_outcome_matches": extraction_outcome_matches,
        "exact_extracted_proof_matches": sum(
            row["exact_extracted_proof_match"] for row in details
        ),
        "pantograph_outcome_matches": pantograph_matches,
        "pantograph_status_matches": sum(
            row["pantograph_status_match"] for row in details
        ),
        "gate_contract": {
            "first_token": "24/24",
            "major_common_prefix": "at least min(8, output lengths) tokens on 24/24",
            "proof_extraction_outcome": "24/24 boolean outcome match",
            "pantograph_verification_outcome": "24/24 success boolean match",
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    source = project / SOURCE
    base = project / BASE
    adapter = project / ADAPTER
    merged = root / "shared/checkpoints/M0-ADDON-B-FROZEN-MERGED"
    equivalence = root / "shared/m0_checkpoint_equivalence"
    logs = equivalence / "logs"
    stage = equivalence / "results"

    manifest = equivalence / "greedy24_manifest.jsonl"
    if not manifest.exists():
        write_jsonl(manifest, read_jsonl(source)[:24])
    frozen_config = project / (
        "outputs/r_random_replication_strict_eval/evaluation/"
        "checkpoint_equivalence/greedy_generation_config.json"
    )
    config_payload = read_json(frozen_config)
    config = equivalence / "greedy_generation_config.json"
    write_json(config, config_payload)
    write_json(
        equivalence / "canary_identity.json",
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
            "comparison": "bf16 dynamic LoRA versus bf16 safe-merged M0",
        },
    )

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
            log=logs / "merge_m0.log",
        )
    run_evaluation(
        project=project,
        config=config,
        manifest=manifest,
        model=base,
        adapter=adapter,
        output=stage / "unmerged",
        log=logs / "equivalence_unmerged_m0.log",
    )
    run_evaluation(
        project=project,
        config=config,
        manifest=manifest,
        model=merged,
        adapter=None,
        output=stage / "merged",
        log=logs / "equivalence_merged_m0.log",
    )
    report = compare(project, stage)
    write_json(equivalence / "equivalence_summary.json", report)
    write_json(
        merged / "READ_ONLY_IDENTITY.json",
        {
            "frozen_model": FROZEN_NAME,
            "source_base": str(base),
            "source_adapter": str(adapter),
            "equivalence_summary": str(equivalence / "equivalence_summary.json"),
            "equivalence_passed": report["passed"],
            "intended_access": "read_only",
        },
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
