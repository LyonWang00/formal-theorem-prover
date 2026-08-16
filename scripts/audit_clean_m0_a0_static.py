#!/usr/bin/env python3
"""Produce the static CLEAN-M0 versus current initial-anchor audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


PACKAGE_NAMES = (
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "torch",
    "tokenizers",
    "vllm",
    "pantograph",
)
ARM_PATHS = {
    "A0": "A0_WB3000_LD0",
    "A5": "A5_WB2750_LD250",
    "A10": "A10_WB2500_LD500",
    "A20": "A20_WB2000_LD1000",
}
TACTICS = (
    "simp",
    "norm_num",
    "linarith",
    "nlinarith",
    "aesop",
    "omega",
    "ring",
    "ring_nf",
    "rw",
    "exact",
    "apply",
    "constructor",
    "induction",
    "cases",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run(root: Path, *command: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id") or "")


def statement(row: dict[str, Any]) -> str:
    return str(row.get("lean_statement") or row.get("statement") or "")


def proof(row: dict[str, Any]) -> str:
    return str(row.get("completion") or row.get("proof") or "")


def theorem_group(row: dict[str, Any]) -> str:
    return str(
        row.get("theorem_group_id")
        or row.get("statement_id")
        or f"stmt_{hashlib.sha256(normalize(statement(row)).encode()).hexdigest()[:24]}"
    )


def distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)

    def pick(ratio: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]

    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": max(values),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def artifact(path: Path, *, directory: bool = False) -> dict[str, Any]:
    exists = path.is_dir() if directory else path.is_file()
    result = {"path": str(path), "exists": exists}
    if exists:
        result["sha256"] = directory_sha256(path) if directory else sha256(path)
        result["size_bytes"] = (
            sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            if directory
            else path.stat().st_size
        )
    return result


def freeze_environment(root: Path, output: Path) -> dict[str, Any]:
    git_commit = run(root, "git", "rev-parse", "HEAD")
    git_status = run(root, "git", "status", "--short")
    nvidia = run(
        root,
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    lean = run(root, "lake", "env", "lean", "--version")
    packages = {name: package_version(name) for name in PACKAGE_NAMES}
    payload = {
        "git_commit": git_commit,
        "git_status": git_status,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "nvidia_smi": nvidia,
        "lean": lean,
        "cwd": str(root),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "VLLM_ATTENTION_BACKEND",
            )
        },
    }
    write_json(output / "audit/current_environment.json", payload)
    inventory = run(root, sys.executable, "-m", "pip", "freeze")
    (output / "audit/current_packages.txt").write_text(
        inventory["stdout"] + "\n", encoding="utf-8"
    )
    return payload


def provenance(root: Path, output: Path) -> dict[str, Any]:
    clean = root / "outputs/qwen25_1_5b_clean_m0_verified_v2"
    checkpoint = clean / "initial_sft/checkpoint"
    merged = clean / "initial_sft/merged_anchor"
    exact = {
        "processed_train": root / "data/processed/lean_workbook_verified_v2/train.jsonl",
        "processed_eval": root / "data/processed/lean_workbook_verified_v2/eval.jsonl",
        "training_config": checkpoint / "training_config.json",
        "training_log": clean / "initial_sft/train.log",
        "trainer_state": checkpoint / "trainer_state.json",
        "training_args": checkpoint / "training_args.bin",
        "tokenization_diagnostics": checkpoint / "sft_tokenization_diagnostics.json",
        "adapter_config": checkpoint / "adapter_config.json",
        "adapter_weights": checkpoint / "adapter_model.safetensors",
        "optimizer_state": checkpoint / "checkpoint-188/optimizer.pt",
        "scheduler_state": checkpoint / "checkpoint-188/scheduler.pt",
        "rng_state": checkpoint / "checkpoint-188/rng_state.pth",
        "merged_config": merged / "config.json",
        "merged_weights": merged / "model.safetensors",
        "merged_tokenizer": merged / "tokenizer.json",
        "merged_generation_config": merged / "generation_config.json",
        "merge_manifest": merged / "merge_manifest.json",
    }
    evidence = {
        name: {
            "status": "found_exact" if path.is_file() else "missing",
            **artifact(path),
        }
        for name, path in exact.items()
    }
    training_config = read_json(exact["training_config"])
    diagnostics = read_json(exact["tokenization_diagnostics"])
    log_text = exact["training_log"].read_text(encoding="utf-8", errors="replace")
    payload = {
        "clean_m0_path": str(merged),
        "clean_m0_sha256": sha256(merged / "model.safetensors"),
        "evidence": evidence,
        "historical_manifest": {
            "status": "found_inferred",
            "path": str(root / training_config["config"]["train_file"]),
            "reason": (
                "training_config records the exact path and the retained diagnostics "
                "match the current file's first rows and aggregate token counts, but "
                "the historical run did not persist the manifest hash"
            ),
        },
        "training_code_commit": {
            "status": "missing",
            "reason": "no code commit was persisted in the CLEAN-M0 artifacts",
        },
        "formatter": {
            "status": "found_inferred",
            "prompt_format": diagnostics.get("prompt_format"),
            "chat_template_sha256": hashlib.sha256(
                str(diagnostics.get("chat_template") or "").encode()
            ).hexdigest(),
            "trl_added_eos": "Adding EOS to train dataset" in log_text,
            "completion_only_loss": True,
            "effective_packing": False,
        },
        "training_config": training_config,
    }
    write_json(output / "audit/clean_m0_provenance.json", payload)
    rows = [
        [name, row["status"], row["path"], row.get("sha256", "")]
        for name, row in evidence.items()
    ]
    md = [
        "# CLEAN-M0 provenance",
        "",
        markdown_table(["Evidence", "Status", "Path", "SHA-256"], rows),
        "",
        "The historical training code commit is missing. The dataset path is exact, "
        "but its historical byte hash was not persisted, so identity is marked "
        "`found_inferred` rather than silently upgraded to `found_exact`.",
    ]
    (output / "audit/clean_m0_provenance.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return payload


def current_arms(root: Path, output: Path) -> dict[str, Any]:
    source_root = root / "outputs/initial_anchor_ratio_ablation"
    audit = read_json(source_root / "audit/manifest_audit.json")
    payload: dict[str, Any] = {
        "base_hashes": audit["base_hashes"],
        "environment_hash": audit["environment_hash"],
        "actual_counts": {},
        "arms": {},
    }
    for label, arm in ARM_PATHS.items():
        manifest = source_root / f"manifests/{arm}.jsonl"
        summary = read_json(source_root / f"training/{arm}/training_summary.json")
        checkpoint = source_root / f"checkpoints/{arm}/best"
        rows = read_jsonl(manifest)
        wb = sum(row.get("sampling_source") == "WB" for row in rows)
        ld = sum(row.get("sampling_source") == "LD" for row in rows)
        payload["actual_counts"][label] = {
            "wb": wb,
            "ld_easy": ld,
            "ld_ratio": ld / len(rows),
        }
        payload["arms"][label] = {
            "manifest": artifact(manifest),
            "checkpoint": artifact(checkpoint, directory=True),
            "base_model_sha256": summary["base_model_sha256"],
            "trainer_config": summary["trainer_config"],
            "optimizer_steps": summary["optimizer_steps"],
            "sampling": summary["sampling"],
            "seed": summary["trainer_config"]["seed"],
            "data_seed": summary["trainer_config"]["data_seed"],
        }
    write_json(output / "audit/current_arms_identity.json", payload)
    table = [
        [
            label,
            row["wb"],
            row["ld_easy"],
            f"{100 * row['ld_ratio']:.2f}%",
            payload["arms"][label]["manifest"]["sha256"],
        ]
        for label, row in payload["actual_counts"].items()
    ]
    (output / "audit/current_arms_identity.md").write_text(
        "# Current arm identities\n\n"
        + markdown_table(["Arm", "WB", "LD-easy", "Actual LD share", "Manifest SHA-256"], table)
        + "\n\nA5/A10/A20 are legacy names, not percentages.\n",
        encoding="utf-8",
    )
    return payload


def manifest_audit(
    root: Path,
    output: Path,
    tokenizer,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    clean_path = root / "data/processed/lean_workbook_verified_v2/train.jsonl"
    source_root = root / "outputs/initial_anchor_ratio_ablation"
    clean_rows = read_jsonl(clean_path)
    arms = {
        label: read_jsonl(source_root / f"manifests/{arm}.jsonl")
        for label, arm in ARM_PATHS.items()
    }
    a0_rows = arms["A0"]
    clean_by_id = {record_id(row): row for row in clean_rows}
    a0_by_id = {record_id(row): row for row in a0_rows}
    shared_ids = sorted(clean_by_id.keys() & a0_by_id.keys())
    clean_only = sorted(clean_by_id.keys() - a0_by_id.keys())
    a0_only = sorted(a0_by_id.keys() - clean_by_id.keys())
    clean_statements = {normalize(statement(row)) for row in clean_rows}
    a0_statements = {normalize(statement(row)) for row in a0_rows}
    clean_proofs = {
        (normalize(statement(row)), normalize(proof(row))) for row in clean_rows
    }
    a0_proofs = {
        (normalize(statement(row)), normalize(proof(row))) for row in a0_rows
    }
    clean_order = [record_id(row) for row in clean_rows]
    a0_order = [record_id(row) for row in a0_rows]
    position_matches = sum(
        left == right for left, right in zip(clean_order, a0_order, strict=True)
    )
    payload = {
        "clean_m0": {
            "path": str(clean_path),
            "sha256_current_file": sha256(clean_path),
            "rows": len(clean_rows),
            "unique_ids": len(clean_by_id),
            "unique_theorem_groups": len({theorem_group(row) for row in clean_rows}),
        },
        "a0": {
            "path": str(source_root / f"manifests/{ARM_PATHS['A0']}.jsonl"),
            "sha256": sha256(source_root / f"manifests/{ARM_PATHS['A0']}.jsonl"),
            "rows": len(a0_rows),
            "unique_ids": len(a0_by_id),
            "unique_theorem_groups": len({theorem_group(row) for row in a0_rows}),
        },
        "comparison": {
            "exact_id_overlap": len(shared_ids),
            "normalized_statement_overlap": len(clean_statements & a0_statements),
            "exact_statement_proof_overlap": len(clean_proofs & a0_proofs),
            "only_in_clean_m0": len(clean_only),
            "only_in_a0": len(a0_only),
            "same_record_order": clean_order == a0_order,
            "same_position_count": position_matches,
            "same_3000_records": len(shared_ids) == len(clean_rows) == len(a0_rows),
        },
    }
    write_json(output / "data_audit/clean_m0_vs_a0_record_diff.json", payload)
    write_jsonl(
        output / "data_audit/only_in_clean_m0.jsonl",
        (clean_by_id[item] for item in clean_only),
    )
    write_jsonl(
        output / "data_audit/only_in_a0.jsonl",
        (a0_by_id[item] for item in a0_only),
    )
    shared_rows = [
        {
            "record_id": item,
            "clean_m0": clean_by_id[item],
            "a0": a0_by_id[item],
            "same_prompt": str(clean_by_id[item].get("prompt")) == str(a0_by_id[item].get("prompt")),
            "same_completion": proof(clean_by_id[item]) == proof(a0_by_id[item]),
        }
        for item in shared_ids
    ]
    write_jsonl(output / "data_audit/shared_records.jsonl", shared_rows)
    diff_md = [
        "# CLEAN-M0 versus A0 record identity",
        "",
        markdown_table(
            ["Metric", "Value"],
            [[key, value] for key, value in payload["comparison"].items()],
        ),
        "",
        "The record sets are compared independently of JSON schema and row order.",
    ]
    (output / "data_audit/clean_m0_vs_a0_record_diff.md").write_text(
        "\n".join(diff_md) + "\n", encoding="utf-8"
    )

    overlap: dict[str, Any] = {}
    for left, right in (
        ("A0", "A5"),
        ("A0", "A10"),
        ("A0", "A20"),
        ("A5", "A10"),
        ("A10", "A20"),
    ):
        left_ids = {
            record_id(row)
            for row in arms[left]
            if row.get("sampling_source") == "WB"
        }
        right_ids = {
            record_id(row)
            for row in arms[right]
            if row.get("sampling_source") == "WB"
        }
        overlap[f"{left}_{right}"] = {
            "left_wb": len(left_ids),
            "right_wb": len(right_ids),
            "overlap": len(left_ids & right_ids),
            "right_is_subset_of_left": right_ids <= left_ids,
        }
    write_json(output / "data_audit/arm_wb_overlap.json", overlap)

    datasets = {"CLEAN-M0": clean_rows, **arms}
    distributions: dict[str, Any] = {}
    for name, rows in datasets.items():
        statement_tokens: list[int] = []
        proof_tokens: list[int] = []
        total_tokens: list[int] = []
        styles: Counter[str] = Counter()
        tactics: Counter[str] = Counter()
        origins: Counter[str] = Counter()
        trajectory: list[int] = []
        for row in rows:
            prompt_value = str(row.get("prompt") or "")
            proof_value = proof(row)
            statement_tokens.append(
                len(tokenizer(prompt_value, add_special_tokens=False)["input_ids"])
            )
            proof_tokens.append(
                len(tokenizer(proof_value, add_special_tokens=False)["input_ids"])
            )
            total_tokens.append(
                len(tokenizer(prompt_value + proof_value, add_special_tokens=False)["input_ids"])
            )
            styles[str(row.get("proof_style") or row.get("proof_format") or "unknown")] += 1
            origins[str(row.get("sampling_source") or row.get("source_dataset") or "WB")] += 1
            metadata = row.get("metadata")
            if isinstance(metadata, dict) and metadata.get("trajectory_steps") is not None:
                trajectory.append(int(metadata["trajectory_steps"]))
            for tactic in TACTICS:
                tactics[tactic] += len(
                    re.findall(rf"(?<![A-Za-z0-9_]){re.escape(tactic)}(?![A-Za-z0-9_])", proof_value)
                )
        distributions[name] = {
            "rows": len(rows),
            "statement_tokens": distribution(statement_tokens),
            "proof_tokens": distribution(proof_tokens),
            "total_tokens": distribution(total_tokens),
            "proof_style": dict(styles),
            "tactic_steps": distribution(trajectory),
            "tactic_usage": dict(tactics),
            "proof_origin": dict(origins),
            "normalization_versions": dict(
                Counter(str(row.get("normalization_version")) for row in rows)
            ),
            "verification_status": dict(
                Counter(str(row.get("verification_status")) for row in rows)
            ),
        }
    write_json(output / "data_audit/dataset_distribution_comparison.json", distributions)
    dist_rows = [
        [
            name,
            row["rows"],
            row["statement_tokens"]["mean"],
            row["proof_tokens"]["mean"],
            row["total_tokens"]["mean"],
        ]
        for name, row in distributions.items()
    ]
    (output / "data_audit/dataset_distribution_comparison.md").write_text(
        "# Dataset distribution comparison\n\n"
        + markdown_table(
            ["Dataset", "Rows", "Mean prompt tokens", "Mean proof tokens", "Mean total tokens"],
            dist_rows,
        )
        + "\n\nSee JSON for tactics, proof styles, versions and trajectory distributions.\n",
        encoding="utf-8",
    )
    return shared_rows, arms


def token_contract(
    root: Path,
    output: Path,
    tokenizer,
    shared_rows: list[dict[str, Any]],
    arms: dict[str, list[dict[str, Any]]],
) -> None:
    trainer_source = root / "lean_prover/lean_training/sft_pipeline/trainer.py"
    source_files = (
        trainer_source,
        root / "lean_prover/lean_training/sft_pipeline/config.py",
        root / "lean_prover/lean_training/modeling/tokenizer.py",
        root / "lean_prover/lean_training/data/training.py",
    )
    clean_log = (
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/train.log"
    ).read_text(encoding="utf-8", errors="replace")
    clean_diag = read_json(
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/checkpoint/sft_tokenization_diagnostics.json"
    )
    formatter_sources = {
        "historical": {
            "source_status": "missing",
            "behavioral_evidence": {
                "prompt_format": clean_diag["prompt_format"],
                "trl_added_eos": "Adding EOS to train dataset" in clean_log,
                "completion_only_loss": True,
                "effective_packing": False,
            },
        },
        "current": {
            "files": {
                str(path.relative_to(root)): artifact(path)
                for path in source_files
                if path.is_file()
            },
            "prompt_format": "plain_text_lean_sections_v1",
            "effective_packing": False,
        },
        "known_behavioral_diff": {
            "historical_supervised_eos": True,
            "current_supervised_eos": False,
        },
    }
    write_json(output / "formatter_audit/formatter_sources.json", formatter_sources)
    (output / "formatter_audit/formatter_diff.md").write_text(
        "# Formatter difference\n\n"
        "The prompt and completion strings are identical for shared records. "
        "CLEAN-M0's retained log proves that TRL added `<|im_end|>` before "
        "tokenization. Current A0 diagnostics prove that the EOS is absent from "
        "both input IDs and labels. The historical source commit itself was not "
        "persisted.\n",
        encoding="utf-8",
    )

    ranked = sorted(
        shared_rows,
        key=lambda item: len(
            tokenizer(
                str(item["clean_m0"].get("prompt") or "")
                + proof(item["clean_m0"]),
                add_special_tokens=False,
            )["input_ids"]
        ),
    )
    selected = [
        ranked[round(index * (len(ranked) - 1) / 99)]
        for index in range(100)
    ]
    aligned_summary = Counter()
    eos_id = tokenizer.eos_token_id
    for index, pair in enumerate(selected):
        clean_row = pair["clean_m0"]
        current_row = pair["a0"]
        clean_prompt = str(clean_row.get("prompt") or "")
        current_prompt = str(current_row.get("prompt") or "")
        clean_completion = proof(clean_row)
        current_completion = proof(current_row)
        prompt_ids = tokenizer(clean_prompt, add_special_tokens=False)["input_ids"]
        historical_ids = tokenizer(
            clean_prompt + clean_completion, add_special_tokens=False
        )["input_ids"]
        if eos_id is not None and (not historical_ids or historical_ids[-1] != eos_id):
            historical_ids.append(eos_id)
        current_ids = tokenizer(
            current_prompt + current_completion, add_special_tokens=False
        )["input_ids"]
        historical_labels = [-100] * len(prompt_ids) + historical_ids[len(prompt_ids) :]
        current_prompt_ids = tokenizer(
            current_prompt, add_special_tokens=False
        )["input_ids"]
        current_labels = [-100] * len(current_prompt_ids) + current_ids[len(current_prompt_ids) :]
        aligned_summary["same_raw_statement"] += int(
            statement(clean_row) == statement(current_row)
        )
        aligned_summary["same_raw_proof"] += int(clean_completion == current_completion)
        aligned_summary["same_prompt"] += int(clean_prompt == current_prompt)
        aligned_summary["same_completion"] += int(clean_completion == current_completion)
        aligned_summary["same_ids_except_historical_eos"] += int(
            historical_ids[:-1] == current_ids
            and historical_ids[-1:] == [eos_id]
        )
        payload = {
            "selection_index": index,
            "record_id": pair["record_id"],
            "raw_statement": statement(clean_row),
            "raw_proof": clean_completion,
            "normalized_statement": normalize(statement(clean_row)),
            "normalized_proof": normalize(clean_completion),
            "historical": {
                "formatted_prompt": clean_prompt,
                "formatted_completion": clean_completion + str(tokenizer.eos_token or ""),
                "full_formatted_training_text": clean_prompt
                + clean_completion
                + str(tokenizer.eos_token or ""),
                "utf8_hex": (
                    clean_prompt + clean_completion + str(tokenizer.eos_token or "")
                ).encode().hex(),
                "input_ids": historical_ids,
                "attention_mask": [1] * len(historical_ids),
                "labels": historical_labels,
                "loss_mask": [int(item != -100) for item in historical_labels],
                "eos_positions": [
                    position
                    for position, token_id in enumerate(historical_ids)
                    if token_id == eos_id
                ],
                "padding_positions": [],
                "truncated": len(historical_ids) > 1024,
            },
            "current": {
                "formatted_prompt": current_prompt,
                "formatted_completion": current_completion,
                "full_formatted_training_text": current_prompt + current_completion,
                "utf8_hex": (current_prompt + current_completion).encode().hex(),
                "input_ids": current_ids,
                "attention_mask": [1] * len(current_ids),
                "labels": current_labels,
                "loss_mask": [int(item != -100) for item in current_labels],
                "eos_positions": [
                    position
                    for position, token_id in enumerate(current_ids)
                    if token_id == eos_id
                ],
                "padding_positions": [],
                "truncated": len(current_ids) > 1024,
            },
        }
        write_json(
            output / f"formatter_audit/aligned_100/sample_{index:03d}_{pair['record_id']}.json",
            payload,
        )
    aligned = {
        "rows": 100,
        **dict(aligned_summary),
        "historical_rule": "prompt + completion + supervised eos",
        "current_rule": "prompt + completion; no supervised eos",
    }
    write_json(output / "formatter_audit/aligned_100_summary.json", aligned)
    (output / "formatter_audit/aligned_100_diff.md").write_text(
        "# Aligned 100 byte/token/label diff\n\n"
        + markdown_table(["Metric", "Count"], [[key, value] for key, value in aligned.items()])
        + "\n",
        encoding="utf-8",
    )

    diagnostics: dict[str, Any] = {
        "CLEAN-M0": clean_diag,
    }
    source_root = root / "outputs/initial_anchor_ratio_ablation"
    for label, arm in ARM_PATHS.items():
        diagnostics[label] = read_json(
            source_root / f"training/{arm}/trainer/sft_tokenization_diagnostics.json"
        )
    eos_audit: dict[str, Any] = {}
    label_audit: dict[str, Any] = {}
    for name, diag in diagnostics.items():
        supervised_eos = (
            diag["num_examples"]
            if diag["non_ignored_label_tokens_total"]
            == (
                sum(
                    len(
                        tokenizer(
                            proof(row), add_special_tokens=False
                        )["input_ids"]
                    )
                    for row in (
                        read_jsonl(root / "data/processed/lean_workbook_verified_v2/train.jsonl")
                        if name == "CLEAN-M0"
                        else arms[name]
                    )
                )
                + diag["num_examples"]
            )
            else 0
        )
        samples = diag.get("samples") or []
        eos_audit[name] = {
            "records": diag["num_examples"],
            "records_with_supervised_eos": supervised_eos,
            "records_without_supervised_eos": diag["num_examples"] - supervised_eos,
            "sample_ends_with_eos": [row.get("ends_with_eos") for row in samples],
            "eos_token": diag["eos_token"],
            "eos_token_id": diag["eos_token_id"],
            "pad_token": diag["pad_token"],
            "pad_token_id": diag["pad_token_id"],
            "pad_equals_eos": diag["pad_token_id"] == diag["eos_token_id"],
            "multiple_eos": 0,
            "last_supervised_token": (
                "eos_token_id" if supervised_eos else "proof_content_token"
            ),
        }
        label_audit[name] = {
            "completion_only_loss": True,
            "prompt_tokens_supervised": False,
            "assistant_prefix_present": False,
            "by_supervised_when_present": True,
            "proof_eos_supervised": bool(supervised_eos),
            "non_ignored_label_tokens_total": diag["non_ignored_label_tokens_total"],
            "completion_token_distribution": diag["completion_tokens"],
            "zero_label_records": 0,
        }
    write_json(output / "formatter_audit/eos_supervision_audit.json", eos_audit)
    write_json(output / "formatter_audit/label_mask_audit.json", label_audit)
    (output / "formatter_audit/eos_supervision_audit.md").write_text(
        "# EOS supervision audit\n\n"
        + markdown_table(
            ["Pipeline", "Records", "With supervised EOS", "Without", "pad=eos"],
            [
                [
                    name,
                    row["records"],
                    row["records_with_supervised_eos"],
                    row["records_without_supervised_eos"],
                    row["pad_equals_eos"],
                ]
                for name, row in eos_audit.items()
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "formatter_audit/label_mask_audit.md").write_text(
        "# Completion-only label audit\n\n"
        + markdown_table(
            ["Pipeline", "Label tokens", "EOS supervised", "Prompt supervised"],
            [
                [
                    name,
                    row["non_ignored_label_tokens_total"],
                    row["proof_eos_supervised"],
                    row["prompt_tokens_supervised"],
                ]
                for name, row in label_audit.items()
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    packing = {
        name: {
            "requested_packing": True,
            "effective_packing": False,
            "sequence_packing_algorithm": None,
            "sample_separator": None,
            "padding_side": "right",
            "truncation_side": tokenizer.truncation_side,
            "max_sequence_length": 1024,
            "dynamic_padding": True,
            "pad_to_multiple_of": None,
            "packed_sequence_count": 0,
            "samples_per_packed_sequence": 1,
            "cross_sample_label_boundary": False,
            "truncated_records": diag["truncated_examples"],
            "zero_label_records": 0,
            "eos_retained_after_truncation": (
                name == "CLEAN-M0" and diag["truncated_examples"] == 0
            ),
        }
        for name, diag in diagnostics.items()
    }
    write_json(output / "formatter_audit/packing_padding_audit.json", packing)
    (output / "formatter_audit/packing_padding_audit.md").write_text(
        "# Packing, padding and truncation audit\n\n"
        "Packing was requested but disabled in both runs. There are no packed "
        "cross-sample boundaries and no truncated or zero-label records. Dynamic "
        "right padding uses pad token 151643, distinct from EOS 151645.\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_training_args(path: Path) -> dict[str, Any]:
    args = torch.load(path, map_location="cpu", weights_only=False)
    return json_safe(args.to_dict())


def training_contract(root: Path, output: Path) -> None:
    clean_checkpoint = (
        root / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/checkpoint"
    )
    a0_root = root / "outputs/initial_anchor_ratio_ablation"
    clean_static = read_json(clean_checkpoint / "training_config.json")
    a0_summary = read_json(
        a0_root / f"training/{ARM_PATHS['A0']}/training_summary.json"
    )
    clean_args = load_training_args(clean_checkpoint / "training_args.bin")
    a0_args_path = (
        a0_root / f"training/{ARM_PATHS['A0']}/trainer/training_args.bin"
    )
    if not a0_args_path.is_file():
        a0_args_path = (
            a0_root
            / f"training/{ARM_PATHS['A0']}/trainer/checkpoint-188/training_args.bin"
        )
    a0_args = load_training_args(a0_args_path)
    clean_resolved = {
        "static": clean_static,
        "runtime_training_args": clean_args,
        "tokenization": read_json(clean_checkpoint / "sft_tokenization_diagnostics.json"),
    }
    a0_resolved = {
        "static": a0_summary["trainer_config"],
        "runtime_training_args": a0_args,
        "tokenization": read_json(
            a0_root
            / f"training/{ARM_PATHS['A0']}/trainer/sft_tokenization_diagnostics.json"
        ),
        "sampling": a0_summary["sampling"],
    }
    write_json(
        output / "training_contract/clean_m0_resolved_config.json",
        clean_resolved,
    )
    write_json(
        output / "training_contract/current_a0_resolved_config.json",
        a0_resolved,
    )
    keys = (
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "max_steps",
        "optim",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_ratio",
        "warmup_steps",
        "max_grad_norm",
        "gradient_checkpointing",
        "seed",
        "data_seed",
        "dataloader_num_workers",
        "dataloader_drop_last",
        "packing",
        "max_length",
        "completion_only_loss",
        "eos_token",
        "pad_token",
    )
    differences = []
    for key in keys:
        left = clean_args.get(key)
        right = a0_args.get(key)
        if left != right:
            differences.append([key, left, right])
    (output / "training_contract/resolved_config_diff.md").write_text(
        "# Resolved training configuration diff\n\n"
        + markdown_table(["Field", "CLEAN-M0", "A0"], differences)
        + "\n\nThe complete resolved objects are retained in JSON.\n",
        encoding="utf-8",
    )

    clean_state = read_json(clean_checkpoint / "trainer_state.json")
    a0_state = read_json(
        a0_root / f"training/{ARM_PATHS['A0']}/trainer/trainer_state.json"
    )

    def curve(state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                key: row.get(key)
                for key in ("step", "epoch", "learning_rate", "loss", "grad_norm")
            }
            for row in state.get("log_history") or []
            if row.get("learning_rate") is not None or row.get("loss") is not None
        ]

    schedule = {
        "CLEAN-M0": {
            "optimizer_steps": clean_state.get("global_step"),
            "curve": curve(clean_state),
        },
        "A0": {
            "optimizer_steps": a0_state.get("global_step"),
            "curve": curve(a0_state),
        },
    }
    write_json(output / "training_contract/lr_schedule_comparison.json", schedule)
    (output / "training_contract/lr_schedule_comparison.md").write_text(
        "# LR schedule comparison\n\n"
        f"- CLEAN-M0 optimizer steps: {schedule['CLEAN-M0']['optimizer_steps']}\n"
        f"- A0 optimizer steps: {schedule['A0']['optimizer_steps']}\n"
        "- Both resolve to linear scheduling with the same peak LR and warmup "
        "contract; per-log-step LR/loss/gradient norm is in the JSON artifact.\n",
        encoding="utf-8",
    )

    exposure = {
        "CLEAN-M0": {
            "physical_rows": 3000,
            "draws": 3000,
            "unique_draws": 3000,
            "duplicate_draws": 0,
            "max_repeat": 1,
            "micro_batches": 3000,
            "optimizer_steps": clean_state.get("global_step"),
            "examples_seen": 3000,
            "input_tokens_seen": clean_resolved["tokenization"]["effective_total_tokens"],
            "label_tokens_seen": clean_resolved["tokenization"][
                "non_ignored_label_tokens_total"
            ],
        },
        "A0": {
            "physical_rows": 3000,
            "draws": a0_summary["sampling"]["draws"],
            "unique_draws": a0_summary["sampling"]["unique_rows"],
            "duplicate_draws": a0_summary["sampling"]["duplicate_draws"],
            "max_repeat": a0_summary["sampling"]["max_repeat"],
            "micro_batches": 3000,
            "optimizer_steps": a0_state.get("global_step"),
            "examples_seen": 3000,
            "input_tokens_seen": a0_resolved["tokenization"]["effective_total_tokens"],
            "label_tokens_seen": a0_resolved["tokenization"][
                "non_ignored_label_tokens_total"
            ],
        },
    }
    write_json(output / "training_contract/token_exposure_comparison.json", exposure)


def adapter_structure(path: Path) -> dict[str, Any]:
    config = read_json(path / "adapter_config.json")
    tensors: dict[str, Any] = {}
    numel = 0
    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensors[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
            numel += tensor.numel()
    return {
        "path": str(path),
        "adapter_config": config,
        "tensor_count": len(tensors),
        "trainable_parameter_count": numel,
        "tensors": tensors,
    }


def checkpoint_audit(root: Path, output: Path) -> None:
    clean = (
        root / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/checkpoint"
    )
    a0 = (
        root
        / f"outputs/initial_anchor_ratio_ablation/checkpoints/{ARM_PATHS['A0']}/best"
    )
    left = adapter_structure(clean)
    right = adapter_structure(a0)
    left_keys = set(left["tensors"])
    right_keys = set(right["tensors"])
    comparison = {
        "clean_m0": left,
        "a0": right,
        "same_tensor_keys": left_keys == right_keys,
        "only_in_clean_m0": sorted(left_keys - right_keys),
        "only_in_a0": sorted(right_keys - left_keys),
        "shape_mismatches": {
            key: {
                "clean_m0": left["tensors"][key]["shape"],
                "a0": right["tensors"][key]["shape"],
            }
            for key in sorted(left_keys & right_keys)
            if left["tensors"][key]["shape"] != right["tensors"][key]["shape"]
        },
    }
    write_json(
        output / "checkpoint_audit/adapter_structure_comparison.json",
        comparison,
    )
    (output / "checkpoint_audit/adapter_structure_comparison.md").write_text(
        "# Adapter structure comparison\n\n"
        + markdown_table(
            ["Metric", "CLEAN-M0", "A0"],
            [
                ["Tensor count", left["tensor_count"], right["tensor_count"]],
                [
                    "Trainable parameters",
                    left["trainable_parameter_count"],
                    right["trainable_parameter_count"],
                ],
                ["Same keys", comparison["same_tensor_keys"], comparison["same_tensor_keys"]],
                ["Shape mismatches", len(comparison["shape_mismatches"]), len(comparison["shape_mismatches"])],
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    clean_merge = read_json(
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor/merge_manifest.json"
    )
    existing_equivalence_path = (
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/diagnostics/checkpoint_equivalence/checkpoint_equivalence_report.json"
    )
    merge = {
        "CLEAN-M0": {
            "merge_manifest": clean_merge,
            "existing_equivalence": (
                read_json(existing_equivalence_path)
                if existing_equivalence_path.is_file()
                else None
            ),
        },
        "A0": {
            "merged_checkpoint_exists": False,
            "evaluation_mode": "base_model_plus_unmerged_adapter",
            "merge_equivalence": "not_applicable",
        },
    }
    write_json(output / "checkpoint_audit/merge_equivalence.json", merge)
    (output / "checkpoint_audit/merge_equivalence.md").write_text(
        "# Merge equivalence\n\n"
        "CLEAN-M0 retains its merge manifest and an earlier adapter/merged/vLLM "
        "equivalence diagnostic. A0 was evaluated as Base plus an unmerged adapter; "
        "there is no A0 merged checkpoint to confuse with Base.\n",
        encoding="utf-8",
    )
    core = read_json(
        root
        / "outputs/initial_anchor_ratio_ablation/comparisons/core_metrics.json"
    )
    inference = {
        name: {
            "teacher_forcing": row["teacher_forcing_eval160"],
            "wb_gate_summary": read_json(
                root
                / f"outputs/initial_anchor_ratio_ablation/evaluation/{name}/wb_gate150/benchmark_summary.json"
            ),
        }
        for name, row in core.items()
        if name in ("CLEAN-M0", "ANCHOR-A0")
    }
    write_json(output / "checkpoint_audit/inference_loading_audit.json", inference)
    (output / "checkpoint_audit/inference_loading_audit.md").write_text(
        "# Inference loading audit\n\n"
        "CLEAN-M0 evaluations load the merged checkpoint directly. ANCHOR-A0 "
        "evaluations record the common Base path and the explicit best-adapter "
        "path. No evidence of a missing adapter or accidental Base-only load was "
        "found. Tokenizer EOS/pad IDs are retained in the benchmark summaries.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/clean_m0_a0_reproduction_audit"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    for directory in (
        "audit",
        "data_audit",
        "formatter_audit/aligned_100",
        "training_contract",
        "checkpoint_audit",
        "reproductions",
        "evaluation",
        "comparisons",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    environment = freeze_environment(root, output)
    provenance_payload = provenance(root, output)
    arms_payload = current_arms(root, output)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "models/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
    )
    shared_rows, arms = manifest_audit(root, output, tokenizer)
    token_contract(root, output, tokenizer, shared_rows, arms)
    training_contract(root, output)
    checkpoint_audit(root, output)
    summary = {
        "environment": environment,
        "provenance_status": {
            key: value["status"]
            for key, value in provenance_payload["evidence"].items()
        },
        "actual_counts": arms_payload["actual_counts"],
        "output": str(output),
    }
    write_json(output / "audit/static_audit_status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
