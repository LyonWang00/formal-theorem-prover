#!/usr/bin/env python3
"""Freeze and preflight the four EOS-fixed initial-anchor training arms."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
)
from lean_prover.lean_training.sft_pipeline.trainer import (
    assert_supervised_eos_contract,
)


EXPECTED_BASE_HASH = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
EXPECTED_ENVIRONMENT_HASH = (
    "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
)
ARMS = {
    "A0": {
        "name": "ANCHOR-A0-EOS-FIXED",
        "source": "A0_WB3000_LD0.jsonl",
        "sha256": "9c4d2f5aee256c70937353a5f14e9d275d5d869d4feb2fb11c03771357f11686",
        "wb": 3000,
        "ld_easy": 0,
    },
    "A5": {
        "name": "ANCHOR-A5-EOS-FIXED",
        "source": "A5_WB2750_LD250.jsonl",
        "sha256": "640d344dd0ac0ac76524f880309bed13a1dd8c2d5c959b05790872e95e003ec2",
        "wb": 2750,
        "ld_easy": 250,
    },
    "A10": {
        "name": "ANCHOR-A10-EOS-FIXED",
        "source": "A10_WB2500_LD500.jsonl",
        "sha256": "8cca18628ff588e4a7fe8c47df807f942f16d4bbab4925e32b9132fbeedbb34d",
        "wb": 2500,
        "ld_easy": 500,
    },
    "A20": {
        "name": "ANCHOR-A20-EOS-FIXED",
        "source": "A20_WB2000_LD1000.jsonl",
        "sha256": "41ef30bedff5224c8ec29a2e15f55a2c8bbde8334f06c4d096e4c8e82736f981",
        "wb": 2000,
        "ld_easy": 1000,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_frozen(source: Path, destination: Path, expected_hash: str) -> None:
    actual = sha256(source)
    if actual != expected_hash:
        raise RuntimeError(
            f"frozen manifest hash mismatch: {source}: {actual} != {expected_hash}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"refusing to overwrite changed frozen copy: {destination}")
        return
    shutil.copyfile(source, destination)
    if sha256(destination) != expected_hash:
        raise RuntimeError(f"frozen manifest copy changed bytes: {destination}")


def training_contract(project: Path, tokenizer) -> dict[str, Any]:
    previous = json.loads(
        (
            project
            / "outputs/clean_m0_a0_reproduction_audit/training_contract/"
            "current_a0_resolved_config.json"
        ).read_text(encoding="utf-8-sig")
    )
    static = dict(previous["static"])
    static.update(
        {
            "train_file": "ARM_SPECIFIC_FROZEN_SOURCE_PLUS_EOS",
            "output_dir": "ARM_SPECIFIC_NEW_OUTPUT",
            "require_supervised_eos": True,
        }
    )
    return {
        "selection": (
            "old A0 resolved optimization/sampling contract with the sole "
            "training-input change of one supervised tokenizer EOS per row"
        ),
        "static_sft_config": static,
        "effective": {
            "base_model": str(project / "models/Qwen2.5-1.5B-Instruct"),
            "base_model_sha256": EXPECTED_BASE_HASH,
            "tokenizer_json_sha256": sha256(
                project / "models/Qwen2.5-1.5B-Instruct/tokenizer.json"
            ),
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template or "").encode("utf-8")
            ).hexdigest(),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "completion_eos_materialization": (
                "append tokenizer.eos_token exactly once at completion end"
            ),
            "completion_only_loss": True,
            "packing": False,
            "padding_side": "right",
            "truncation_side": tokenizer.truncation_side,
            "allow_overlength": False,
            "sampling": "fixed_manifest_without_replacement",
            "shuffle": "single torch.randperm seeded by data_seed=42",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "normalization_version": NORMALIZATION_VERSION,
            "assembler_version": ASSEMBLER_VERSION,
            "environment_hash": EXPECTED_ENVIRONMENT_HASH,
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "trl", "peft", "bitsandbytes")
            },
        },
    }


def main() -> None:
    project = Path(".").resolve()
    source_root = project / "outputs/initial_anchor_ratio_ablation/manifests"
    output = project / "outputs/anchor_ratio_eos_fixed"
    audit = output / "audit"
    manifests = output / "manifests"
    model = project / "models/Qwen2.5-1.5B-Instruct"
    if sha256(model / "model.safetensors") != EXPECTED_BASE_HASH:
        raise RuntimeError("Base model hash changed")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if (
        tokenizer.eos_token_id != 151645
        or tokenizer.pad_token_id != 151643
        or tokenizer.eos_token_id == tokenizer.pad_token_id
        or not tokenizer.eos_token
    ):
        raise RuntimeError("tokenizer EOS/PAD identity changed")

    contract = training_contract(project, tokenizer)
    write_json(audit / "frozen_training_contract.json", contract)
    contract_hash = sha256(audit / "frozen_training_contract.json")
    identities: dict[str, Any] = {
        "environment_hash": EXPECTED_ENVIRONMENT_HASH,
        "base_model_sha256": EXPECTED_BASE_HASH,
        "training_contract_sha256": contract_hash,
        "arms": {},
    }
    gates: dict[str, Any] = {}
    exposures: dict[str, Any] = {}

    for arm, spec in ARMS.items():
        source = source_root / spec["source"]
        frozen = manifests / f"{arm}_source_manifest.jsonl"
        copy_frozen(source, frozen, spec["sha256"])
        rows = read_jsonl(source)
        if len(rows) != 3000:
            raise RuntimeError(f"{arm} row count changed")
        ids = [str(row.get("record_id") or row.get("id") or "") for row in rows]
        if not all(ids) or len(set(ids)) != 3000:
            raise RuntimeError(f"{arm} no longer contains 3000 unique IDs")
        wb = sum(str(row.get("sampling_source") or "").upper() == "WB" for row in rows)
        ld = sum(str(row.get("sampling_source") or "").upper() == "LD" for row in rows)
        if (wb, ld) != (spec["wb"], spec["ld_easy"]):
            raise RuntimeError(f"{arm} WB/LD counts changed: {(wb, ld)}")

        effective_rows: list[dict[str, Any]] = []
        old_label_total = 0
        new_label_total = 0
        aligned_samples: list[dict[str, Any]] = []
        for index, original in enumerate(rows):
            completion = str(original.get("completion") or "")
            if not completion.strip():
                raise RuntimeError(f"{arm} row {index} has empty completion")
            if tokenizer.eos_token in completion:
                raise RuntimeError(f"{arm} row {index} already contains EOS text")
            prompt = str(original.get("prompt") or "")
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            old_ids = tokenizer(
                prompt + completion, add_special_tokens=True
            )["input_ids"]
            new_completion = completion + tokenizer.eos_token
            new_ids = tokenizer(
                prompt + new_completion, add_special_tokens=True
            )["input_ids"]
            if old_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(f"{arm} row {index} prompt boundary changed")
            if new_ids != old_ids + [tokenizer.eos_token_id]:
                raise RuntimeError(f"{arm} row {index} differs by more than final EOS")
            old_labels = [-100] * len(prompt_ids) + old_ids[len(prompt_ids) :]
            new_labels = old_labels + [tokenizer.eos_token_id]
            old_label_total += sum(label != -100 for label in old_labels)
            new_label_total += sum(label != -100 for label in new_labels)
            row = dict(original)
            row["completion"] = new_completion
            effective_rows.append(row)
            if index < 100:
                aligned_samples.append(
                    {
                        "index": index,
                        "record_id": ids[index],
                        "raw_statement_equal": True,
                        "raw_proof_equal": True,
                        "formatted_prompt_equal": True,
                        "old_input_tokens": len(old_ids),
                        "new_input_tokens": len(new_ids),
                        "old_valid_labels": sum(label != -100 for label in old_labels),
                        "new_valid_labels": sum(label != -100 for label in new_labels),
                        "only_difference": "one final supervised eos_token_id",
                        "eos_position": len(new_ids) - 1,
                        "last_valid_label": new_labels[-1],
                    }
                )

        effective_path = output / "training" / spec["name"] / "input/effective_train.jsonl"
        write_jsonl(effective_path, effective_rows)
        gate = assert_supervised_eos_contract(
            Dataset.from_list(effective_rows),
            tokenizer,
            max_seq_length=1024,
        )
        gate.update(
            {
                "arm": arm,
                "model_name": spec["name"],
                "status": "EOS_GATE_PASSED",
                "source_manifest_sha256": spec["sha256"],
                "effective_manifest_path": str(effective_path),
                "effective_manifest_sha256": sha256(effective_path),
            }
        )
        gates[arm] = gate
        write_json(audit / f"eos_gate_{arm}.json", gate)
        exposure = {
            "old_supervised_label_tokens": old_label_total,
            "new_supervised_label_tokens": new_label_total,
            "delta": new_label_total - old_label_total,
            "expected_delta": 3000,
            "passed": new_label_total - old_label_total == 3000,
        }
        if not exposure["passed"]:
            raise RuntimeError(f"{arm} label exposure delta is not 3000")
        exposures[arm] = exposure
        aligned_md = [
            f"# {arm} aligned-100 EOS 差异审计",
            "",
            "100/100 条的 statement、proof、prompt、既有 token 和 mask 起点完全相同。",
            "新输入和 labels 仅在 completion 末尾增加一个受监督 eos_token_id=151645。",
            "",
            f"- old label tokens: {old_label_total}",
            f"- new label tokens: {new_label_total}",
            f"- delta: {new_label_total - old_label_total}",
            "",
            "逐条机器可审计摘要：",
            "",
            "```json",
            json.dumps(aligned_samples, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        (audit / f"aligned_100_{arm}.md").write_text(
            "\n".join(aligned_md), encoding="utf-8"
        )
        identities["arms"][arm] = {
            "name": spec["name"],
            "source_path": str(source),
            "source_sha256": sha256(source),
            "frozen_copy_path": str(frozen),
            "frozen_copy_sha256": sha256(frozen),
            "rows": len(rows),
            "ordered_ids_sha256": hash_values(ids),
            "wb_rows": wb,
            "ld_easy_rows": ld,
            "actual_ld_row_share": ld / len(rows),
            "theorem_groups": len(
                {
                    str(row.get("theorem_group_id") or row.get("lean_statement"))
                    for row in rows
                }
            ),
            "statement_hashes_sha256": hash_values(
                [
                    hashlib.sha256(
                        str(row.get("lean_statement") or row.get("statement") or "").encode("utf-8")
                    ).hexdigest()
                    for row in rows
                ]
            ),
            "proof_hashes_sha256": hash_values(
                [
                    hashlib.sha256(str(row.get("proof") or "").encode("utf-8")).hexdigest()
                    for row in rows
                ]
            ),
        }

    write_json(audit / "frozen_manifest_identity.json", identities)
    write_json(audit / "token_exposure_comparison.json", exposures)
    exposure_lines = ["# EOS token exposure comparison", ""]
    for arm, row in exposures.items():
        exposure_lines.append(
            f"- {arm}: {row['old_supervised_label_tokens']} → "
            f"{row['new_supervised_label_tokens']} (Δ={row['delta']}, passed={row['passed']})"
        )
    (audit / "token_exposure_comparison.md").write_text(
        "\n".join(exposure_lines) + "\n", encoding="utf-8"
    )
    gate_lines = ["# 四臂 EOS 启动门禁", ""]
    for arm, gate in gates.items():
        gate_lines.append(
            f"- {arm}: {gate['status']}; 3000/3000 supervised EOS; "
            "zero-label=0; empty=0; semantic truncation=0"
        )
    (audit / "eos_gate_summary.md").write_text(
        "\n".join(gate_lines) + "\n", encoding="utf-8"
    )
    experiment_contract = {
        "training_contract_sha256": contract_hash,
        "arm_names": [spec["name"] for spec in ARMS.values()],
        "evaluation": {
            "eval160": "data/processed/lean_workbook_verified_v2/eval.jsonl",
            "wb_gate150": "outputs/expert_sft_anchor_ablation/gates/anchor_gate_150.jsonl",
            "ld_easy64": "outputs/ld_length_difficulty_pipeline/pilot_sft/evaluation/ld_easy_holdout_64.jsonl",
            "monitor64": "outputs/b2_expanded_validation/datasets/monitor_minif2f_valid_64.jsonl",
            "generation": {
                "samples_per_statement": 4,
                "max_new_tokens": 256,
                "seed_wb": 20261001,
                "seed_ld": 20261002,
                "seed_monitor": 20261003,
            },
        },
        "forbidden": ["rank_ablation", "expert_iteration", "grpo", "benchmark"],
    }
    write_json(audit / "experiment_contract.json", experiment_contract)
    print(audit / "eos_gate_summary.md")


if __name__ == "__main__":
    main()
