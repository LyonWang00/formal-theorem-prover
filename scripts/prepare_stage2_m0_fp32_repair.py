#!/usr/bin/env python3
"""Build an FP32 CPU base view and safe-merged M0 for equivalence repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    args = parser.parse_args()
    project = args.project.resolve()
    base = project / "models/Qwen2.5-1.5B-Instruct"
    adapter = project / (
        "outputs/wb_ld_budget_support_replay_ablation/checkpoints/"
        "ADDON-B-WB2000-LD1000/best"
    )
    root = project / "outputs/stage2_sft_incremental_ablation/shared/checkpoints"
    base_view = root / "Qwen2.5-1.5B-Instruct-FP32-VIEW"
    merged = root / "M0-ADDON-B-FROZEN-MERGED-FP32"

    base_view.mkdir(parents=True, exist_ok=True)
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    base_config["dtype"] = "float32"
    base_config["torch_dtype"] = "float32"
    write_json(base_view / "config.json", base_config)
    for name in (
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        source = base / name
        target = base_view / name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(source, target)
    write_json(
        base_view / "FP32_VIEW_IDENTITY.json",
        {
            "source_base": str(base),
            "source_model_sha256": sha256(base / "model.safetensors"),
            "purpose": "force identical FP32 CPU loading for equivalence repair",
            "weights_are_symlinked_read_only": True,
        },
    )

    if (merged / "model.safetensors").is_file():
        print(json.dumps({"status": "reused", "merged": str(merged)}, indent=2))
        return
    if merged.exists():
        raise RuntimeError(f"refusing incomplete existing output: {merged}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter),
        is_trainable=False,
        local_files_only=True,
    )
    model = model.merge_and_unload(safe_merge=True)
    merged.mkdir(parents=True)
    model.save_pretrained(
        merged,
        safe_serialization=True,
        max_shard_size="8GB",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False
    )
    tokenizer.save_pretrained(merged)
    merged_config = json.loads((merged / "config.json").read_text(encoding="utf-8"))
    merged_config["dtype"] = "float32"
    merged_config["torch_dtype"] = "float32"
    write_json(merged / "config.json", merged_config)
    write_json(
        merged / "merge_manifest.json",
        {
            "status": "merged_m0_fp32_cpu",
            "base_model": str(base),
            "adapter": str(adapter),
            "output": str(merged),
            "dtype": "torch.float32",
            "device": "cpu",
            "safe_merge": True,
            "merge_method": "PEFT merge_and_unload(safe_merge=True)",
            "input_sha256": {
                "base_model": sha256(base / "model.safetensors"),
                "adapter_model": sha256(adapter / "adapter_model.safetensors"),
                "adapter_config": sha256(adapter / "adapter_config.json"),
            },
            "output_model_sha256": sha256(merged / "model.safetensors"),
        },
    )
    print(
        json.dumps(
            {
                "status": "created",
                "base_view": str(base_view),
                "merged": str(merged),
                "model_sha256": sha256(merged / "model.safetensors"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
