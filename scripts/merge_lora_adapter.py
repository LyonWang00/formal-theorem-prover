#!/usr/bin/env python3
"""Merge one LoRA adapter into its base model with explicit provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = Path(args.base_model).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged model: {output}")
    required = [
        base / "config.json",
        base / "model.safetensors",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing merge inputs: {missing}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        str(adapter),
        trust_remote_code=True,
    ).save_pretrained(output)
    manifest = {
        "status": "merged_clean_m0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "dtype": str(dtype),
        "safe_merge": True,
        "input_sha256": {
            str(path): file_sha256(path) for path in required
        },
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
