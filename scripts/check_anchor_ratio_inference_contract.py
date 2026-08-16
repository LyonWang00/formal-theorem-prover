#!/usr/bin/env python3
"""Smoke-test EOS-fixed adapter loading and merged/unmerged greedy equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = (
    "ANCHOR-A0-EOS-FIXED",
    "ANCHOR-A5-EOS-FIXED",
    "ANCHOR-A10-EOS-FIXED",
    "ANCHOR-A20-EOS-FIXED",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "model_prompt", "formatted_prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    statement = str(
        row.get("statement")
        or row.get("lean_statement")
        or row.get("formal_statement")
        or ""
    ).strip()
    return (
        "Complete the Lean 4 proof. Return only the proof body.\n\n"
        f"{statement}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/anchor_ratio_eos_fixed"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    root = (project / args.root).resolve()
    base = project / "models/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise RuntimeError(
            "frozen tokenizer identity failed: "
            f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
        )
    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        raise RuntimeError("EOS and PAD must be distinct")
    canary = read_jsonl(
        root
        / "evaluation/behavior_canary/manifests/wb_canary30.jsonl"
    )[0]
    prompt = prompt_text(canary)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    results: dict[str, Any] = {}
    for name in MODELS:
        adapter = root / "checkpoints" / name / "best"
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model = PeftModel.from_pretrained(base_model, adapter).eval()
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            unmerged = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=16,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        merged_model = model.merge_and_unload().eval()
        with torch.inference_mode():
            merged = merged_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=16,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        unmerged_tokens = unmerged[0, encoded["input_ids"].shape[1] :].tolist()
        merged_tokens = merged[0, encoded["input_ids"].shape[1] :].tolist()
        results[name] = {
            "base_path": str(base),
            "adapter_path": str(adapter),
            "device": device,
            "dtype": str(dtype),
            "tokenizer_path": str(base),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "stop_token_ids": [tokenizer.eos_token_id],
            "unmerged_tokens": unmerged_tokens,
            "merged_tokens": merged_tokens,
            "exact_token_match": unmerged_tokens == merged_tokens,
            "loaded_trained_adapter": True,
        }
        del merged_model, model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "status": (
            "PASSED"
            if all(row["exact_token_match"] for row in results.values())
            else "FAILED"
        ),
        "models": results,
    }
    output = root / "audit/inference_contract.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "PASSED":
        raise RuntimeError(f"merged/unmerged smoke test failed; inspect {output}")
    print(output)


if __name__ == "__main__":
    main()
