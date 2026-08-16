#!/usr/bin/env python3
"""Generate and compare fixed greedy outputs across clean-M0 checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def fixed_prompts(dataset: Path, *, count: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset)
    if len(rows) < count:
        raise ValueError(f"dataset has {len(rows)} rows, needs {count}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = []
    for index in indices:
        row = rows[index]
        prompt = str(row.get("prompt") or "")
        if not prompt:
            raise ValueError(f"dataset row {index} has no prompt")
        selected.append(
            {
                "prompt_index": index,
                "record_id": row.get("record_id") or row.get("id"),
                "statement_id": row.get("statement_id"),
                "prompt": prompt,
            }
        )
    return selected


def run_transformers(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model).expanduser().resolve()
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    tokenizer_path = adapter_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    prompts = fixed_prompts(Path(args.dataset), count=args.count, seed=args.seed)
    output_rows = []
    for row in prompts:
        inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        token_ids = generated[0, input_length:].tolist()
        output_rows.append(
            {
                **row,
                "backend": "transformers",
                "label": args.label,
                "model_path": str(model_path),
                "adapter_path": str(adapter_path) if adapter_path else None,
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "eos_token_id": tokenizer.eos_token_id,
                "token_ids": token_ids,
                "output_text": tokenizer.decode(token_ids, skip_special_tokens=False),
            }
        )
    write_jsonl(Path(args.output), output_rows)


def run_vllm(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = Path(args.model).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompts = fixed_prompts(Path(args.dataset), count=args.count, seed=args.seed)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        seed=args.seed,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=(
            [tokenizer.eos_token_id]
            if tokenizer.eos_token_id is not None
            else None
        ),
    )
    generated = llm.generate([row["prompt"] for row in prompts], params)
    output_rows = []
    for row, request in zip(prompts, generated, strict=True):
        candidate = request.outputs[0]
        token_ids = list(candidate.token_ids)
        output_rows.append(
            {
                **row,
                "backend": "vllm",
                "label": args.label,
                "model_path": str(model_path),
                "adapter_path": None,
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "eos_token_id": tokenizer.eos_token_id,
                "token_ids": token_ids,
                "output_text": candidate.text,
                "finish_reason": candidate.finish_reason,
            }
        )
    write_jsonl(Path(args.output), output_rows)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def build_report(files: dict[str, Path]) -> dict[str, Any]:
    tables = {label: read_jsonl(path) for label, path in files.items()}
    counts = {label: len(rows) for label, rows in tables.items()}
    if len(set(counts.values())) != 1 or not counts or next(iter(counts.values())) != 10:
        raise ValueError(f"equivalence inputs must each contain 10 rows: {counts}")
    labels = ("base", "adapter", "merged", "vllm")
    details = []
    for index in range(10):
        rows = {label: tables[label][index] for label in labels}
        record_ids = {row["record_id"] for row in rows.values()}
        if len(record_ids) != 1:
            raise ValueError(f"prompt alignment mismatch at index {index}")
        token_ids = {label: list(rows[label]["token_ids"]) for label in labels}
        details.append(
            {
                "record_id": rows["base"]["record_id"],
                "base_adapter_exact": token_ids["base"] == token_ids["adapter"],
                "adapter_merged_exact": token_ids["adapter"] == token_ids["merged"],
                "merged_vllm_exact": token_ids["merged"] == token_ids["vllm"],
                "adapter_merged_common_prefix_tokens": common_prefix_length(
                    token_ids["adapter"], token_ids["merged"]
                ),
                "merged_vllm_common_prefix_tokens": common_prefix_length(
                    token_ids["merged"], token_ids["vllm"]
                ),
                "token_counts": {
                    label: len(value) for label, value in token_ids.items()
                },
                "outputs": {
                    label: rows[label]["output_text"] for label in labels
                },
            }
        )
    base_adapter_different = sum(not row["base_adapter_exact"] for row in details)
    adapter_merged_exact = sum(row["adapter_merged_exact"] for row in details)
    merged_vllm_exact = sum(row["merged_vllm_exact"] for row in details)
    adapter_merged_first = sum(
        row["adapter_merged_common_prefix_tokens"] >= 1 for row in details
    )
    merged_vllm_first = sum(
        row["merged_vllm_common_prefix_tokens"] >= 1 for row in details
    )
    return {
        "success": (
            base_adapter_different > 0
            and adapter_merged_first == 10
            and merged_vllm_first == 10
        ),
        "generation_contract": {
            "prompts": 10,
            "do_sample": False,
            "temperature": 0.0,
        },
        "actual_paths": {
            label: tables[label][0]["model_path"] for label in labels
        },
        "base_adapter_different": base_adapter_different,
        "adapter_merged_exact_matches": adapter_merged_exact,
        "adapter_merged_first_token_matches": adapter_merged_first,
        "merged_vllm_exact_matches": merged_vllm_exact,
        "merged_vllm_first_token_matches": merged_vllm_first,
        "details": details,
    }


def run_report(args: argparse.Namespace) -> None:
    files = {
        label: Path(getattr(args, label))
        for label in ("base", "adapter", "merged", "vllm")
    }
    report = build_report(files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("transformers", "vllm"):
        child = subparsers.add_parser(name)
        child.add_argument("--model", required=True)
        child.add_argument("--adapter", default=None)
        child.add_argument("--dataset", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--label", required=True)
        child.add_argument("--count", type=int, default=10)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--max-new-tokens", type=int, default=64)
    vllm_parser = subparsers.choices["vllm"]
    vllm_parser.add_argument("--max-model-len", type=int, default=1152)
    vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    report = subparsers.add_parser("report")
    for label in ("base", "adapter", "merged", "vllm"):
        report.add_argument(f"--{label}", required=True)
    report.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "transformers":
        run_transformers(args)
    elif args.command == "vllm":
        run_vllm(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
