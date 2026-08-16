#!/usr/bin/env python3
"""Low-memory root-cause diagnostics for the sole Task 0A divergence.

Models are loaded strictly one at a time.  BF16 paths repeat greedy generation
for the one disputed prompt; FP32 paths perform only one next-token forward on
the frozen common prefix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


STATEMENT_ID = "stmt_f6798f8287b817aaae741301"
BASE_REL = Path("models/Qwen2.5-1.5B-Instruct")
ADAPTER_REL = Path(
    "outputs/wb_ld_budget_support_replay_ablation/checkpoints/"
    "ADDON-B-WB2000-LD1000/best"
)
SHARED_REL = Path("outputs/stage2_sft_incremental_ablation/shared")
MERGED_BF16_REL = SHARED_REL / "checkpoints/M0-ADDON-B-FROZEN-MERGED"
BASE_FP32_REL = SHARED_REL / "checkpoints/Qwen2.5-1.5B-Instruct-FP32-VIEW"
MERGED_FP32_REL = SHARED_REL / "checkpoints/M0-ADDON-B-FROZEN-MERGED-FP32"
REPAIR_REL = SHARED_REL / "m0_checkpoint_equivalence_repair"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_tensor_dtypes(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                counts[str(handle.get_slice(key).get_dtype())] += 1
    return dict(sorted(counts.items()))


def parameter_dtypes(model: torch.nn.Module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in model.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def current_rss_mib() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 2)
    return -1.0


def cuda_memory_mib() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved": round(torch.cuda.memory_reserved() / (1024**2), 2),
    }


def free_model(model: torch.nn.Module | None) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def token_info(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "id": int(token_id),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False),
    }


def logits_summary(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    left_target: int,
    right_target: int,
    k: int = 10,
) -> dict[str, Any]:
    values = logits.detach().float().cpu()
    top_values, top_ids = torch.topk(values, k=k)
    rows = []
    for token_id, value in zip(top_ids.tolist(), top_values.tolist()):
        row = token_info(tokenizer, token_id)
        row["logit"] = float(value)
        rows.append(row)
    return {
        "logits_dtype_before_float_cast": str(logits.dtype),
        "top_k": rows,
        "top1_top2_margin": float(top_values[0] - top_values[1]),
        "left_target": {
            **token_info(tokenizer, left_target),
            "logit": float(values[left_target]),
            "gap_from_top1": float(top_values[0] - values[left_target]),
        },
        "right_target": {
            **token_info(tokenizer, right_target),
            "logit": float(values[right_target]),
            "gap_from_top1": float(top_values[0] - values[right_target]),
        },
    }


def load_model(
    model_path: Path,
    *,
    dtype: torch.dtype,
    adapter_path: Path | None,
    cpu_only: bool,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if cpu_only:
        kwargs["device_map"] = {"": "cpu"}
    else:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def model_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def build_inputs(
    tokenizer: Any, prompt: str, common_completion_ids: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    suffix = torch.tensor([common_completion_ids], dtype=torch.long)
    input_ids = torch.cat([encoded["input_ids"], suffix], dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_forward(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    common_completion_ids: list[int],
    *,
    left_target: int,
    right_target: int,
) -> dict[str, Any]:
    inputs = build_inputs(tokenizer, prompt, common_completion_ids, model_device(model))
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
        next_logits = output.logits[0, -1]
    summary = logits_summary(
        next_logits,
        tokenizer,
        left_target=left_target,
        right_target=right_target,
    )
    summary.update(
        {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "model_device": str(model_device(model)),
            "parameter_dtypes": parameter_dtypes(model),
            "rss_mib": current_rss_mib(),
            "cuda_memory_mib": cuda_memory_mib(),
        }
    )
    del output, next_logits, inputs
    return summary


def repeat_greedy_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    repeats: int = 2,
    left_target: int | None = None,
    right_target: int | None = None,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device(model)) for key, value in encoded.items()}
    completions: list[str] = []
    token_sequences: list[list[int]] = []
    first_repeat_step_scores: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        torch.manual_seed(20261701)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20261701)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=(repeat_index == 0),
                output_scores=(repeat_index == 0),
            )
        sequences = generated.sequences if repeat_index == 0 else generated
        completion_ids = sequences[0, inputs["input_ids"].shape[-1] :].tolist()
        if repeat_index == 0 and left_target is not None and right_target is not None:
            first_repeat_step_scores = [
                logits_summary(
                    score[0],
                    tokenizer,
                    left_target=left_target,
                    right_target=right_target,
                )
                for score in generated.scores
            ]
        token_sequences.append(completion_ids)
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
        del generated
    return {
        "repeat_count": repeats,
        "identical_token_sequences": all(
            row == token_sequences[0] for row in token_sequences[1:]
        ),
        "completion_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in completions
        ],
        "completion_token_counts": [len(row) for row in token_sequences],
        "completions": completions,
        "token_sequences": token_sequences,
        "first_repeat_step_scores": first_repeat_step_scores,
    }


def tokenizer_contract(
    tokenizer: Any, prompt: str, left_output: str, right_output: str
) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    left_ids = tokenizer(left_output, add_special_tokens=False)["input_ids"]
    right_ids = tokenizer(right_output, add_special_tokens=False)["input_ids"]
    prefix_len = common_prefix(left_ids, right_ids)
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "prompt_token_count": len(prompt_ids),
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_ids": prompt_ids,
        "left_completion_ids": left_ids,
        "right_completion_ids": right_ids,
        "completion_common_prefix_tokens": prefix_len,
        "left_fork": token_info(tokenizer, left_ids[prefix_len]),
        "right_fork": token_info(tokenizer, right_ids[prefix_len]),
        "common_prefix_text": tokenizer.decode(
            left_ids[:prefix_len], skip_special_tokens=False
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--skip-bf16-generation", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    base = project / BASE_REL
    adapter = project / ADAPTER_REL
    merged_bf16 = project / MERGED_BF16_REL
    base_fp32 = project / BASE_FP32_REL
    merged_fp32 = project / MERGED_FP32_REL
    repair = project / REPAIR_REL
    repair.mkdir(parents=True, exist_ok=True)

    left_row = next(
        row
        for row in read_jsonl(repair / "results/unmerged/generations.jsonl")
        if row["statement_id"] == STATEMENT_ID
    )
    right_row = next(
        row
        for row in read_jsonl(repair / "results/merged/generations.jsonl")
        if row["statement_id"] == STATEMENT_ID
    )
    prompt = left_row["prompt"]
    if prompt != right_row["prompt"]:
        raise RuntimeError("Disputed rows do not share an identical prompt")
    left_output = left_row["raw_output"]
    right_output = right_row["raw_output"]

    tokenizer_paths = {"base": base, "adapter": adapter, "merged": merged_bf16}
    tokenizers: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    for name, path in tokenizer_paths.items():
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizers[name] = tokenizer
        contracts[name] = tokenizer_contract(
            tokenizer, prompt, left_output, right_output
        )

    base_contract = contracts["base"]
    merged_contract = contracts["merged"]
    token_alignment = {
        "base_vs_adapter_prompt_ids_equal": (
            contracts["base"]["prompt_ids"] == contracts["adapter"]["prompt_ids"]
        ),
        "base_vs_merged_prompt_ids_equal": (
            contracts["base"]["prompt_ids"] == contracts["merged"]["prompt_ids"]
        ),
        "base_vs_merged_left_completion_ids_equal": (
            contracts["base"]["left_completion_ids"]
            == contracts["merged"]["left_completion_ids"]
        ),
        "base_vs_merged_right_completion_ids_equal": (
            contracts["base"]["right_completion_ids"]
            == contracts["merged"]["right_completion_ids"]
        ),
    }
    prefix_len = int(base_contract["completion_common_prefix_tokens"])
    if prefix_len != int(merged_contract["completion_common_prefix_tokens"]):
        raise RuntimeError("Base and merged tokenizers disagree on fork position")
    common_ids = base_contract["left_completion_ids"][:prefix_len]
    if common_ids != merged_contract["left_completion_ids"][:prefix_len]:
        raise RuntimeError("Base and merged tokenizers disagree on common prefix IDs")
    left_target = int(base_contract["left_completion_ids"][prefix_len])
    right_target = int(base_contract["right_completion_ids"][prefix_len])

    adapter_config = read_json(adapter / "adapter_config.json")
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    use_rslora = bool(adapter_config.get("use_rslora", False))
    scaling = alpha / (rank**0.5 if use_rslora else rank)
    audit = {
        "statement_id": STATEMENT_ID,
        "paths": {
            "base": str(base),
            "adapter": str(adapter),
            "merged_bf16": str(merged_bf16),
            "base_fp32": str(base_fp32),
            "merged_fp32": str(merged_fp32),
        },
        "checkpoint_tensor_dtypes": {
            "base": model_tensor_dtypes(base),
            "adapter": model_tensor_dtypes(adapter),
            "merged_bf16": model_tensor_dtypes(merged_bf16),
            "base_fp32": model_tensor_dtypes(base_fp32),
            "merged_fp32": model_tensor_dtypes(merged_fp32),
        },
        "config_dtypes": {
            "base": read_json(base / "config.json").get("torch_dtype")
            or read_json(base / "config.json").get("dtype"),
            "merged_bf16": read_json(merged_bf16 / "config.json").get(
                "torch_dtype"
            )
            or read_json(merged_bf16 / "config.json").get("dtype"),
            "base_fp32": read_json(base_fp32 / "config.json").get("torch_dtype")
            or read_json(base_fp32 / "config.json").get("dtype"),
            "merged_fp32": read_json(merged_fp32 / "config.json").get(
                "torch_dtype"
            )
            or read_json(merged_fp32 / "config.json").get("dtype"),
        },
        "adapter": {
            "r": rank,
            "lora_alpha": alpha,
            "use_rslora": use_rslora,
            "derived_scaling": scaling,
            "target_modules": adapter_config.get("target_modules"),
            "fan_in_fan_out": adapter_config.get("fan_in_fan_out"),
            "bias": adapter_config.get("bias"),
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        },
        "merge_manifests": {
            "bf16": read_json(merged_bf16 / "merge_manifest.json"),
            "fp32": read_json(merged_fp32 / "merge_manifest.json"),
        },
        "generation_configs": {
            "base": read_json(base / "generation_config.json"),
            "merged_bf16": read_json(merged_bf16 / "generation_config.json"),
            "matched_eval": read_json(repair / "matched_transformers_bf16_config.json"),
            "effective_call": {
                "backend": "transformers",
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": "not passed when temperature <= 0",
                "top_p": "not passed when temperature <= 0",
                "seed": 20261701,
                "num_return_sequences": 1,
                "padding_side": "left",
            },
        },
        "tokenizer_file_hashes": {
            name: {
                filename: sha256(path / filename)
                for filename in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
            }
            for name, path in tokenizer_paths.items()
        },
        "tokenizer_contracts": contracts,
        "token_alignment": token_alignment,
        "fork": {
            "completion_token_index_zero_based": prefix_len,
            "generated_token_number_one_based": prefix_len + 1,
            "prompt_token_count": len(base_contract["prompt_ids"]),
            "absolute_next_token_position_zero_based": len(
                base_contract["prompt_ids"]
            )
            + prefix_len,
            "common_prefix_ids": common_ids,
            "common_prefix_text": base_contract["common_prefix_text"],
            "unmerged_next": token_info(tokenizers["base"], left_target),
            "merged_next": token_info(tokenizers["base"], right_target),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "initial_rss_mib": current_rss_mib(),
        },
    }

    results: dict[str, Any] = {
        "schema_version": 1,
        "audit": audit,
        "bf16": {},
        "fp32_single_step": {},
    }

    for side, model_path, adapter_path, tokenizer_name in (
        ("unmerged", base, adapter, "base"),
        ("merged", merged_bf16, None, "merged"),
    ):
        print(f"Loading BF16 {side} path", flush=True)
        model = load_model(
            model_path,
            dtype=torch.bfloat16,
            adapter_path=adapter_path,
            cpu_only=False,
        )
        side_result: dict[str, Any] = {}
        if not args.skip_bf16_generation:
            side_result["determinism"] = repeat_greedy_generation(
                model, tokenizers[tokenizer_name], prompt
            )
        side_result["fork_logits"] = run_forward(
            model,
            tokenizers[tokenizer_name],
            prompt,
            common_ids,
            left_target=left_target,
            right_target=right_target,
        )
        results["bf16"][side] = side_result
        del side_result
        free_model(model)
        model = None
        results["bf16"][side]["post_release_rss_mib"] = current_rss_mib()
        results["bf16"][side]["post_release_cuda_memory_mib"] = cuda_memory_mib()
        (repair / "task0a_root_cause_diagnostics.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for side, model_path, adapter_path, tokenizer_name in (
        ("unmerged", base_fp32, adapter, "base"),
        ("merged", merged_fp32, None, "merged"),
    ):
        print(f"Loading FP32 {side} path for one forward", flush=True)
        model = load_model(
            model_path,
            dtype=torch.float32,
            adapter_path=adapter_path,
            cpu_only=True,
        )
        results["fp32_single_step"][side] = run_forward(
            model,
            tokenizers[tokenizer_name],
            prompt,
            common_ids,
            left_target=left_target,
            right_target=right_target,
        )
        free_model(model)
        model = None
        results["fp32_single_step"][side]["post_release_rss_mib"] = (
            current_rss_mib()
        )
        (repair / "task0a_root_cause_diagnostics.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    results["runtime_final"] = {
        "rss_mib": current_rss_mib(),
        "cuda_memory_mib": cuda_memory_mib(),
    }
    output = repair / "task0a_root_cause_diagnostics.json"
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial = repair / "task0a_root_cause_diagnostics.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps({"output": str(output), "fork": audit["fork"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
