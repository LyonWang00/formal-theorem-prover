#!/usr/bin/env python3
"""Create a genuine zero-step LoRA save/merge/load control from clean M0."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import torch

from lean_prover.lean_training.data.random_manifest import file_sha256, write_json
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import build_trainer


EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-contract", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C0 output: {output}")
    adapter = output / "adapter"
    merged_path = output / "merged"
    manifest = json.loads(args.experiment_manifest.read_text(encoding="utf-8"))
    model_path = args.model.resolve()
    if str(model_path) != manifest["m0"]["absolute_path"]:
        raise ValueError("C0 model does not match recorded clean M0")

    config = SFTTrainConfig(
        model_name_or_path=str(model_path),
        train_file=str(args.train_contract.resolve()),
        validation_file=str(args.eval.resolve()),
        output_dir=str(adapter),
        sampling_strategy="fixed_manifest_without_replacement",
        sample_weight_field=None,
        requested_packing=False,
        require_pantograph_verified=True,
        max_seq_length=1024,
        allow_overlength=False,
        per_device_train_batch_size=1,
        per_device_validation_batch_size=1,
        gradient_accumulation_steps=16,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        max_steps=-1,
        learning_rate=2e-6,
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        warmup_ratio=0.0,
        warmup_steps=0,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        device_map="auto",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        seed=args.seed,
        data_seed=args.seed,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    lora_b_tensors = {
        name: parameter.detach()
        for name, parameter in trainer.model.named_parameters()
        if "lora_B" in name
    }
    if not lora_b_tensors:
        raise RuntimeError("C0 model contains no LoRA B matrices")
    lora_b_abs_max = max(
        float(parameter.float().abs().max().cpu())
        for parameter in lora_b_tensors.values()
    )
    if lora_b_abs_max != 0.0:
        raise RuntimeError(
            f"standard zero-delta LoRA initialization violated: {lora_b_abs_max}"
        )
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if set(adapter_config.get("target_modules") or []) != EXPECTED_TARGET_MODULES:
        raise ValueError("C0 target modules differ from the experiment contract")
    if adapter_config.get("r") != 32 or adapter_config.get("lora_alpha") != 64:
        raise ValueError("C0 LoRA rank/alpha differs from the experiment contract")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged_path.mkdir(parents=True)
    merged.save_pretrained(merged_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        str(adapter), trust_remote_code=True
    ).save_pretrained(merged_path)
    del merged, peft_model, base
    gc.collect()

    contract = {
        "checkpoint_role": "zero_step_control",
        "not_m1": True,
        "optimizer_steps": 0,
        "initialization_checkpoint": str(model_path),
        "m0_model_hash": manifest["m0"]["model_hash"],
        "m0_config_hash": manifest["m0"]["config_hash"],
        "tokenizer_hash": manifest["m0"]["tokenizer_hash"],
        "generation_config_hash": manifest["m0"]["generation_config_hash"],
        "training_manifest_sha256": file_sha256(args.train_contract),
        "load_in_4bit": True,
        "adapter_saved": True,
        "safe_merge": True,
        "merge_dtype": "torch.bfloat16",
        "lora_b_abs_max_at_initialization": lora_b_abs_max,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": sorted(EXPECTED_TARGET_MODULES),
        },
        "adapter_path": str(adapter),
        "merged_path": str(merged_path),
        "adapter_model_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "merged_model_sha256": file_sha256(merged_path / "model.safetensors"),
        "wall_seconds": round(time.monotonic() - started, 4),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(output / "checkpoint_contract.json", contract)
    write_json(adapter / "checkpoint_contract.json", contract)
    write_json(merged_path / "checkpoint_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
