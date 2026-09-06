"""Shared 4-bit model loading for QLoRA training."""

import os

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def qlora_compute_dtype(workflow: str) -> torch.dtype:
    """Select BF16 when supported and FP16 otherwise."""

    if not torch.cuda.is_available():
        raise RuntimeError(f"{workflow} requires CUDA; no CUDA device is available.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_qlora_model(
    model_name_or_path: str,
    *,
    compute_dtype: torch.dtype,
    device_map: str | dict[str, int] | None,
):
    """Load an NF4-quantized causal model and prepare it for k-bit training."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    if os.environ.get("LEAN_TRAINING_SKIP_CUDA_ALLOCATOR_WARMUP") == "1":
        import transformers.modeling_utils as modeling_utils

        original_warmup = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        finally:
            modeling_utils.caching_allocator_warmup = original_warmup
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model)
