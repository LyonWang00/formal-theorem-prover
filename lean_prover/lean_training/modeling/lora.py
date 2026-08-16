"""Workflow-specific PEFT LoRA configuration helpers."""

from peft import LoraConfig


def build_sft_lora_config(*, r: int, alpha: int, dropout: float) -> LoraConfig:
    """Build the LoRA adapter configuration used by supervised fine-tuning."""

    return _build_causal_lora_config(r=r, alpha=alpha, dropout=dropout)


def build_grpo_lora_config(*, r: int, alpha: int, dropout: float) -> LoraConfig:
    """Build the independently configurable LoRA adapter used by GRPO."""

    return _build_causal_lora_config(r=r, alpha=alpha, dropout=dropout)


def _build_causal_lora_config(*, r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


# Backward-compatible import for external callers; new pipelines use the
# workflow-specific builders above.
build_causal_lora_config = build_sft_lora_config
