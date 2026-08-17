"""Configuration schema for supervised QLoRA training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SFTTrainConfig:
    """Complete dataset, QLoRA, optimization, evaluation, and output configuration."""

    model_name_or_path: str
    train_file: str
    output_dir: str
    adapter_path: str | None = None
    validation_file: str | None = None
    train_manifest_file: str | None = None
    validation_manifest_file: str | None = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    sample_weight_field: str | None = None
    sampling_strategy: str = "auto"
    requested_packing: bool = False
    require_pantograph_verified: bool = True
    require_supervised_eos: bool = True
    max_seq_length: int = 1024
    allow_overlength: bool = False
    per_device_train_batch_size: int = 1
    per_device_validation_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    max_steps: int = -1
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    eval_steps: int | None = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int | None = None
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    # Preserve the legacy programmatic contract.  The canonical CLI and DDP
    # launcher explicitly select local_rank; distributed validation rejects auto.
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 20260711
    data_seed: int = 20260711
