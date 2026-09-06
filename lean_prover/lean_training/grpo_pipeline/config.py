"""Configuration schema for GRPO training with Pantograph rewards."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GRPOTrainConfig:
    """Complete GRPO, QLoRA, Pantograph reward, and output configuration."""

    model_name_or_path: str
    train_file: str
    output_dir: str
    validation_file: str | None = None
    prompt_field: str = "prompt"
    statement_field: str = "lean_statement"
    id_field: str = "id"
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    per_device_validation_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    eval_steps: int | None = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lean_project_path: str = "lean_project"
    pantograph_imports: str = "Mathlib"
    lean_timeout: int = 120
    compile_success_reward: float = 1.0
    format_reward: float = 0.1
    brevity_reward: float = 0.1
    failure_reward: float = 0.0
    reward_log_file: str | None = None
    seed: int = 20260715
    data_seed: int = 20260715
