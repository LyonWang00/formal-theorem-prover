"""GRPO training pipeline with Pantograph-backed rewards."""

from .config import GRPOTrainConfig
from .rewards import PantographRewardFunction
from .trainer import build_trainer, main, parse_args

__all__ = [
    "GRPOTrainConfig",
    "PantographRewardFunction",
    "build_trainer",
    "main",
    "parse_args",
]
