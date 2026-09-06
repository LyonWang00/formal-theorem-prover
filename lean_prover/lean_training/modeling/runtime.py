"""Shared package-version and reproducibility helpers."""

from importlib import metadata
from typing import Protocol

import torch


class SeedConfig(Protocol):
    seed: int


def package_version(package: str) -> str:
    """Return an installed package version or ``not-installed``."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def set_reproducible_seeds(config: SeedConfig) -> None:
    """Seed Python and Torch from a training configuration."""

    import random

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

