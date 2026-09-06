"""Small, explicit distributed-runtime helpers for SFT.

The Hugging Face Trainer owns DDP wrapping and gradient synchronization.  This
module only resolves process identity, binds each QLoRA process to its local
GPU, and exposes a main-process gate for non-Trainer filesystem writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from accelerate import PartialState


@dataclass(frozen=True)
class DistributedContext:
    """Process identity initialized from ``accelerate launch``."""

    state: PartialState
    world_size: int
    rank: int
    local_rank: int
    is_main_process: bool

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def wait_for_everyone(self) -> None:
        self.state.wait_for_everyone()

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "is_main_process": self.is_main_process,
            "distributed_type": str(self.state.distributed_type),
            "device": str(self.state.device),
        }


def initialize_distributed_context() -> DistributedContext:
    """Initialize Accelerate process state without wrapping model or data."""

    state = PartialState()
    context = DistributedContext(
        state=state,
        world_size=int(state.num_processes),
        rank=int(state.process_index),
        local_rank=int(state.local_process_index),
        is_main_process=bool(state.is_main_process),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA SFT requires CUDA on every Accelerate process")
    visible_devices = torch.cuda.device_count()
    if context.local_rank < 0 or context.local_rank >= visible_devices:
        raise RuntimeError(
            "LOCAL_RANK is outside the visible CUDA device set: "
            f"local_rank={context.local_rank}, visible_devices={visible_devices}"
        )
    return context


def resolve_qlora_device_map(
    requested: str,
    context: DistributedContext,
) -> str | dict[str, int] | None:
    """Return a one-process/one-GPU map and reject unsafe DDP placement.

    ``device_map=auto`` remains accepted for legacy single-process experiments,
    but is deliberately rejected when multiple processes are active because it
    may shard one model across GPUs already owned by other DDP ranks.
    """

    normalized = requested.strip().lower()
    if normalized in {"local_rank", "local"}:
        return {"": context.local_rank}
    if context.is_distributed:
        raise ValueError(
            "Distributed SFT requires --device_map local_rank; "
            f"received {requested!r}. Do not use auto or a fixed cuda:x map under DDP."
        )
    if normalized == "auto":
        return "auto"
    if normalized in {"none", "null"}:
        return None
    if normalized.startswith("cuda") or normalized.isdigit():
        raise ValueError(
            "Fixed CUDA placement is not supported by the SFT entry point; "
            "use --device_map local_rank (recommended) or auto for a legacy "
            "single-process run."
        )
    raise ValueError(f"unsupported --device_map value: {requested!r}")


def global_batch_size(
    *, per_device_batch_size: int, gradient_accumulation_steps: int, world_size: int
) -> int:
    """Compute the effective batch without changing its component settings."""

    return per_device_batch_size * gradient_accumulation_steps * world_size


def validate_no_duplicate_train_sharding(
    *,
    dataset_size: int,
    per_device_batch_size: int,
    world_size: int,
    drop_last: bool,
) -> int:
    """Require equal DDP microbatch counts without padding or dropping rows.

    Accelerate can make uneven DataLoaders equal by replaying samples.  The SFT
    contract forbids that implicit duplication, so a non-divisible training
    plan is rejected for a human to resolve instead of being silently changed.
    Returns the number of microbatches assigned to each rank.
    """

    if dataset_size <= 0 or per_device_batch_size <= 0 or world_size <= 0:
        raise ValueError("dataset, batch, and world sizes must be positive")
    total_microbatches = (
        dataset_size // per_device_batch_size
        if drop_last
        else math.ceil(dataset_size / per_device_batch_size)
    )
    if world_size > 1 and total_microbatches % world_size:
        raise ValueError(
            "No-duplicate DDP sharding requires the number of training "
            "microbatches to be divisible by world_size: "
            f"dataset_size={dataset_size}, per_device_batch_size={per_device_batch_size}, "
            f"drop_last={drop_last}, total_microbatches={total_microbatches}, "
            f"world_size={world_size}. Refusing to replay or discard records."
        )
    return total_microbatches // world_size
