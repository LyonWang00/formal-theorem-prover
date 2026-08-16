"""Resolved immutable runtime policies for supported hardware classes."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    vllm_persistent: bool
    vllm_batch_size: int
    pantograph_persistent: bool
    pantograph_workers: int
    generation_verification_mode: str
    streaming: bool
    max_queue_size: int
    cache_backend: str
    save_full_source_on_failure_only: bool
    memory_guard_enabled: bool
    memory_warning_ratio: float
    memory_critical_ratio: float


_DEFAULTS = {
    "laptop": RuntimeProfile(
        name="laptop",
        vllm_persistent=False,
        vllm_batch_size=100,
        pantograph_persistent=False,
        pantograph_workers=2,
        generation_verification_mode="sequential",
        streaming=True,
        max_queue_size=32,
        cache_backend="sqlite",
        save_full_source_on_failure_only=True,
        memory_guard_enabled=True,
        memory_warning_ratio=0.85,
        memory_critical_ratio=0.95,
    ),
    "server": RuntimeProfile(
        name="server",
        vllm_persistent=True,
        vllm_batch_size=512,
        pantograph_persistent=True,
        pantograph_workers=16,
        generation_verification_mode="pipeline",
        streaming=True,
        max_queue_size=1000,
        cache_backend="sqlite",
        save_full_source_on_failure_only=True,
        memory_guard_enabled=True,
        memory_warning_ratio=0.85,
        memory_critical_ratio=0.95,
    ),
}


def resolve_runtime_profile(config: RuntimeConfig) -> RuntimeProfile:
    """Apply explicit runtime overrides to one of the two stable defaults."""

    base = _DEFAULTS[config.profile]
    generation = config.generation
    verification = config.verification
    generation_persistent = generation.persistent
    if generation_persistent is None and generation.mode is not None:
        generation_persistent = generation.mode == "persistent"
    verification_persistent = verification.persistent
    if verification_persistent is None and verification.lifecycle is not None:
        verification_persistent = verification.lifecycle == "persistent"
    return RuntimeProfile(
        name=base.name,
        vllm_persistent=(base.vllm_persistent if generation_persistent is None else generation_persistent),
        vllm_batch_size=generation.batch_size or base.vllm_batch_size,
        pantograph_persistent=(base.pantograph_persistent if verification_persistent is None else verification_persistent),
        pantograph_workers=verification.workers or base.pantograph_workers,
        generation_verification_mode=(config.generation_verification_mode or base.generation_verification_mode),
        streaming=(base.streaming if config.memory.streaming is None else config.memory.streaming),
        max_queue_size=config.memory.max_queue_size or base.max_queue_size,
        cache_backend=config.cache.backend,
        save_full_source_on_failure_only=config.debug.save_full_source_on_failure_only,
        memory_guard_enabled=config.memory_guard.enabled,
        memory_warning_ratio=config.memory_guard.warning_ratio,
        memory_critical_ratio=config.memory_guard.critical_ratio,
    )
