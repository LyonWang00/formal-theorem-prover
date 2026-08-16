"""Validated user overrides for hardware-aware runtime profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RuntimeGenerationConfig(BaseModel):
    mode: Literal["isolated", "persistent"] | None = None
    batch_size: int | None = Field(default=None, gt=0)
    persistent: bool | None = None


class RuntimeVerificationConfig(BaseModel):
    lifecycle: Literal["stage", "persistent"] | None = None
    workers: int | None = Field(default=None, gt=0)
    persistent: bool | None = None


class RuntimeMemoryConfig(BaseModel):
    streaming: bool | None = None
    max_queue_size: int | None = Field(default=None, gt=0)


class RuntimeCacheConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"


class RuntimeDebugConfig(BaseModel):
    save_full_source_on_failure_only: bool = True


class MemoryGuardConfig(BaseModel):
    enabled: bool = True
    warning_ratio: float = Field(default=0.85, gt=0, lt=1)
    critical_ratio: float = Field(default=0.95, gt=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "MemoryGuardConfig":
        if self.warning_ratio >= self.critical_ratio:
            raise ValueError("memory_guard.warning_ratio must be below critical_ratio")
        return self


class RuntimeConfig(BaseModel):
    profile: Literal["laptop", "server"] = "laptop"
    generation_verification_mode: Literal["sequential", "pipeline", "async"] | None = None
    generation: RuntimeGenerationConfig = Field(default_factory=RuntimeGenerationConfig)
    verification: RuntimeVerificationConfig = Field(default_factory=RuntimeVerificationConfig)
    memory: RuntimeMemoryConfig = Field(default_factory=RuntimeMemoryConfig)
    cache: RuntimeCacheConfig = Field(default_factory=RuntimeCacheConfig)
    debug: RuntimeDebugConfig = Field(default_factory=RuntimeDebugConfig)
    memory_guard: MemoryGuardConfig = Field(default_factory=MemoryGuardConfig)
