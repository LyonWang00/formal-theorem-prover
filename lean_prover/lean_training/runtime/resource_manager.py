"""Coordinator-facing facade for resource lifetimes and memory safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .lifecycle import ResourceLifecycleManager
from .memory_monitor import MemoryMonitor
from .profile import RuntimeProfile


class ResourceManager:
    def __init__(self, profile: RuntimeProfile, *, run_dir: str | Path, generation_stopper: Callable[[], None], pantograph_factory: Callable[[], Any]) -> None:
        run_dir = Path(run_dir)
        self.profile = profile
        self.monitor = MemoryMonitor(
            run_dir / "runtime" / "memory.jsonl",
            enabled=profile.memory_guard_enabled,
            warning_ratio=profile.memory_warning_ratio,
            critical_ratio=profile.memory_critical_ratio,
        )
        self.lifecycle = ResourceLifecycleManager(
            profile, output_path=run_dir / "runtime" / "lifecycle.jsonl",
            memory_monitor=self.monitor, generation_stopper=generation_stopper,
            pantograph_factory=pantograph_factory,
        )
        self.verification_identity: str | None = None

    @property
    def pantograph_pool(self):
        return self.lifecycle.pantograph_pool

    def start_generation(self) -> None:
        self.lifecycle.start_generation_resources()
        self.check_memory("GENERATE_START")

    def stop_generation(self) -> None:
        self.lifecycle.stop_generation_resources()
        self.check_memory("GENERATE_END")

    def start_verification(self, *, identity: str | None = None):
        if (
            self.pantograph_pool is not None
            and identity is not None
            and self.verification_identity != identity
        ):
            self.lifecycle.stop_pantograph(force=True)
        pool = self.lifecycle.start_verification_resources()
        self.verification_identity = identity
        self.check_memory("VERIFY_START")
        return pool

    def stop_verification(self) -> None:
        self.lifecycle.stop_verification_resources()
        self.check_memory("VERIFY_END")

    def before_training(self) -> None:
        self.lifecycle.before_training()
        self.check_memory("TRAIN_START")

    def after_training(self) -> None:
        self.lifecycle.after_training()
        self.check_memory("TRAIN_END")

    def check_memory(self, stage: str, *, raise_on_critical: bool = False):
        return self.monitor.snapshot(
            stage, active_processes=self.lifecycle.active_processes(),
            on_critical=self.lifecycle.cleanup_all, raise_on_critical=raise_on_critical,
        )

    def cleanup_all(self) -> None:
        self.lifecycle.cleanup_all()
        self.verification_identity = None
        self.check_memory("CLEANUP_END")
