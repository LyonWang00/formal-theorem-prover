"""Profile-driven lifecycle state machine for heavyweight resources."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .memory_monitor import MemoryMonitor, release_unused_memory
from .profile import RuntimeProfile


class ResourceLifecycleManager:
    def __init__(self, profile: RuntimeProfile, *, output_path: str | Path, memory_monitor: MemoryMonitor, generation_stopper: Callable[[], None], pantograph_factory: Callable[[], Any]) -> None:
        self.profile = profile
        self.output_path = Path(output_path)
        self.memory_monitor = memory_monitor
        self.generation_stopper = generation_stopper
        self.pantograph_factory = pantograph_factory
        self.pantograph_pool: Any | None = None
        self.generation_active = False

    def start_generation_resources(self) -> None:
        if not self.profile.pantograph_persistent:
            self.stop_pantograph(force=True)
        self.generation_active = True
        self._event("vllm_start", persistent=self.profile.vllm_persistent)

    def stop_generation_resources(self, *, force: bool = False) -> None:
        if not self.generation_active:
            return
        if self.profile.vllm_persistent and not force:
            return
        self.generation_stopper()
        self.generation_active = False
        self._event("vllm_stop", persistent=self.profile.vllm_persistent)
        release_unused_memory()

    def start_verification_resources(self):
        if not self.profile.vllm_persistent:
            self.stop_generation_resources()
        if self.pantograph_pool is None or not self.pantograph_pool.started:
            self.pantograph_pool = self.pantograph_factory()
            self._event("pantograph_start", persistent=self.profile.pantograph_persistent, workers=self.profile.pantograph_workers)
        return self.pantograph_pool

    def stop_verification_resources(self, *, force: bool = False) -> None:
        if force or not self.profile.pantograph_persistent:
            self.stop_pantograph(force=True)

    def stop_pantograph(self, *, force: bool = False) -> None:
        pool = self.pantograph_pool
        if pool is None:
            return
        if force or not self.profile.pantograph_persistent:
            pool.close()
            self.pantograph_pool = None
            self._event("pantograph_stop", persistent=self.profile.pantograph_persistent)
            release_unused_memory()

    def before_training(self) -> None:
        self.stop_generation_resources(force=True)
        self.stop_pantograph(force=True)
        self._event("trainer_start")
        release_unused_memory()

    def after_training(self) -> None:
        self._event("trainer_stop")
        release_unused_memory()

    def cleanup_all(self) -> None:
        self.stop_generation_resources(force=True)
        self.stop_pantograph(force=True)
        self._event("cleanup_all")
        release_unused_memory()

    def active_processes(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.generation_active:
            rows.append({"kind": "generation", "active": True})
        if self.pantograph_pool is not None:
            for worker in self.pantograph_pool.runtime_snapshot().get("workers", []):
                rows.append({"kind": "pantograph", **worker})
        return rows

    def _event(self, event: str, **values: Any) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": time.time(), "event": event, **values}, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
