"""Low-overhead RAM/GPU telemetry and cooperative memory-pressure guards."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


class MemoryPressureError(RuntimeError):
    """Raised after a stage was stopped and durable artifacts were flushed."""


class MemoryMonitor:
    def __init__(self, output_path: str | Path, *, enabled: bool = True, warning_ratio: float = 0.85, critical_ratio: float = 0.95) -> None:
        self.output_path = Path(output_path)
        self.enabled = enabled
        self.warning_ratio = warning_ratio
        self.critical_ratio = critical_ratio
        self.peak_system_ratio = 0.0
        self.peak_process_rss_bytes = 0

    def snapshot(self, stage: str, *, active_processes: list[dict[str, Any]] | None = None, on_critical: Callable[[], None] | None = None, raise_on_critical: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"stage": stage, "enabled": False}
        memory = _memory_info()
        ratio = float(memory["system_used_ratio"])
        rss = int(memory["process_rss_bytes"])
        self.peak_system_ratio = max(self.peak_system_ratio, ratio)
        self.peak_process_rss_bytes = max(self.peak_process_rss_bytes, rss)
        level = "critical" if ratio >= self.critical_ratio else "warning" if ratio >= self.warning_ratio else "normal"
        row = {
            "timestamp": time.time(), "stage": stage, "level": level, **memory,
            "gpu": _gpu_memory(), "active_processes": active_processes or [],
            "peak_system_used_ratio": self.peak_system_ratio,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
        }
        self._append(row)
        if level == "critical":
            if on_critical is not None:
                on_critical()
            release_unused_memory()
            if raise_on_critical:
                raise MemoryPressureError(
                    f"critical system memory pressure at {ratio:.1%} during {stage}; "
                    "stage resources were stopped and durable JSONL can be resumed"
                )
        return row

    def summary(self) -> dict[str, Any]:
        return {"peak_system_used_ratio": self.peak_system_ratio, "peak_process_rss_bytes": self.peak_process_rss_bytes, "log_path": str(self.output_path)}

    def _append(self, row: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def release_unused_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def _memory_info() -> dict[str, int | float]:
    try:
        import psutil
        virtual = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        return {
            "process_rss_bytes": int(process.memory_info().rss),
            "system_total_bytes": int(virtual.total),
            "system_available_bytes": int(virtual.available),
            "system_used_bytes": int(virtual.total - virtual.available),
            "system_used_ratio": float((virtual.total - virtual.available) / virtual.total),
        }
    except ImportError:
        return {"process_rss_bytes": 0, "system_total_bytes": 0, "system_available_bytes": 0, "system_used_bytes": 0, "system_used_ratio": 0.0}


def _gpu_memory() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"index": int(parts[0]), "used_mib": int(parts[1]), "total_mib": int(parts[2])})
    return rows
