"""Execution-mode seam for sequential discovery and future pipelines."""

from __future__ import annotations

from typing import Any, Callable

from .profile import RuntimeProfile
from .resource_manager import ResourceManager


class GenerationVerificationEngine:
    """Select an execution strategy without embedding hardware checks upstream."""

    def __init__(self, profile: RuntimeProfile, resources: ResourceManager) -> None:
        self.profile = profile
        self.resources = resources

    def run(self, generation: Callable[[], Any], verification: Callable[[], Any]) -> tuple[Any, Any]:
        mode = self.profile.generation_verification_mode
        if mode == "sequential":
            return self.run_sequential(generation, verification)
        if mode == "pipeline":
            return self.run_pipeline(generation, verification)
        return self.run_async(generation, verification)

    def run_sequential(self, generation: Callable[[], Any], verification: Callable[[], Any]) -> tuple[Any, Any]:
        self.resources.start_generation()
        try:
            generated = generation()
        finally:
            self.resources.stop_generation()
        self.resources.start_verification()
        try:
            verified = verification()
        finally:
            self.resources.stop_verification()
        return generated, verified

    def run_generation_stage(self, callback: Callable[[], Any]) -> Any:
        self.resources.start_generation()
        try:
            return callback()
        finally:
            self.resources.stop_generation()

    def run_verification_stage(self, callback: Callable[[], Any], *, identity: str | None = None) -> Any:
        self.resources.start_verification(identity=identity)
        try:
            return callback()
        finally:
            self.resources.stop_verification()

    def run_pipeline(self, generation: Callable[[], Any], verification: Callable[[], Any]):
        """Reserved server seam; this release intentionally remains sequential."""
        return self.run_sequential(generation, verification)

    def run_async(self, generation: Callable[[], Any], verification: Callable[[], Any]):
        """Reserved GRPO seam; online reward execution is not implemented."""
        raise NotImplementedError("async generation/verification is reserved for future GRPO")
