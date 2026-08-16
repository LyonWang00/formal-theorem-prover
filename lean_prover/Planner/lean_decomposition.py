"""Dedicated API interface for Lean-native Blueprint decomposition."""

from __future__ import annotations

from typing import Any

from .client import LLMClient
from .prompts import (
    LEAN_DECOMPOSITION_SYSTEM_PROMPT,
    build_lean_decomposition_prompt,
)
from .schemas import Blueprint, LeanEnvironmentIdentity, TheoremProblem


class LeanDecompositionAPI:
    """Call the Lean-only decomposition contract through the configured API."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def decompose(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
    ) -> Blueprint:
        raw = self.decompose_candidate(
            problem=problem,
            environment=environment,
        )
        blueprint = Blueprint.model_validate(raw)
        if not blueprint.problem_hash:
            blueprint.problem_hash = problem.problem_hash
        return blueprint

    def decompose_candidate(
        self,
        *,
        problem: TheoremProblem,
        environment: LeanEnvironmentIdentity,
    ) -> dict[str, Any] | str:
        """Preserve malformed nonempty output for the repair boundary."""

        generate_raw = getattr(self.client, "generate_text_or_json", None)
        generate = generate_raw if callable(generate_raw) else self.client.generate_json
        return generate(
            system_prompt=LEAN_DECOMPOSITION_SYSTEM_PROMPT,
            user_prompt=build_lean_decomposition_prompt(problem, environment),
            empty_response_message="分解模型空响应",
        )


__all__ = ["LeanDecompositionAPI"]
