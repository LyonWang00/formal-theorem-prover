"""API client for Planner."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI
from pydantic import ValidationError

from lean_prover.model_api import MAX_EMPTY_RESPONSE_ATTEMPTS

from .schemas import Blueprint, BlueprintPlan


class PlannerClientError(RuntimeError):
    pass


def _extract_json_object(content: str) -> dict[str, Any] | None:
    """Recover one JSON object from harmless transport-level decoration."""

    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


@dataclass(frozen=True)
class PlannerClientConfig:
    """
    Configuration for the Planner API client.
    You only need to set the `api_key` in your environment variables.
    The other parameters have reasonable defaults, but you can override them if needed.
    """
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.2
    max_tokens: int = 8192

    @classmethod
    def from_env(cls) -> "PlannerClientConfig":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise PlannerClientError(
                "DEEPSEEK_API_KEY is not set"
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL",
                cls.base_url,
            ),
            model=os.environ.get("DEEPSEEK_MODEL", cls.model),
            temperature=float(
                os.environ.get(
                    "DEEPSEEK_TEMPERATURE",
                    str(cls.temperature),
                )
            ),
            max_tokens=int(
                os.environ.get(
                    "DEEPSEEK_MAX_TOKENS",
                    str(cls.max_tokens),
                )
            ),
        )

class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        ...

    def generate_text_or_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any] | str:
        ...

class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_text_or_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any] | str:
        """Preserve nonempty malformed JSON for the BluePrintRepair boundary."""

        content: str | None = None
        for _ in range(MAX_EMPTY_RESPONSE_ATTEMPTS):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *(history or []),
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                break
        else:
            raise PlannerClientError(empty_response_message)

        return _extract_json_object(content) or content

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        empty_response_message: str = "模型空响应",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        retry_history = list(history or [])
        for attempt in range(MAX_EMPTY_RESPONSE_ATTEMPTS):
            candidate = self.generate_text_or_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                empty_response_message=empty_response_message,
                history=retry_history,
            )
            if isinstance(candidate, dict):
                return candidate
            if attempt + 1 < MAX_EMPTY_RESPONSE_ATTEMPTS:
                retry_history.extend(
                    [
                        {"role": "assistant", "content": candidate},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not one valid JSON "
                                "object. Return only the requested JSON object; "
                                "preserve the requested keys and put no text "
                                "before or after it."
                            ),
                        },
                    ]
                )
        raise PlannerClientError(
            "LLM did not return valid JSON after "
            f"{MAX_EMPTY_RESPONSE_ATTEMPTS} attempts"
        )

    def generate_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> BlueprintPlan:
        """Generate and validate the natural-language decomposition stage."""

        raw = self.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            empty_response_message="分解模型空响应",
        )
        try:
            return BlueprintPlan.model_validate(raw)
        except ValidationError as error:
            raise PlannerClientError(
                f"BlueprintPlan schema validation failed: {error}"
            ) from error

    def generate_blueprint(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Blueprint:
        raw = self.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            empty_response_message="节点形式化模型空响应",
        )

        try:
            return Blueprint.model_validate(raw)
        except ValidationError as error:
            raise PlannerClientError(
                f"Blueprint schema validation failed: {error}"
            ) from error


def create_deepseek_client_from_env() -> OpenAICompatibleClient:
    config = PlannerClientConfig.from_env()
    return OpenAICompatibleClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
