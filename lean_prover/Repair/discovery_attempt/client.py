"""Observable API client for EI discovery-attempt proof repair."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .prompting import REPAIR_SYSTEM_PROMPT, build_repair_prompt
from .schema import APIObservation, RepairClientResponse, RepairRequest


class RepairClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: str = "api_error",
        observation: APIObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.observation = observation


@dataclass(frozen=True)
class RepairClientConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    response_format_json: bool = True

    @classmethod
    def from_env(cls) -> "RepairClientConfig":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RepairClientError("DEEPSEEK_API_KEY is not set", kind="configuration_error")
        return cls(
            api_key=key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", cls.base_url),
            model=os.environ.get("DEEPSEEK_MODEL", cls.model),
            temperature=float(os.environ.get("DEEPSEEK_REPAIR_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("DEEPSEEK_REPAIR_MAX_TOKENS", "4096")),
            timeout_seconds=float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "120")),
            response_format_json=os.environ.get("DEEPSEEK_JSON_RESPONSE", "1") != "0",
        )


def _usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if value is not None else None


class DeepSeekRepairClient:
    def __init__(self, config: RepairClientConfig | None = None) -> None:
        self.config = config or RepairClientConfig.from_env()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self.config.model

    def repair(self, request: RepairRequest) -> RepairClientResponse:
        started = time.monotonic()
        kwargs: dict[str, Any] = {}
        if self.config.response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_repair_prompt(request)},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs,
            )
        except Exception as error:
            raise RepairClientError(str(error), kind="api_error") from error

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        observation = APIObservation(
            provider="deepseek",
            base_url=self.config.base_url,
            model=str(getattr(response, "model", None) or self.config.model),
            request_id=str(getattr(response, "id", None) or ""),
            finish_reason=str(getattr(choice, "finish_reason", None) or ""),
            prompt_tokens=_usage_value(usage, "prompt_tokens"),
            completion_tokens=_usage_value(usage, "completion_tokens"),
            reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
            total_tokens=_usage_value(usage, "total_tokens"),
            latency_ms=round((time.monotonic() - started) * 1000),
            response_content_empty=not bool(content.strip()),
            reasoning_content_present=bool(str(reasoning_content or "").strip()),
        )
        if not content.strip():
            raise RepairClientError(
                "DeepSeek returned an empty response content field",
                kind="empty_response",
                observation=observation,
            )
        return RepairClientResponse(raw_content=content, observation=observation)
