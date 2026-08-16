import pytest

from lean_prover.Planner.client import (
    OpenAICompatibleClient,
    PlannerClientError,
    _extract_json_object,
)
from lean_prover.Planner.schemas import Blueprint, BlueprintPlan
from lean_prover.Planner.tests.helpers import environment


class FakeCompletions:
    def __init__(self, content: str | None | list[str | None]) -> None:
        self.content = content
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        self.kwargs = kwargs
        content = self.content.pop(0) if isinstance(self.content, list) else self.content
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeOpenAIClient:
    def __init__(self, content: str | None | list[str | None]) -> None:
        self.chat = type(
            "Chat",
            (),
            {"completions": FakeCompletions(content)},
        )()


def make_client(
    content: str | None | list[str | None],
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client._client = FakeOpenAIClient(content)
    client.model = "fake-model"
    client.temperature = 0.1
    client.max_tokens = 123
    return client


def test_generate_json_returns_parsed_object() -> None:
    client = make_client('{"ok": true}')

    result = client.generate_json(
        system_prompt="system",
        user_prompt="user",
    )

    assert result == {"ok": True}
    assert len(client._client.chat.completions.kwargs["messages"]) == 2
    assert client._client.chat.completions.kwargs["response_format"] == {
        "type": "json_object"
    }
    assert client._client.chat.completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_generate_json_rejects_invalid_json() -> None:
    client = make_client("not json")

    with pytest.raises(PlannerClientError, match="valid JSON"):
        client.generate_json(system_prompt="system", user_prompt="user")

    assert client._client.chat.completions.call_count == 5


def test_generate_json_recovers_decorated_json_object() -> None:
    client = make_client('```json\n{"ok": true}\n```')

    assert client.generate_json(system_prompt="system", user_prompt="user") == {
        "ok": True
    }
    assert _extract_json_object('Result: {"ok": true} done') == {"ok": True}


def test_generate_json_retries_empty_response_until_success() -> None:
    client = make_client([None, " ", None, "", '{"ok": true}'])

    result = client.generate_json(system_prompt="system", user_prompt="user")

    assert result == {"ok": True}
    assert client._client.chat.completions.call_count == 5


def test_generate_json_raises_after_five_empty_responses() -> None:
    client = make_client([None, " ", "", None, "  "])

    with pytest.raises(PlannerClientError, match="分解模型空响应"):
        client.generate_json(
            system_prompt="system",
            user_prompt="user",
            empty_response_message="分解模型空响应",
        )

    assert client._client.chat.completions.call_count == 5


def test_generate_blueprint_validates_response() -> None:
    client = make_client(
        """
        {
          "blueprint_summary": "demo",
          "nodes": [],
          "root_dependencies": [],
          "environment": {
            "lean_version": "Lean test",
            "lean_commit": "f72c35b3f637c8c6571d353742168ab66cc22c00",
            "mathlib_commit": "5e932f97dd25535344f80f9dd8da3aab83df0fe6",
            "environment_hash": "4444444444444444444444444444444444444444444444444444444444444444"
          }
        }
        """
    )

    blueprint = client.generate_blueprint(
        system_prompt="system",
        user_prompt="user",
    )

    assert isinstance(blueprint, Blueprint)
    assert blueprint.nodes == []


def test_generate_plan_validates_empty_lean_statements() -> None:
    client = make_client(
        """
        {
          "blueprint_summary": "direct",
          "nodes": [],
          "root_dependencies": []
        }
        """
    )

    plan = client.generate_plan(system_prompt="system", user_prompt="user")

    assert isinstance(plan, BlueprintPlan)
    assert plan.nodes == []
