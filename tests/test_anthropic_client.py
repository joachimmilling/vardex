from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from vardex.anthropic_client import AnthropicClient
from vardex.config import Settings
from vardex.llm import Message, MissingAPIKeyError, ModelError, Request

SETTINGS = Settings(anthropic_api_key="test", model="claude-sonnet-5-5")
REQUEST = Request(system="Be brief.", messages=[Message("user", "What is an organisation number?")])


def reply(**kwargs):
    """A response in the shape the SDK returns."""
    return SimpleNamespace(
        model=kwargs["model"],
        content=[
            SimpleNamespace(type="thinking", thinking=""),  # hidden thinking has no text
            SimpleNamespace(type="text", text="A nine-digit number that identifies…"),
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=40,
            output_tokens=300,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=1200,
            output_tokens_details=SimpleNamespace(thinking_tokens=250),
        ),
    )


class FakeStream:
    """Stands in for the SDK's stream: a context manager that yields text."""

    def __init__(self, message):
        self.message = message
        self.text_stream = iter(["A nine-digit ", "number that identifies…"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    """Stands in for sdk.messages and keeps the arguments of every call."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return reply(**kwargs)

    def stream(self, **kwargs):
        return FakeStream(self.create(**kwargs))

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(input_tokens=17)


def client_with(settings=SETTINGS, error=None):
    messages = FakeMessages(error)
    return AnthropicClient(settings, sdk=SimpleNamespace(messages=messages)), messages


def test_send_returns_the_text_and_what_it_cost():
    client, messages = client_with()

    answer = client.send(REQUEST)

    assert answer.text.startswith("A nine-digit number")
    assert answer.thinking_tokens == 250
    assert answer.usage.cache_read_tokens == 1200
    assert answer.cost_usd == Decimal("0.00332")  # 40 × $2 + 300 × $10 + 1,200 × $0.20
    call = messages.calls[0]
    assert call["system"] == "Be brief."
    assert call["messages"] == [{"role": "user", "content": "What is an organisation number?"}]
    assert "output_config" not in call


def test_a_streamed_answer_passes_each_piece_on():
    client, _ = client_with()
    pieces = []
    answer = client.send(REQUEST, on_text=pieces.append)
    assert answer.text == "".join(pieces)
    assert answer.first_text_seconds <= answer.seconds


class Figures(BaseModel):
    revenue: int | None


@pytest.mark.parametrize(
    ("anthropic_reason", "vardex_reason"),
    [
        ("end_turn", "end"),
        ("max_tokens", "max_tokens"),
        ("tool_use", "tool_use"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),  # anything Vardex does not handle is "other"
    ],
)
def test_the_stop_reason_is_put_in_vardexs_own_words(anthropic_reason, vardex_reason):
    client, _ = client_with()
    response = reply(model="claude-sonnet-5-5")
    response.stop_reason = anthropic_reason

    answer = client.to_answer(response, seconds=1.0, first_text_seconds=None)

    assert answer.stop_reason == vardex_reason


def test_effort_schema_and_cache_go_into_the_request():
    client, messages = client_with()
    request = Request(
        system="Extract.",
        messages=[Message("user", "…")],
        effort="low",
        output_schema=Figures,
        cache_system=True,
    )

    client.send(request)

    call = messages.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["effort"] == "low"
    schema = call["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False  # structured outputs require it
    assert schema["required"] == ["revenue"]


def test_a_cached_conversation_turns_on_automatic_caching():
    client, messages = client_with()
    client.send(REQUEST)
    client.send(replace(REQUEST, cache_conversation=True))
    assert "cache_control" not in messages.calls[0]
    assert messages.calls[1]["cache_control"] == {"type": "ephemeral"}


def test_effort_is_left_out_for_a_model_that_rejects_it():
    client, messages = client_with(Settings(anthropic_api_key="test", model="claude-haiku-4-5"))
    client.send(Request(system="s", messages=[Message("user", "q")], effort="low"))
    assert "output_config" not in messages.calls[0]


def test_api_errors_become_model_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request)
    error = anthropic.RateLimitError("Rate limited", response=response, body=None)
    client, _ = client_with(error=error)
    with pytest.raises(ModelError, match="429"):
        client.send(REQUEST)


def test_a_lost_connection_becomes_a_model_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = client_with(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(ModelError, match="Could not reach the API"):
        client.send(REQUEST)


def test_count_tokens_uses_the_configured_model():
    client, messages = client_with()
    assert client.count_tokens("Hei") == 17
    assert messages.calls[0]["model"] == "claude-sonnet-5-5"


def test_no_key_is_a_clear_error():
    with pytest.raises(MissingAPIKeyError):
        AnthropicClient(Settings(anthropic_api_key=None, model="claude-sonnet-5-5"))
