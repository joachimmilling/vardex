from decimal import Decimal
from types import SimpleNamespace

import pytest

from vardex.config import Settings
from vardex.llm import MissingAPIKeyError, ask, count_tokens


class FakeMessages:
    """Stands in for client.messages, so tests never call the real API."""

    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            content=[
                SimpleNamespace(type="thinking", thinking=""),  # hidden thinking has no text
                SimpleNamespace(type="text", text="A nine-digit number that identifies…"),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=40,
                output_tokens=300,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                output_tokens_details=SimpleNamespace(thinking_tokens=250),
            ),
        )

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(input_tokens=17)


class FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.messages = FakeMessages(stop_reason)


SETTINGS = Settings(anthropic_api_key="test", model="claude-sonnet-5-5")


def test_ask_returns_the_text_and_what_it_cost():
    client = FakeClient()

    answer = ask("What is an organisation number?", SETTINGS, client=client)

    assert answer.text.startswith("A nine-digit number")
    assert answer.usage.input_tokens == 40
    assert answer.thinking_tokens == 250
    assert answer.cost_usd == Decimal("0.00308")  # 40 × $2 + 300 × $10, per million
    assert not answer.truncated
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["messages"] == [{"role": "user", "content": "What is an organisation number?"}]


def test_effort_is_sent_only_when_chosen():
    client = FakeClient()
    ask("q", SETTINGS, client=client)
    ask("q", SETTINGS, client=client, effort="low")
    assert "output_config" not in client.messages.calls[0]
    assert client.messages.calls[1]["output_config"] == {"effort": "low"}


def test_an_answer_cut_off_by_max_tokens_is_flagged():
    answer = ask("q", SETTINGS, client=FakeClient(stop_reason="max_tokens"))
    assert answer.truncated


def test_unknown_model_gives_an_answer_without_a_cost():
    settings = Settings(anthropic_api_key="test", model="some-future-model")
    answer = ask("q", settings, client=FakeClient())
    assert answer.cost_usd is None


def test_count_tokens_uses_the_configured_model():
    client = FakeClient()
    assert count_tokens("Hei", SETTINGS, client=client) == 17
    assert client.messages.calls[0]["model"] == "claude-sonnet-5-5"


def test_no_key_is_a_clear_error():
    with pytest.raises(MissingAPIKeyError):
        ask("q", Settings(anthropic_api_key=None, model="claude-sonnet-5-5"))
