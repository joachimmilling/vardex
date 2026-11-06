from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from vardex import tracing
from vardex.anthropic_client import PROVIDER, AnthropicClient, trace_attempts
from vardex.config import Settings
from vardex.llm import Hooked, Image, Message, MissingAPIKeyError, ModelError, Request
from vardex.traces import FileTraces, traces

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
    assert answer.cost_usd == Decimal("0.0032")  # 40 × $2 + 300 × $10 + 1,200 × $0.10
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


def test_a_conversation_is_cached_up_to_the_last_message():
    client, messages = client_with()
    client.send(Request(system="s", messages=[Message("user", "q")], cache_conversation=True))
    assert messages.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_effort_is_left_out_for_a_model_that_rejects_it():
    client, messages = client_with(Settings(anthropic_api_key="test", model="claude-haiku-4-5"))
    client.send(Request(system="s", messages=[Message("user", "q")], effort="low"))
    assert "output_config" not in messages.calls[0]


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (401, False), (429, True), (500, True), (529, True)]
)
def test_api_errors_become_model_errors_that_say_whether_to_try_again(status, retryable):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    error = anthropic.APIStatusError("Refused", response=response, body=None)
    client, _ = client_with(error=error)
    with pytest.raises(ModelError, match=str(status)) as caught:
        client.send(REQUEST)
    assert caught.value.retryable is retryable


def test_a_lost_connection_becomes_a_model_error_worth_trying_again():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = client_with(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(ModelError, match="Could not reach the API") as caught:
        client.send(REQUEST)
    assert caught.value.retryable


def test_count_tokens_uses_the_configured_model():
    client, messages = client_with()
    assert client.count_tokens("Hei") == 17
    assert messages.calls[0]["model"] == "claude-sonnet-5-5"


def test_no_key_is_a_clear_error():
    with pytest.raises(MissingAPIKeyError):
        AnthropicClient(Settings(anthropic_api_key=None, model="claude-sonnet-5-5"))


def test_a_picture_is_sent_before_the_text_as_an_image_block():
    client, messages = client_with()
    page = Message("user", "Write out this page.", images=(Image(b"\x89PNG\r\n"),))
    client.send(Request(system="Be exact.", messages=[page]))
    content = messages.calls[0]["messages"][0]["content"]
    assert content == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"},
        },
        {"type": "text", "text": "Write out this page."},
    ]


def test_a_system_tail_comes_after_the_cache_breakpoint():
    client, messages = client_with()
    request = Request(
        system="Tables.", messages=REQUEST.messages, cache_system=True, system_tail="Examples."
    )
    client.send(request)
    assert messages.calls[0]["system"] == [
        {"type": "text", "text": "Tables.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Examples."},
    ]


def test_each_http_attempt_is_a_span_inside_the_call(tmp_path):
    """The SDK retries an overloaded API itself. Each attempt gets its own span, numbered,
    inside the model call's span, so a slow call shows whether it was slow or retried."""
    statuses = iter([529, 200])
    message = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "Ni siffer."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }

    def handler(request):
        status = next(statuses)
        if status != 200:  # retry-after-ms keeps the test fast; the SDK honours it
            error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
            return httpx2.Response(status, json=error, headers={"retry-after-ms": "1"})
        return httpx2.Response(200, json=message, headers={"request-id": "req_01"})

    sdk = anthropic.Anthropic(
        api_key="test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        middleware=[trace_attempts],
        max_retries=2,
    )
    tracing.setup(tmp_path / "traces")
    client = Hooked(AnthropicClient(SETTINGS, sdk=sdk), (tracing.trace_calls(PROVIDER),))
    answer = client.send(REQUEST)
    assert answer.id == "msg_01"

    (trace,) = traces(FileTraces(tmp_path / "traces"), datetime.now(UTC) - timedelta(hours=1))
    chat = trace.root
    attempts = trace.children(chat)
    assert chat.name == "chat claude-sonnet-5-5" and chat.get("gen_ai.response.id") == "msg_01"
    tried = [
        (a.get("http.request.resend_count"), a.get("http.response.status_code")) for a in attempts
    ]
    assert tried == [(0, 529), (1, 200)]
    assert attempts[0].failed and not attempts[1].failed
    assert attempts[1].get("anthropic.request.id") == "req_01"
