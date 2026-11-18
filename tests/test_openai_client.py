import json
from decimal import Decimal
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from vardex.config import Settings
from vardex.llm import (
    Image,
    Message,
    MissingAPIKeyError,
    ModelError,
    Request,
    ToolCall,
    ToolReply,
    ToolSpec,
)
from vardex.openai_client import (
    LOCAL_TIMEOUT_SECONDS,
    OpenAIClient,
    connect,
    connect_ollama,
    strict_schema,
)

SETTINGS = Settings(model="ollama/gemma4:12b")
REQUEST = Request(system="Be brief.", messages=[Message("user", "What is an organisation number?")])


def usage(prompt=1300, completion=300, cached=1200, reasoning=250):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def reply(content="A nine-digit number.", calls=(), finish="stop", **extra):
    """A response in the shape the SDK returns."""
    message = SimpleNamespace(content=content, tool_calls=list(calls) or None, **extra)
    return SimpleNamespace(
        id="chatcmpl-1",
        model="gemma4:12b",
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=usage(),
    )


def chunk(content=None, calls=None, finish=None, usage_=None, reasoning=None):
    delta = SimpleNamespace(content=content, tool_calls=calls, reasoning=reasoning)
    choices = [SimpleNamespace(delta=delta, finish_reason=finish)] if not usage_ else []
    return SimpleNamespace(id="chatcmpl-2", model="gemma4:12b", choices=choices, usage=usage_)


class FakeCompletions:
    """Stands in for sdk.chat.completions and keeps the arguments of every call."""

    def __init__(self, response=None, error=None, chunks=None):
        self.calls = []
        self.response = response or reply()
        self.error = error
        self.chunks = chunks

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self.response


def client_with(free=True, **fake):
    completions = FakeCompletions(**fake)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClient(SETTINGS, "gemma4:12b", provider="ollama", free=free, sdk=sdk), completions


def test_an_answer_counts_cached_tokens_apart_and_a_local_model_costs_nothing():
    client, calls = client_with()
    answer = client.send(REQUEST)
    assert answer.text == "A nine-digit number."
    assert answer.stop_reason == "end"
    assert answer.usage.input_tokens == 100  # 1,300 in the prompt, 1,200 of them cached
    assert answer.usage.cache_read_tokens == 1200
    assert answer.thinking_tokens == 250
    assert answer.cost_usd == Decimal(0)
    sent = calls.calls[0]
    assert sent["messages"][0] == {"role": "system", "content": "Be brief."}
    assert sent["max_tokens"] == 4096  # the name Ollama reads; it ignores max_completion_tokens


def test_a_model_with_no_price_has_no_cost():
    client, _ = client_with(free=False)
    assert client.send(REQUEST).cost_usd is None


def test_the_tail_follows_the_system_prompt_and_effort_is_mapped():
    client, calls = client_with()
    client.send(Request(system="Rules.", system_tail="Examples.", messages=[], effort="max"))
    sent = calls.calls[0]
    assert sent["messages"][0]["content"] == "Rules.\n\nExamples."
    assert sent["reasoning_effort"] == "high"  # the most Chat Completions has


def test_a_schema_is_sent_strict_every_object_closed_and_every_field_required():
    class Plan(BaseModel):
        action: str
        sql: str | None = None

    client, calls = client_with()
    client.send(Request(system="", messages=[], output_schema=Plan))
    sent = calls.calls[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["strict"] is True
    schema = sent["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action", "sql"]
    assert strict_schema(Plan) == schema


def test_extra_reaches_the_server_untouched():
    client, calls = client_with()
    extra = {"structured_outputs": {"regex": r"\d{9}"}}
    client.send(Request(system="", messages=[], extra=extra))
    assert calls.calls[0]["extra_body"] == extra


def test_tools_calls_and_replies_go_as_chat_completions_wants_them():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="run_sql", arguments='{"sql": "select 1"}'),
    )
    client, calls = client_with(response=reply(content=None, calls=[call], finish="tool_calls"))
    tool = ToolSpec("run_sql", "Run a query.", {"type": "object", "properties": {}})
    answer = client.send(Request(system="", messages=[Message("user", "Hi")], tools=(tool,)))
    assert answer.stop_reason == "tool_use"
    assert answer.tool_calls == (ToolCall("call-1", "run_sql", {"sql": "select 1"}),)
    assert calls.calls[0]["tools"][0]["function"]["name"] == "run_sql"

    turn = answer.as_message()
    replies = Message("user", "", tool_replies=(ToolReply("call-1", "no such table", True),))
    client.send(Request(system="", messages=[Message("user", "Hi"), turn, replies]))
    sent = calls.calls[1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"sql": "select 1"})
    assert sent[2] == {"role": "tool", "tool_call_id": "call-1", "content": "Error: no such table"}


def test_a_picture_is_a_data_url_before_the_text():
    client, calls = client_with()
    page = Message("user", "Write out this page.", images=(Image(b"\x89PNG"),))
    client.send(Request(system="", messages=[page]))
    content = calls.calls[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
    assert content[1] == {"type": "text", "text": "Write out this page."}


def test_reasoning_is_kept_under_whichever_name_the_server_gives_it():
    client, _ = client_with(response=reply(reasoning_content="First, the register…"))
    assert client.send(REQUEST).reasoning[0].text == "First, the register…"


def test_a_streamed_answer_is_put_back_together_calls_included():
    pieces = [
        chunk(reasoning="Hmm."),
        chunk(content="A nine-"),
        chunk(content="digit number."),
        chunk(
            calls=[
                SimpleNamespace(
                    index=0, id="call-1", function=SimpleNamespace(name="run_sql", arguments='{"s')
                )
            ]
        ),
        chunk(
            calls=[
                SimpleNamespace(
                    index=0, id=None, function=SimpleNamespace(name=None, arguments='ql": 1}')
                )
            ],
            finish="tool_calls",
        ),
        chunk(usage_=usage()),
    ]
    client, calls = client_with(chunks=pieces)
    seen = []
    answer = client.send(REQUEST, on_text=seen.append)
    assert seen == ["A nine-", "digit number."]
    assert answer.text == "A nine-digit number."
    assert answer.tool_calls == (ToolCall("call-1", "run_sql", {"sql": 1}),)
    assert answer.reasoning[0].text == "Hmm."
    assert answer.usage.cache_read_tokens == 1200
    assert answer.first_text_seconds is not None
    assert calls.calls[0]["stream_options"] == {"include_usage": True}


def test_a_server_that_is_down_is_a_retryable_error_that_says_where():
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    client, _ = client_with(error=openai.APIConnectionError(request=request))
    client.where = "http://localhost:11434/v1"
    with pytest.raises(ModelError) as caught:
        client.send(REQUEST)
    assert caught.value.retryable
    assert "localhost:11434" in str(caught.value)


@pytest.mark.parametrize(("status", "retryable"), [(503, True), (429, True), (400, False)])
def test_a_refusal_says_whether_to_retry(status, retryable):
    response = httpx.Response(status, request=httpx.Request("POST", "http://x"))
    error = openai.APIStatusError("no", response=response, body=None)
    client, _ = client_with(error=error)
    with pytest.raises(ModelError) as caught:
        client.send(REQUEST)
    assert caught.value.retryable is retryable


def test_tokens_are_counted_by_asking_for_one():
    client, calls = client_with()
    assert client.count_tokens("Hei") == 1300
    assert calls.calls[0]["max_tokens"] == 1


def test_ollama_needs_no_key_and_is_found_at_its_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11500")
    client = connect_ollama(SETTINGS, "gemma4:12b")
    assert client.where == "http://0.0.0.0:11500/v1"
    assert client.provider == "ollama"
    assert client.free


@pytest.mark.parametrize(
    ("url", "timeout"),
    [
        ("http://127.0.0.1:8080/v1", LOCAL_TIMEOUT_SECONDS),
        ("http://localhost:1234/v1", LOCAL_TIMEOUT_SECONDS),
        ("https://api.together.xyz/v1", SETTINGS.timeout_seconds),
        (None, SETTINGS.timeout_seconds),
    ],
)
def test_a_server_on_this_machine_is_given_a_laptops_time(monkeypatch, url, timeout):
    monkeypatch.setenv("OPENAI_API_KEY", "local")
    if url:
        monkeypatch.setenv("OPENAI_BASE_URL", url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert connect(SETTINGS, "default_model").sdk.timeout == timeout


def test_openai_without_a_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        connect(SETTINGS, "gpt-oss-120b")
