"""Tracing: spans that nest, cross threads, keep content at home, and land in files."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fakes import FakeModel

from vardex import tracing
from vardex.config import load_settings
from vardex.llm import Answer, Hooked, Message, ModelError, Request
from vardex.pricing import Usage
from vardex.traces import FileTraces


@pytest.fixture
def folder(tmp_path):
    """Trace into files in a folder, and read them back with `spans(folder)`."""
    path = tmp_path / "traces"
    tracing.setup(path)
    return path


def spans(folder):
    return FileTraces(folder).spans(datetime.now(UTC) - timedelta(hours=1))


def by_name(folder):
    return {span.name: span for span in spans(folder)}


def test_nothing_is_traced_before_setup(tmp_path):
    with tracing.span("work") as work:
        assert work.trace_id == ""  # the default tracer records nothing and costs nothing
    assert not (tmp_path / "traces").exists()


def test_a_span_inside_another_is_its_child(folder):
    with tracing.span("request") as outer, tracing.span("step") as inner:
        assert inner.trace_id == outer.trace_id
    found = by_name(folder)
    assert found["step"].parent_id == found["request"].span_id
    assert found["request"].parent_id is None


def test_a_new_trace_has_no_parent_and_links_back(folder):
    with tracing.span("eval") as run, tracing.span("case", new_trace=True) as case:
        assert case.trace_id != run.trace_id
    found = by_name(folder)
    assert found["case"].parent_id is None


def test_an_exception_fails_the_span_and_names_its_type(folder):
    with pytest.raises(ValueError), tracing.span("work"):
        raise ValueError("bad input")
    work = by_name(folder)["work"]
    assert work.failed
    assert work.get("error.type") == "ValueError"


def test_a_thread_pool_loses_the_trace_unless_the_work_is_carried(folder):
    def step(name):
        with tracing.span(name) as span:
            return span.trace_id

    with tracing.span("request") as request, ThreadPoolExecutor(2) as pool:
        lost = list(pool.map(step, ["lost-1", "lost-2"]))
        carried = list(pool.map(tracing.carry(step), ["carried-1", "carried-2"]))
    assert request.trace_id not in lost  # each started a trace of its own
    assert carried == [request.trace_id, request.trace_id]
    found = by_name(folder)
    assert found["carried-1"].parent_id == found["request"].span_id


def test_attributes_keep_money_exact_and_drop_nothing_but_none(folder):
    with tracing.span("work") as work:
        work.update({"cost": Decimal("0.0042"), "rows": 3, "gone": None, "names": ["a", "b"]})
        work.set("parts", [{"type": "text"}])
    work = by_name(folder)["work"]
    assert work.get("cost") == "0.0042"
    assert work.get("rows") == 3
    assert "gone" not in work.attributes
    assert work.get("names") == ["a", "b"]
    assert json.loads(work.get("parts")) == [{"type": "text"}]


def test_content_stays_in_the_files_and_never_leaves_without_consent(tmp_path):
    record = tmp_path / "local"
    tracing.setup(record, content="local")
    with tracing.span("work", {tracing.QUESTION: "Hvor mange ansatte?", "vardex.rows": 3}):
        pass
    kept = by_name(record)["work"]
    assert kept.get(tracing.QUESTION) == "Hvor mange ansatte?"

    stripped = tracing.without_content(_one_span())
    assert tracing.QUESTION not in stripped.attributes
    assert stripped.attributes["vardex.rows"] == 3


def test_content_none_keeps_no_content_even_in_the_files(tmp_path):
    tracing.setup(tmp_path / "none", content="none")
    with tracing.span("work", {tracing.ANSWER: "120", "vardex.rows": 3}):
        pass
    work = by_name(tmp_path / "none")["work"]
    assert tracing.ANSWER not in work.attributes
    assert work.get("vardex.rows") == 3


def _one_span():
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    with provider.get_tracer("t").start_as_current_span(
        "work", attributes={tracing.QUESTION: "secret", "vardex.rows": 3}
    ) as span:
        pass
    return span


def test_old_trace_files_are_deleted_when_tracing_starts(tmp_path):
    folder = tmp_path / "traces"
    folder.mkdir()
    old = datetime.now(UTC).date() - timedelta(days=40)
    (folder / f"{old.isoformat()}.jsonl").write_text("{}\n")
    (folder / "notes.jsonl").write_text("not a day\n")
    tracing.setup(folder, keep_days=30)
    assert sorted(p.name for p in folder.iterdir()) == ["notes.jsonl"]


def test_hooks_run_in_order_each_around_the_next():
    order = []

    def hook(name):
        def around(model, request, send):
            order.append(f"{name} before")
            answer = send(request)
            order.append(f"{name} after")
            return answer

        return around

    client = Hooked(FakeModel("hi"), (hook("outer"), hook("inner")))
    client.send(Request(system="s", messages=[Message("user", "q")]))
    assert order == ["outer before", "inner before", "inner after", "outer after"]


def test_a_hook_can_refuse_a_call_before_it_goes():
    fake = FakeModel("hi")

    def refuse(model, request, send):
        raise ModelError("not today")

    with pytest.raises(ModelError):
        Hooked(fake, (refuse,)).send(Request(system="s", messages=[Message("user", "q")]))
    assert fake.requests == []


def test_a_model_call_is_a_chat_span_in_opentelemetrys_words(folder):
    class Cached(FakeModel):
        def send(self, request, on_text=None):
            answer = super().send(request, on_text)
            usage = Usage(input_tokens=200, output_tokens=50, cache_read_tokens=6000)
            return Answer(**{**answer.__dict__, "usage": usage, "id": "msg_01"})

    client = Hooked(Cached('{"ok": true}'), (tracing.trace_calls("anthropic"),))
    request = Request(system="You write SQL.", messages=[Message("user", "Hvor mange?")])
    client.send(request)
    chat = by_name(folder)["chat claude-sonnet-5-5"]
    assert chat.get("gen_ai.operation.name") == "chat"
    assert chat.get("gen_ai.provider.name") == "anthropic"
    assert chat.get("gen_ai.response.id") == "msg_01"
    # Every input token, the cached ones included; the cache count is a part of it
    assert chat.get("gen_ai.usage.input_tokens") == 6200
    assert chat.get("gen_ai.usage.cache_read.input_tokens") == 6000
    assert chat.get("vardex.cost_usd") == "0.004"
    assert chat.get("vardex.prompt.sha") == tracing.fingerprint("You write SQL.")
    messages = json.loads(chat.get("gen_ai.input.messages"))
    assert messages == [{"role": "user", "parts": [{"type": "text", "content": "Hvor mange?"}]}]


def test_a_failed_call_is_a_failed_span(folder):
    class Down(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("overloaded", retryable=True)

    client = Hooked(Down(), (tracing.trace_calls("anthropic"),))
    with pytest.raises(ModelError):
        client.send(Request(system="s", messages=[Message("user", "q")]))
    chat = by_name(folder)["chat claude-sonnet-5-5"]
    assert chat.failed
    assert chat.get("error.type") == "ModelError"


def test_a_check_is_recorded_as_an_evaluation_event(folder):
    with tracing.span("write_sql", {tracing.ATTEMPT: 1}) as attempt:
        tracing.failed(attempt, "sql_checks", ["no such column: revnue"])
    with tracing.span("write_sql", {tracing.ATTEMPT: 2}) as attempt:
        tracing.passed(attempt, "sql_checks")
    first, second = sorted(spans(folder), key=lambda s: s.get(tracing.ATTEMPT))
    assert first.failed and first.verdicts() == [("sql_checks", False)]
    assert first.get(tracing.PROBLEMS) == ["no such column: revnue"]
    assert not second.failed and second.verdicts() == [("sql_checks", True)]


def test_a_backend_that_is_down_is_counted_and_never_breaks_the_request():
    from opentelemetry.sdk.trace.export import SpanExportResult

    class Down:
        def export(self, spans):
            raise ConnectionError("refused")

        def shutdown(self):
            pass

    before = tracing.Delivery.undelivered
    assert tracing.Delivery(Down()).export([]) == SpanExportResult.FAILURE
    assert tracing.Delivery.undelivered == before + 1


def test_no_test_sends_its_traces_to_a_backend_or_reads_from_one():
    settings = load_settings()  # loads the developer's .env, which may name a real backend
    assert not settings.export_traces
    assert settings.trace_store == "files"
