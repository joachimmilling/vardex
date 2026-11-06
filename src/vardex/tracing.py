"""Tracing: what one request did, as a tree of timed steps, in OpenTelemetry's terms.

A trace is one request: a question answered, an eval case, a scheduled run. A span is one
timed piece of it: a model call, a query, a tool, one attempt at passing the checks. Each span
knows its parent, so the pieces form a tree, and the tree answers the question a log cannot:
why did it say that? Every input, every output and every second, in the order they happened.

Vardex writes its spans with OpenTelemetry, so anything that speaks OTLP can show them:
Langfuse, Microsoft Foundry, Grafana, Jaeger, or the files Vardex keeps itself. The attribute
names follow OpenTelemetry's conventions for generative AI and for databases, so a viewer that
knows those conventions finds the model, the tokens and the query without being told where
to look. The names are written out below rather than imported: the conventions for generative
AI are still in development, and the names are the contract, not a library's constants.

This is the only module that imports opentelemetry; a test enforces it. Everything else calls
`span` and gets a plain `Span` back, so the engine depends on the idea of a trace and not on
one library's version of it.

Nothing is traced until `setup` is called, which the command-line tool does for each command.
An application that uses Vardex as a library and has set up OpenTelemetry itself gets
Vardex's spans inside its own traces, under its own requests.
"""

import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Link, SpanKind, Status, StatusCode

from vardex import __version__
from vardex.llm import Answer, Hook, Message, Request, Send

# OpenTelemetry's names for a model call, a tool and an agent
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
MAX_TOKENS = "gen_ai.request.max_tokens"
OUTPUT_TYPE = "gen_ai.output.type"
FINISH_REASONS = "gen_ai.response.finish_reasons"
FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
# Every input token, cached or not; the cache counts are parts of it, not additions to it
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens"
REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
AGENT_NAME = "gen_ai.agent.name"
WORKFLOW_NAME = "gen_ai.workflow.name"
RETRIEVAL_QUERY = "gen_ai.retrieval.query.text"
EVALUATION = "gen_ai.evaluation.result"  # an event: one check, one verdict
EVALUATION_NAME = "gen_ai.evaluation.name"
EVALUATION_LABEL = "gen_ai.evaluation.score.label"
EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
# OpenTelemetry's names for a query and an HTTP request
DB_SYSTEM = "db.system.name"
DB_OPERATION = "db.operation.name"
DB_QUERY = "db.query.text"
DB_ROWS = "db.response.returned_rows"
HTTP_METHOD = "http.request.method"
HTTP_STATUS = "http.response.status_code"
HTTP_RESEND = "http.request.resend_count"  # 0 on the first attempt, 1 on the first retry
SERVER = "server.address"
ERROR_TYPE = "error.type"
# Vardex's own, where the conventions have no name
COST = "vardex.cost_usd"
EFFORT = "vardex.effort"
PROMPT = "vardex.prompt.sha"  # which system prompt, so a changed one shows in the trace
COMMAND = "vardex.command"
PACK = "vardex.pack"
PACK_VERSION = "vardex.pack.version"
VERSION = "vardex.version"
QUESTION = "vardex.question"
ANSWER = "vardex.answer"
SQL = "vardex.sql"
OUTCOME = "vardex.outcome"  # answer, clarify, refuse, not_found, error …
ATTEMPT = "vardex.attempt"
PASSED = "vardex.checks.passed"
PROBLEMS = "vardex.checks.problems"
CASE = "vardex.case"
SUITE = "vardex.suite"

# What people asked and what the model wrote, as opposed to how long, how many and how much.
# These stay on the machine unless the setting says they may leave it: see `setup`.
CONTENT = frozenset(
    {
        SYSTEM_INSTRUCTIONS,
        INPUT_MESSAGES,
        OUTPUT_MESSAGES,
        TOOL_ARGUMENTS,
        TOOL_RESULT,
        RETRIEVAL_QUERY,
        DB_QUERY,
        QUESTION,
        ANSWER,
        SQL,
        PROBLEMS,
        EVALUATION_EXPLANATION,
    }
)

# Where content goes: into the local files only (the default), into every backend, or nowhere
Content = Literal["local", "all", "none"]
Kind = Literal["internal", "client"]
KINDS = {"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT}
KEEP_DAYS = 30  # trace files older than this are deleted when tracing starts
# How long sending one batch to a backend may take, retries included. The exporter's own
# default is ten seconds, and a command waits for its last batch before it exits: with the
# backend down, every answer would end with ten seconds of nothing.
EXPORT_SECONDS = 3.0

_provider: TracerProvider | None = None  # set by setup; None means use the application's own


class Span:
    """One span, as the rest of Vardex sees it: something to add facts to while it runs."""

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    @property
    def trace_id(self) -> str:
        """The trace's id as 32 hex digits, or "" when nothing is being traced."""
        context = self._span.get_span_context()
        return trace.format_trace_id(context.trace_id) if context.is_valid else ""

    @property
    def span_id(self) -> str:
        context = self._span.get_span_context()
        return trace.format_span_id(context.span_id) if context.is_valid else ""

    def set(self, name: str, value: Any) -> None:
        """Add one fact. None is left out, rather than written as a value."""
        if value is not None:
            self._span.set_attribute(name, attribute(value))

    def update(self, attributes: dict[str, Any]) -> None:
        for name, value in attributes.items():
            self.set(name, value)

    def event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Something that happened at one moment inside the span, such as a check's verdict."""
        self._span.add_event(name, clean(attributes))

    def fail(self, reason: str, error_type: str = "failed") -> None:
        """Mark the span as failed without an exception: a check that did not pass."""
        self._span.set_status(Status(StatusCode.ERROR, reason))
        self._span.set_attribute(ERROR_TYPE, error_type)


def attribute(value: Any) -> Any:
    """A value OpenTelemetry accepts: text, a number, a bool, or a list of one of those.
    Money stays exact as text; anything else becomes JSON."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
        return list(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def clean(attributes: dict[str, Any] | None) -> dict[str, Any]:
    return {k: attribute(v) for k, v in (attributes or {}).items() if v is not None}


def tracer() -> trace.Tracer:
    """Vardex's tracer: from the provider `setup` made, or else the application's own."""
    if _provider is not None:
        return _provider.get_tracer("vardex", __version__)
    return trace.get_tracer("vardex", __version__)


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    kind: Kind = "internal",
    new_trace: bool = False,
) -> Iterator[Span]:
    """A span around a block of work, as a child of the span around it.

    An exception that leaves the block marks the span as failed, with the exception kept on
    it. With `new_trace`, the span starts a trace of its own, linked to the one it was
    started in: an eval case is a request of its own, not one step of the eval.
    """
    parent = None
    links: list[Link] = []
    if new_trace:
        around = trace.get_current_span().get_span_context()
        links = [Link(around)] if around.is_valid else []
        parent = otel_context.Context()  # an empty context: no parent, so a new trace
    with tracer().start_as_current_span(
        name, context=parent, kind=KINDS[kind], attributes=clean(attributes), links=links
    ) as raw:
        try:
            yield Span(raw)
        except Exception as err:
            raw.set_attribute(ERROR_TYPE, type(err).__name__)
            raise


def current() -> Span:
    """The span the caller is running in, so a command can add what it learnt to its own."""
    return Span(trace.get_current_span())


def carry[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """The function, made to run inside the caller's trace when another thread runs it.

    The current span is kept in a context variable, and a thread pool does not copy the
    submitting thread's context into its workers. Without this, a tool call run in a pool
    starts a trace of its own, and the request's tree loses the branch that did the work.
    """
    captured = otel_context.get_current()

    def carried(*args: P.args, **kwargs: P.kwargs) -> R:
        token = otel_context.attach(captured)
        try:
            return function(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return carried


def setup(
    folder: Path | None,
    *,
    content: Content = "local",
    export: bool = False,
    resource: dict[str, str] | None = None,
    keep_days: int = KEEP_DAYS,
) -> None:
    """Start tracing for this process: into files in `folder`, and to an OTLP backend when
    `export` is set. The backend's address and credentials come from OpenTelemetry's own
    variables, OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS, which the
    exporter reads itself.

    The files are written as each span ends, so a crash loses nothing that finished. The
    backend is sent spans in batches from a background thread, so a slow backend never slows
    an answer down; `flush` sends what is waiting.
    """
    global _provider
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "vardex", "service.version": __version__, **(resource or {})}
        )
    )
    if folder is not None:
        prune(folder, keep_days)
        files: SpanExporter = FileExporter(folder)
        provider.add_span_processor(
            SimpleSpanProcessor(files if content != "none" else WithoutContent(files))
        )
    if export:
        # OTEL_EXPORTER_OTLP_TIMEOUT, when set, is the operator's choice and wins
        timeout = None if os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT") else EXPORT_SECONDS
        backend: SpanExporter = Delivery(OTLPSpanExporter(timeout=timeout))
        provider.add_span_processor(
            BatchSpanProcessor(backend if content == "all" else WithoutContent(backend))
        )
        # The exporter logs every retry to stderr, after the answer has been printed. A
        # backend that is down is said once, by `flush`, in words a person can act on.
        logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
    if _provider is not None:
        _provider.shutdown()
    _provider = provider


def flush(timeout_seconds: float = 2 * EXPORT_SECONDS) -> int:
    """Send every span that is still waiting to go, waiting at most this long, and return
    how many batches the backend has refused or never received since tracing started. The
    local files have every span either way."""
    if _provider is None:
        return 0
    _provider.force_flush(int(timeout_seconds * 1000))
    return Delivery.undelivered


def stop() -> None:
    """Stop tracing: flush, and go back to the application's own tracer."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
    _provider = None


def traces_path(data_dir: Path, pack: str | None) -> Path:
    """Where Vardex keeps a pack's traces, one file per day; commands without a pack share one."""
    return data_dir / pack / "traces" if pack else data_dir / "traces"


def day_file(folder: Path, day: date) -> Path:
    return folder / f"{day.isoformat()}.jsonl"


def prune(folder: Path, keep_days: int) -> None:
    """Delete trace files older than `keep_days`. Traces hold questions and answers, and
    keeping them for ever is a decision nobody made."""
    oldest = datetime.now(UTC).date() - timedelta(days=keep_days)
    for file in folder.glob("*.jsonl"):
        try:
            day = date.fromisoformat(file.stem)
        except ValueError:
            continue
        if day < oldest:
            file.unlink(missing_ok=True)


def timestamp(nanoseconds: int | None) -> str | None:
    if nanoseconds is None:
        return None
    moment = datetime.fromtimestamp(nanoseconds / 1e9, UTC)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def as_record(span: ReadableSpan) -> dict[str, Any]:
    """A finished span as one JSON object: the ids that put it in its tree, when it ran, how
    it ended, and every attribute and event."""
    context = span.get_span_context()
    status = span.status.status_code
    return {
        "trace_id": trace.format_trace_id(context.trace_id) if context else "",
        "span_id": trace.format_span_id(context.span_id) if context else "",
        "parent_id": trace.format_span_id(span.parent.span_id) if span.parent else None,
        "name": span.name,
        "kind": "client" if span.kind == SpanKind.CLIENT else "internal",
        "start": timestamp(span.start_time),
        "end": timestamp(span.end_time),
        "status": {StatusCode.ERROR: "error", StatusCode.OK: "ok"}.get(status, "unset"),
        "error": span.status.description or None,
        "attributes": dict(span.attributes or {}),
        "events": [
            {"name": e.name, "time": timestamp(e.timestamp), "attributes": dict(e.attributes or {})}
            for e in span.events
        ],
        "links": [trace.format_trace_id(link.context.trace_id) for link in span.links],
        # What made the span: the service, its version, and the pack and its version
        "resource": dict(span.resource.attributes) if span.resource else {},
    }


class FileExporter(SpanExporter):
    """Vardex's own trace store: each finished span as one JSON line, in a file per day."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.lock = threading.Lock()  # spans end in several threads; lines must not interleave

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        by_day: dict[date, list[str]] = {}
        for span in spans:
            ended = datetime.fromtimestamp((span.end_time or 0) / 1e9, UTC).date()
            line = json.dumps(as_record(span), ensure_ascii=False, default=str)
            by_day.setdefault(ended, []).append(line)
        try:
            with self.lock:
                self.folder.mkdir(parents=True, exist_ok=True)
                for day, lines in by_day.items():
                    with day_file(self.folder, day).open("a", encoding="utf-8") as file:
                        file.write("".join(f"{line}\n" for line in lines))
        except OSError:
            return SpanExportResult.FAILURE  # a trace that cannot be written never fails a request
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class Delivery(SpanExporter):
    """An exporter that passes spans on and counts the batches that did not arrive."""

    undelivered = 0  # for the whole process: there is one backend, however many setups

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self.exporter.export(spans)
        except Exception:  # a broken backend must never break the request it watches
            result = SpanExportResult.FAILURE
        if result != SpanExportResult.SUCCESS:
            Delivery.undelivered += 1
        return result

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)


class WithoutContent(SpanExporter):
    """An exporter that passes spans on with what people asked and what the model wrote taken
    out: the timings, tokens, costs and verdicts are left, and nothing that was said."""

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.exporter.export([without_content(span) for span in spans])

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)


def without_content(span: ReadableSpan) -> ReadableSpan:
    def keep(attributes: Any) -> dict[str, Any]:
        return {k: v for k, v in (attributes or {}).items() if k not in CONTENT}

    return ReadableSpan(
        name=span.name,
        context=span.get_span_context(),
        parent=span.parent,
        resource=span.resource,
        attributes=keep(span.attributes),
        events=[Event(e.name, keep(e.attributes), e.timestamp) for e in span.events],
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def fingerprint(text: str) -> str:
    """A short, stable name for a text, such as a system prompt."""
    return sha256(text.encode("utf-8")).hexdigest()[:12]


def message_parts(message: Message) -> list[dict[str, Any]]:
    """One message in OpenTelemetry's shape for chat history: a role and typed parts."""
    parts: list[dict[str, Any]] = []
    for image in message.images:  # how big the picture was, not the picture
        blob = {"type": "blob", "modality": "image", "mime_type": image.media_type}
        parts.append({**blob, "size": len(image.data)})
    if message.text:
        parts.append({"type": "text", "content": message.text})
    for call in message.tool_calls:
        parts.append(
            {"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments}
        )
    for reply in message.tool_replies:
        parts.append({"type": "tool_call_response", "id": reply.id, "response": reply.content})
    return parts


def request_attributes(provider: str, model: str, request: Request) -> dict[str, Any]:
    """What a model call span says about the request, before it is sent."""
    system = [request.system, request.system_tail] if request.system_tail else [request.system]
    return {
        OPERATION: "chat",
        PROVIDER: provider,
        REQUEST_MODEL: model,
        MAX_TOKENS: request.max_tokens,
        OUTPUT_TYPE: "json" if request.output_schema else "text",
        EFFORT: request.effort,
        PROMPT: fingerprint(request.system),
        SYSTEM_INSTRUCTIONS: [{"type": "text", "content": text} for text in system],
        INPUT_MESSAGES: [{"role": m.role, "parts": message_parts(m)} for m in request.messages],
    }


def answer_attributes(answer: Answer) -> dict[str, Any]:
    """What a model call span says about the answer, once it has come back."""
    usage = answer.usage
    return {
        RESPONSE_MODEL: answer.model,
        RESPONSE_ID: answer.id or None,
        FINISH_REASONS: [answer.stop_reason],
        INPUT_TOKENS: usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        OUTPUT_TOKENS: usage.output_tokens,
        CACHE_READ: usage.cache_read_tokens or None,
        CACHE_WRITE: usage.cache_write_tokens or None,
        REASONING_TOKENS: answer.thinking_tokens,
        FIRST_CHUNK: answer.first_text_seconds,
        COST: answer.cost_usd,
        OUTPUT_MESSAGES: [
            {
                "role": "assistant",
                "parts": message_parts(answer.as_message()),
                "finish_reason": answer.stop_reason,
            }
        ],
    }


def trace_calls(provider: str) -> Hook:
    """A hook that puts every model call in a span of its own, named as OpenTelemetry names
    one: chat, then the model. The provider is named by whoever chose it."""

    def hook(model: str, request: Request, send: Send) -> Answer:
        attributes = request_attributes(provider, model, request)
        with span(f"chat {model}", attributes, kind="client") as call:
            answer = send(request)
            call.update(answer_attributes(answer))
            return answer

    return hook


def evaluation(where: Span, name: str, passed: bool, explanation: str = "") -> None:
    """Record one check's verdict on a span, as OpenTelemetry's evaluation event, so a check
    that runs on every request is an evaluation of every request."""
    where.event(
        EVALUATION,
        {
            EVALUATION_NAME: name,
            EVALUATION_LABEL: "pass" if passed else "fail",
            EVALUATION_EXPLANATION: explanation or None,
        },
    )


def passed(attempt: Span, check: str) -> None:
    """Mark an attempt whose output passed a check."""
    attempt.set(PASSED, True)
    evaluation(attempt, check, True)


def failed(attempt: Span, check: str, problems: list[str]) -> None:
    """Mark an attempt whose output failed a check. The request may go on to try again, so
    the attempt's span fails and the trace around it need not."""
    attempt.update({PASSED: False, PROBLEMS: problems})
    attempt.fail("; ".join(problems), check)
    evaluation(attempt, check, False, "; ".join(problems))
