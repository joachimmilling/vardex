"""Reading traces back: from Vardex's own files, or from the backend a company keeps them in.

Writing a trace is half of the job. The other half is reading it again: to find the request
behind a bad answer, to add up what a pack costs, and to turn what people flagged into cases
for the golden set. That half must work wherever the traces are kept, so it goes through a
TraceStore, and nothing that reads traces knows which one it has.

A store gives back Vardex's own `SpanRecord`, built from OpenTelemetry's attribute names,
which every backend keeps as it was sent. A backend's own vocabulary — its observations,
its levels, its idea of a generation — is translated in its store and goes no further.
"""

import base64
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import httpx2

from vardex.tracing import (
    ANSWER,
    ATTEMPT,
    CACHE_READ,
    CACHE_WRITE,
    COMMAND,
    COST,
    EVALUATION,
    EVALUATION_LABEL,
    EVALUATION_NAME,
    INPUT_TOKENS,
    OPERATION,
    OUTCOME,
    OUTPUT_TOKENS,
    PACK,
    PACK_VERSION,
    QUESTION,
    REQUEST_MODEL,
    SQL,
    day_file,
)


class TraceError(Exception):
    """Raised when traces cannot be read, or a trace id names none or more than one."""


@dataclass(frozen=True)
class SpanRecord:
    """One span as it was written: its place in the tree, its times, how it ended, and its
    attributes under OpenTelemetry's names."""

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: datetime
    end: datetime
    status: str  # ok, error or unset
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    resource: dict[str, Any] = field(default_factory=dict)  # what made it: service, pack

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def failed(self) -> bool:
        return self.status == "error"

    def get(self, name: str, default: Any = None) -> Any:
        """An attribute of the span, or else of whatever made it."""
        return self.attributes.get(name, self.resource.get(name, default))

    def verdicts(self) -> list[tuple[str, bool]]:
        """Each check recorded on this span, and whether it passed."""
        return [
            (
                e["attributes"].get(EVALUATION_NAME, ""),
                e["attributes"].get(EVALUATION_LABEL) == "pass",
            )
            for e in self.events
            if e["name"] == EVALUATION
        ]


@dataclass(frozen=True)
class Trace:
    """Every span of one request, in the order they started."""

    spans: list[SpanRecord]

    @property
    def root(self) -> SpanRecord:
        """The span the others hang from: the one whose parent is not in the trace."""
        ids = {s.span_id for s in self.spans}
        return next(s for s in self.spans if s.parent_id is None or s.parent_id not in ids)

    @property
    def trace_id(self) -> str:
        return self.spans[0].trace_id

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def start(self) -> datetime:
        return self.root.start

    @property
    def seconds(self) -> float:
        return self.root.seconds

    @property
    def failed(self) -> bool:
        return self.root.failed

    @property
    def calls(self) -> list[SpanRecord]:
        """The model calls."""
        return [s for s in self.spans if s.get(OPERATION) == "chat"]

    @property
    def cost_usd(self) -> Decimal:
        """What the model calls cost. A call with no price counts as nothing, and `unpriced`
        says how many there were, so a total that is too low says so."""
        return sum((Decimal(str(s.get(COST))) for s in self.calls if s.get(COST)), Decimal(0))

    @property
    def unpriced(self) -> int:
        return sum(1 for s in self.calls if s.get(COST) is None)

    @property
    def input_tokens(self) -> int:
        return sum(int(s.get(INPUT_TOKENS, 0)) for s in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(int(s.get(OUTPUT_TOKENS, 0)) for s in self.calls)

    @property
    def repaired(self) -> bool:
        """True when some step needed a second attempt to pass its checks."""
        return any(int(s.get(ATTEMPT, 1)) > 1 for s in self.spans)

    @property
    def command(self) -> str:
        return str(self.root.get(COMMAND) or self.root.name)

    @property
    def pack(self) -> str | None:
        return self.root.get(PACK)

    @property
    def pack_version(self) -> str | None:
        return self.root.get(PACK_VERSION)

    @property
    def outcome(self) -> str | None:
        return self.root.get(OUTCOME)

    @property
    def question(self) -> str | None:
        return self.root.get(QUESTION)

    @property
    def answer(self) -> str | None:
        return self.root.get(ANSWER)

    @property
    def sql(self) -> str | None:
        return self.root.get(SQL)

    @property
    def model(self) -> str | None:
        models = [
            s.get("gen_ai.response.model") or s.get("gen_ai.request.model") for s in self.calls
        ]
        return next((m for m in models if m), None)

    def children(self, span: SpanRecord) -> list[SpanRecord]:
        return [s for s in self.spans if s.parent_id == span.span_id]

    def walk(self) -> Iterator[tuple[int, SpanRecord]]:
        """Every span with its depth, parents before children, siblings in order."""

        def down(span: SpanRecord, depth: int) -> Iterator[tuple[int, SpanRecord]]:
            yield depth, span
            for child in self.children(span):
                yield from down(child, depth + 1)

        yield from down(self.root, 0)


def as_traces(spans: list[SpanRecord]) -> list[Trace]:
    """Spans grouped into their traces, oldest first, each trace's spans in starting order."""
    groups: dict[str, list[SpanRecord]] = {}
    for span in spans:
        groups.setdefault(span.trace_id, []).append(span)
    traces = [Trace(sorted(group, key=lambda s: s.start)) for group in groups.values()]
    return sorted(traces, key=lambda t: t.start)


class TraceStore(Protocol):
    """Where traces are read back from. Any class with these members will do."""

    name: str

    def spans(self, since: datetime, until: datetime | None = None) -> list[SpanRecord]:
        """Every span that ended in the window, from every trace in it."""
        ...

    def trace(self, trace_id: str) -> Trace:
        """One trace, by its id or the first characters of it."""
        ...


def traces(store: TraceStore, since: datetime, until: datetime | None = None) -> list[Trace]:
    """The traces of every request in a window, oldest first."""
    return as_traces(store.spans(since, until))


def moment(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def from_record(record: dict[str, Any]) -> SpanRecord:
    """A span as Vardex writes it to a file: see tracing.as_record."""
    return SpanRecord(
        trace_id=record["trace_id"],
        span_id=record["span_id"],
        parent_id=record.get("parent_id"),
        name=record["name"],
        start=moment(record["start"]),
        end=moment(record["end"]),
        status=record.get("status", "unset"),
        error=record.get("error"),
        attributes=record.get("attributes") or {},
        events=record.get("events") or [],
        resource=record.get("resource") or {},
    )


@dataclass
class FileTraces:
    """The traces Vardex keeps itself: a folder of JSON lines, one file per day."""

    folder: Path

    @property
    def name(self) -> str:
        return f"files in {self.folder}"

    def spans(self, since: datetime, until: datetime | None = None) -> list[SpanRecord]:
        until = until or datetime.now(UTC)
        found = []
        day = since.date()
        while day <= until.date():
            found += [s for s in self.read(day_file(self.folder, day)) if since <= s.end <= until]
            day += timedelta(days=1)
        return found

    def read(self, path: Path) -> list[SpanRecord]:
        if not path.is_file():
            return []
        spans = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                spans.append(from_record(json.loads(line)))
            except (ValueError, KeyError):
                continue  # a line cut short by a crash is lost; the rest of the file is not
        return spans

    def trace(self, trace_id: str) -> Trace:
        spans = [
            span
            for path in sorted(self.folder.glob("*.jsonl"))
            for span in self.read(path)
            if span.trace_id.startswith(trace_id)
        ]
        return one_trace(spans, trace_id)


def one_trace(spans: list[SpanRecord], trace_id: str) -> Trace:
    """The single trace these spans make up, or a TraceError that says what went wrong."""
    if not trace_id or not spans:
        raise TraceError(f"no trace {trace_id}")
    ids = sorted({s.trace_id for s in spans})
    if len(ids) > 1:
        raise TraceError(f"{trace_id} names {len(ids)} traces: give more of the id")
    return as_traces(spans)[0]


PAGE = 1000  # the most observations Langfuse gives back in one page
TRACE_ID_LENGTH = 32  # hex digits in a whole OpenTelemetry trace id
SEARCH_DAYS = 30  # how far back a trace named by part of its id is looked for
# Where Langfuse moved an attribute it understood, out of the attributes and into a field of
# its own: the field, its key, and the name the attribute was sent under
MOVED = (
    ("usageDetails", "input", INPUT_TOKENS),
    ("usageDetails", "output", OUTPUT_TOKENS),
    ("usageDetails", "cache_read_input_tokens", CACHE_READ),
    ("usageDetails", "cache_creation_input_tokens", CACHE_WRITE),
)


@dataclass
class LangfuseTraces:
    """Traces kept in Langfuse, read back through its public API.

    Langfuse calls a span an observation. It keeps the attributes it does not understand in
    the observation's metadata, and moves the ones it does — the model, the token counts —
    into fields of its own. Reading a trace back means putting those back under the names
    they were sent with, so the rest of Vardex reads a Langfuse trace exactly as it reads
    one of its own files. That translation happens here and nowhere else.
    """

    host: str  # such as https://cloud.langfuse.com
    public_key: str
    secret_key: str
    http: Callable[[], httpx2.Client] | None = None  # for tests: a client on a mock transport

    @property
    def name(self) -> str:
        return f"Langfuse at {self.host}"

    def client(self) -> httpx2.Client:
        if self.http is not None:
            return self.http()
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        return httpx2.Client(
            base_url=self.host.rstrip("/"),
            headers={"Authorization": f"Basic {token}"},
            timeout=30.0,
        )

    def observations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Every observation matching the filters, following the cursor page by page."""
        query = {
            **params,
            "limit": PAGE,
            "fields": "core,basic,metadata,model,usage",
            # Metadata is cut to 200 characters unless asked for in full, and the attributes
            # are the part of it this reads
            "expandMetadata": "attributes",
        }
        found: list[dict[str, Any]] = []
        with self.client() as client:
            while True:
                try:
                    response = client.get("/api/public/v2/observations", params=query)
                    response.raise_for_status()
                    body = response.json()
                except (httpx2.HTTPError, ValueError) as err:
                    raise TraceError(f"Langfuse did not answer: {err}") from err
                found += body.get("data", [])
                cursor = (body.get("meta") or {}).get("cursor")
                if not cursor:
                    return found
                query["cursor"] = cursor

    def spans(self, since: datetime, until: datetime | None = None) -> list[SpanRecord]:
        params = {"fromStartTime": iso(since)}
        if until is not None:
            params["toStartTime"] = iso(until)
        return [span for o in self.observations(params) if (span := from_observation(o))]

    def trace(self, trace_id: str) -> Trace:
        """One trace, by its id or the first characters of it. Langfuse finds a trace only by
        its whole id, so a part is looked for among the traces of the last SEARCH_DAYS days."""
        if len(trace_id) < TRACE_ID_LENGTH:
            since = datetime.now(UTC) - timedelta(days=SEARCH_DAYS)
            found = [s for s in self.spans(since) if s.trace_id.startswith(trace_id)]
            return one_trace(found, trace_id)
        observations = self.observations({"traceId": trace_id})
        return one_trace([s for o in observations if (s := from_observation(o))], trace_id)


def iso(when: datetime) -> str:
    return when.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def from_observation(observation: dict[str, Any]) -> SpanRecord | None:
    """One Langfuse observation as the span it was sent as, or None while it has no end."""
    if not observation.get("endTime"):
        return None
    metadata = observation.get("metadata") or {}
    attributes = kept(metadata, "attributes")
    resource = kept(metadata, "resourceAttributes")
    for field_name, key, name in MOVED:
        value = (observation.get(field_name) or {}).get(key)
        if value is not None and name not in attributes:
            attributes[name] = value
    if observation.get("model") and REQUEST_MODEL not in attributes:
        attributes[REQUEST_MODEL] = observation["model"]
    if observation.get("type") == "GENERATION":
        attributes.setdefault(OPERATION, "chat")
    return SpanRecord(
        trace_id=observation["traceId"],
        span_id=observation["id"],
        parent_id=observation.get("parentObservationId"),
        name=observation.get("name") or "",
        start=moment(observation["startTime"]),
        end=moment(observation["endTime"]),
        status="error" if observation.get("level") == "ERROR" else "unset",
        error=observation.get("statusMessage"),
        attributes=attributes,
        resource=resource,
        events=[],  # verdicts on a span are in Langfuse's scores, which this does not read
    )


def kept(metadata: dict[str, Any], part: str) -> dict[str, Any]:
    """The attributes Langfuse kept in an observation's metadata under `part`, as sent.

    Langfuse may keep them nested, as `{"attributes": {"vardex.pack": …}}`, or flattened into
    the metadata, as `{"attributes.vardex.pack": …}`; its cloud flattens them. Both are read.
    """
    nested = metadata.get(part) or {}
    if isinstance(nested, str):
        nested = json.loads(nested)
    prefix = f"{part}."
    flat = {k.removeprefix(prefix): v for k, v in metadata.items() if k.startswith(prefix)}
    return {name: as_sent(value) for name, value in {**nested, **flat}.items()}


def as_sent(value: Any) -> Any:
    """An attribute as it was sent. Langfuse may keep a number or a list as its JSON text."""
    if (
        isinstance(value, str)
        and value[:1] in ("[", "{")
        or (isinstance(value, str) and value.lstrip("-").replace(".", "", 1).isdigit())
    ):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
