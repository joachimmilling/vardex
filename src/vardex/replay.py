"""Record and replay: a run's model answers, query results and searches, saved and played back.

A recording is taken at Vardex's own interfaces: ModelClient and Warehouse, for documents
Embedder, Reranker and Index, and for scheduled work the change feed. It holds Vardex's words
only: no API key, no provider's wire format, no database driver. A replay needs no key, no
warehouse, no index, no embedding model and no network, costs nothing and gives the same
answers every time.

A replayed request must match a recorded one exactly. Anything else means a prompt, the pack,
the code that builds a request or the model has changed since the recording, and the run must
be recorded again.
"""

import base64
import hashlib
import json
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from vardex.documents import DocumentSpec
from vardex.embeddings import Embedder, Reranker, Vector
from vardex.events import Batch, Event, FeedSpec
from vardex.llm import Answer, ModelClient, Reasoning, Request, ToolCall
from vardex.pricing import Usage
from vardex.retrieval import Filters, Index, Passage
from vardex.sql import Column, QueryError, Result, Table, Warehouse


class NotRecorded(Exception):
    """Raised when a replay meets a model request or a query that was never recorded."""


def digest(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def request_key(model: str, request: Request) -> str:
    """A fingerprint of everything a model is sent: change one character, and it changes."""
    schema = request.output_schema.model_json_schema() if request.output_schema else None
    return digest(
        {
            "model": model,
            "system": request.system,
            # A picture counts by its fingerprint; a message without one keeps its old key
            "messages": [
                [m.role, m.text, *([digest([i.media_type, i.data.hex()]) for i in m.images])]
                for m in request.messages
            ],
            "max_tokens": request.max_tokens,
            "effort": request.effort,
            "schema": schema,
            "cache": [request.cache_system, request.cache_conversation],
            # A request without a tail keeps the key it had before tails existed
            **({"system_tail": request.system_tail} if request.system_tail else {}),
        }
    )


def query_key(sql: str, max_rows: int) -> str:
    return digest([sql, max_rows])


def feed_key(url: str, cursor: str | None) -> str:
    """One read of one feed, by where it read from. A feed read twice from the same cursor
    gives the same changes, which is what makes a recorded run repeatable."""
    return digest([url, cursor or ""])


def pack_vector(vector: Vector) -> str:
    """A vector as text, exactly: its 32-bit numbers, as embedding models give them, in
    base64. A quarter of the size of the numbers written out."""
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def unpack_vector(text: str) -> Vector:
    return array("f", base64.b64decode(text)).tolist()


def vector_key(model: str, kind: str, text: str) -> str:
    return digest([model, kind, text])


def search_key(how: str, what: str, filters: Filters, limit: int) -> str:
    return digest([how, what, asdict(filters), limit])


def to_json(value: Any) -> Any:
    """A value from a row, in a form JSON keeps exactly: a Decimal as its digits."""
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)  # anything rarer, such as an interval, as the text a person would read


def from_json(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    ((kind, text),) = value.items()
    if kind == "decimal":
        return Decimal(text)
    if kind == "datetime":
        return datetime.fromisoformat(text)
    return date.fromisoformat(text)


def dump_answer(answer: Answer) -> dict[str, Any]:
    data = asdict(answer)
    data["cost_usd"] = None if answer.cost_usd is None else str(answer.cost_usd)
    return data


def load_answer(data: dict[str, Any]) -> Answer:
    cost = data["cost_usd"]
    return Answer(
        **{
            **data,
            "usage": Usage(**data["usage"]),
            "cost_usd": cost and Decimal(cost),
            # asdict flattened these to plain dicts, and a replayed turn goes back to the
            # loop exactly as a live one does, so they have to be rebuilt as themselves.
            "tool_calls": tuple(ToolCall(**call) for call in data.get("tool_calls", ())),
            "reasoning": tuple(Reasoning(**block) for block in data.get("reasoning", ())),
        }
    )


def dump_result(result: Result) -> dict[str, Any]:
    data = asdict(result)
    data["rows"] = [[to_json(v) for v in row] for row in result.rows]
    return data


def load_result(data: dict[str, Any]) -> Result:
    rows = [tuple(from_json(v) for v in row) for row in data["rows"]]
    return Result(**{**data, "rows": rows})


@dataclass
class Recording:
    """Every model answer and query result of one run, by the fingerprint of what was asked."""

    model: str
    warehouse: str
    dialect: str
    recorded_on: str  # the day it was recorded, ISO
    tables: list[Table] | None = None
    answers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # in arrival order
    queries: dict[str, dict[str, Any]] = field(default_factory=dict)  # a result or an error
    verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)  # each case's grades
    embedder: str = ""  # the embedding model, when one was used
    reranker: str = ""  # the reranker, when one was used
    vectors: dict[str, str] = field(default_factory=dict)  # by text: a packed vector
    scores: dict[str, list[float]] = field(default_factory=dict)  # by query and texts
    searches: dict[str, list[int]] = field(default_factory=dict)  # by search: passage ids
    passages: dict[int, dict[str, Any]] = field(default_factory=dict)  # each passage once
    feeds: dict[str, dict[str, Any]] = field(default_factory=dict)  # by feed and cursor

    def save(self, path: Path) -> None:
        """Write it as JSON lines: what was recorded first, then one line per call or query,
        then one per case with its verdict."""
        head = {
            "model": self.model,
            "warehouse": self.warehouse,
            "dialect": self.dialect,
            "recorded_on": self.recorded_on,
            "tables": [asdict(table) for table in self.tables or []],
            "embedder": self.embedder,
            "reranker": self.reranker,
        }
        lines = [json.dumps(head, ensure_ascii=False)]
        for key, answers in self.answers.items():
            lines += [json.dumps({"answer": key, **a}, ensure_ascii=False) for a in answers]
        for key, query in self.queries.items():
            lines.append(json.dumps({"query": key, **query}, ensure_ascii=False))
        lines += [json.dumps({"vector": k, "values": v}) for k, v in self.vectors.items()]
        lines += [json.dumps({"scores": k, "values": v}) for k, v in self.scores.items()]
        lines += [json.dumps({"search": k, "passages": v}) for k, v in self.searches.items()]
        for key, batch in self.feeds.items():
            lines.append(json.dumps({"feed": key, **batch}, ensure_ascii=False))
        for pid, passage in self.passages.items():
            lines.append(json.dumps({"passage": pid, **passage}, ensure_ascii=False))
        for case, verdict in self.verdicts.items():
            lines.append(json.dumps({"verdict": case, **verdict}, ensure_ascii=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Recording":
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        head = lines[0]
        tables = [Table(t["name"], [Column(**c) for c in t["columns"]]) for t in head.pop("tables")]
        recording = cls(**head, tables=tables)
        for line in lines[1:]:
            if "answer" in line:
                recording.answers.setdefault(line.pop("answer"), []).append(line)
            elif "query" in line:
                recording.queries[line.pop("query")] = line
            elif "vector" in line:
                recording.vectors[line["vector"]] = line["values"]
            elif "scores" in line:
                recording.scores[line["scores"]] = line["values"]
            elif "search" in line:
                recording.searches[line["search"]] = line["passages"]
            elif "feed" in line:
                recording.feeds[line.pop("feed")] = line
            elif "passage" in line:
                recording.passages[line.pop("passage")] = line
            else:
                recording.verdicts[line.pop("verdict")] = line
        return recording


class RecordingClient:
    """A ModelClient that passes every request on, and records the answer."""

    def __init__(self, client: ModelClient, recording: Recording) -> None:
        self.client = client
        self.model = client.model
        self.recording = recording

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        answer = self.client.send(request, on_text=on_text)
        key = request_key(self.model, request)
        self.recording.answers.setdefault(key, []).append(dump_answer(answer))
        return answer

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)


class ReplayClient:
    """A ModelClient that answers from a recording, in the order the answers were recorded."""

    def __init__(self, recording: Recording) -> None:
        self.model = recording.model
        self.waiting = {key: list(answers) for key, answers in recording.answers.items()}

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        recorded = self.waiting.get(request_key(self.model, request))
        if not recorded:
            raise NotRecorded("a model request that was not recorded")
        answer = load_answer(recorded.pop(0))
        if on_text:
            on_text(answer.text)
        return answer

    def count_tokens(self, text: str) -> int:
        raise NotRecorded("token counts are not recorded")


class RecordingWarehouse:
    """A Warehouse that passes every query on, and records its result or its error."""

    def __init__(self, warehouse: Warehouse, recording: Recording) -> None:
        self.warehouse = warehouse
        self.name = warehouse.name
        self.dialect = warehouse.dialect
        self.recording = recording

    def tables(self) -> list[Table]:
        self.recording.tables = self.warehouse.tables()
        return self.recording.tables

    def run(self, sql: str, max_rows: int = 1000) -> Result:
        key = query_key(sql, max_rows)
        try:
            result = self.warehouse.run(sql, max_rows=max_rows)
        except QueryError as err:
            self.recording.queries[key] = {"error": str(err)}
            raise
        self.recording.queries[key] = {"result": dump_result(result)}
        return result

    def close(self) -> None:
        self.warehouse.close()


class ReplayWarehouse:
    """A Warehouse that answers from a recording. The same query gives the same rows, however
    often it is asked, as it would from a warehouse nobody changes."""

    def __init__(self, recording: Recording) -> None:
        self.name = recording.warehouse
        self.dialect = recording.dialect
        self.recording = recording

    def tables(self) -> list[Table]:
        if self.recording.tables is None:
            raise NotRecorded("the warehouse's tables were not recorded")
        return self.recording.tables

    def run(self, sql: str, max_rows: int = 1000) -> Result:
        recorded = self.recording.queries.get(query_key(sql, max_rows))
        if recorded is None:
            raise NotRecorded(f"a query that was not recorded: {' '.join(sql.split())[:80]}")
        if "error" in recorded:
            raise QueryError(recorded["error"])
        return load_result(recorded["result"])

    def close(self) -> None:
        pass


def dump_batch(batch: Batch) -> dict[str, Any]:
    return {
        "events": [asdict(event) for event in batch.events],
        "cursor": batch.cursor,
        "pages": batch.pages,
        "more": batch.more,
        "read": batch.read,
    }


def load_batch(data: dict[str, Any]) -> Batch:
    return Batch(
        events=[Event(**event) for event in data["events"]],
        cursor=data["cursor"],
        pages=data["pages"],
        more=data["more"],
        read=data["read"],
    )


class RecordingFeed:
    """A feed reader that reads the real feed and records what came back."""

    def __init__(self, read: Callable[[FeedSpec, str | None], Batch], recording: Recording) -> None:
        self.read = read
        self.recording = recording

    def __call__(self, spec: FeedSpec, cursor: str | None) -> Batch:
        batch = self.read(spec, cursor)
        self.recording.feeds[feed_key(spec.url, cursor)] = dump_batch(batch)
        return batch


class ReplayFeed:
    """A feed reader that answers from a recording, and never touches the network."""

    def __init__(self, recording: Recording) -> None:
        self.recording = recording

    def __call__(self, spec: FeedSpec, cursor: str | None) -> Batch:
        recorded = self.recording.feeds.get(feed_key(spec.url, cursor))
        if recorded is None:
            raise NotRecorded(
                f"this run read {spec.url} from {cursor or 'the beginning'}, which the "
                "recording does not have. The cursor has moved since it was recorded: "
                "record the run again, or reset the cursor to the one in the recording."
            )
        return load_batch(recorded)


class RecordingEmbedder:
    """An Embedder that passes every text on, and records its vector."""

    def __init__(self, embedder: Embedder, recording: Recording) -> None:
        self.embedder = embedder
        self.model = embedder.model
        self.dimensions = embedder.dimensions
        self.recording = recording
        recording.embedder = embedder.model

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        vectors = self.embedder.embed_documents(texts)
        for text, vector in zip(texts, vectors, strict=True):
            self.recording.vectors[vector_key(self.model, "document", text)] = pack_vector(vector)
        return vectors

    def embed_query(self, text: str) -> Vector:
        vector = self.embedder.embed_query(text)
        self.recording.vectors[vector_key(self.model, "query", text)] = pack_vector(vector)
        return vector


class ReplayEmbedder:
    """An Embedder that gives back the vectors of a recording: no model is loaded."""

    def __init__(self, recording: Recording) -> None:
        self.model = recording.embedder
        self.dimensions = 0  # not known, and not needed: nothing is stored
        self.recording = recording

    def vector(self, kind: str, text: str) -> Vector:
        packed = self.recording.vectors.get(vector_key(self.model, kind, text))
        if packed is None:
            raise NotRecorded(f"a text that was not embedded: {' '.join(text.split())[:80]}")
        return unpack_vector(packed)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self.vector("document", text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self.vector("query", text)


class RecordingReranker:
    """A Reranker that passes every question on, and records its scores."""

    def __init__(self, reranker: Reranker, recording: Recording) -> None:
        self.reranker = reranker
        self.model = reranker.model
        self.recording = recording
        recording.reranker = reranker.model

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.reranker.score(query, texts)
        self.recording.scores[digest([self.model, query, texts])] = scores
        return scores


class ReplayReranker:
    """A Reranker that gives back the scores of a recording."""

    def __init__(self, recording: Recording) -> None:
        self.model = recording.reranker
        self.recording = recording

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.recording.scores.get(digest([self.model, query, texts]))
        if scores is None:
            raise NotRecorded(f"a rerank that was not recorded: {query[:80]}")
        return scores


def dump_passage(passage: Passage) -> dict[str, Any]:
    return {"document": passage.document.model_dump(), "page": passage.page, "text": passage.text}


class RecordingIndex:
    """An Index that passes every search on, and records the passages it found. Each passage
    is kept once, however many searches find it."""

    def __init__(self, index: Index, recording: Recording) -> None:
        self.index = index
        self.name = index.name
        self.recording = recording

    def keep(self, key: str, passages: list[Passage]) -> list[Passage]:
        self.recording.searches[key] = [p.id for p in passages]
        for p in passages:
            self.recording.passages[p.id] = dump_passage(p)
        return passages

    def documents(self) -> dict[str, str]:
        return self.index.documents()

    def put(self, *args: Any) -> None:
        raise NotImplementedError("a recording only searches")

    def remove(self, document: str) -> None:
        raise NotImplementedError("a recording only searches")

    def by_meaning(self, vector: Vector, filters: Filters, limit: int) -> list[Passage]:
        found = self.index.by_meaning(vector, filters, limit)
        return self.keep(search_key("meaning", pack_vector(vector), filters, limit), found)

    def by_words(self, query: str, filters: Filters, limit: int) -> list[Passage]:
        found = self.index.by_words(query, filters, limit)
        return self.keep(search_key("words", query, filters, limit), found)

    def close(self) -> None:
        self.index.close()


class ReplayIndex:
    """An Index that answers searches from a recording."""

    def __init__(self, recording: Recording) -> None:
        self.name = "a recording"
        self.recording = recording

    def found(self, key: str, what: str) -> list[Passage]:
        ids = self.recording.searches.get(key)
        if ids is None:
            raise NotRecorded(f"a search that was not recorded: {what[:80]}")
        return [self.passage(int(pid)) for pid in ids]

    def passage(self, pid: int) -> Passage:
        data = self.recording.passages[pid]
        return Passage(pid, DocumentSpec(**data["document"]), data["page"], data["text"])

    def documents(self) -> dict[str, str]:
        return {}

    def put(self, *args: Any) -> None:
        raise NotImplementedError("a replay only searches")

    def remove(self, document: str) -> None:
        raise NotImplementedError("a replay only searches")

    def by_meaning(self, vector: Vector, filters: Filters, limit: int) -> list[Passage]:
        return self.found(search_key("meaning", pack_vector(vector), filters, limit), "by meaning")

    def by_words(self, query: str, filters: Filters, limit: int) -> list[Passage]:
        return self.found(search_key("words", query, filters, limit), query)

    def close(self) -> None:
        pass
