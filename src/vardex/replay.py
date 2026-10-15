"""Record and replay: a run's model answers and query results, saved and played back.

A recording is taken at Vardex's own two interfaces, ModelClient and Warehouse, so it holds
Vardex's words only: no API key, no provider's wire format, no database driver. A replay
needs neither a key nor a warehouse, costs nothing and gives the same answers every time.

A replayed request must match a recorded one exactly. Anything else means a prompt, the pack,
the code that builds a request or the model has changed since the recording, and the run must
be recorded again.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from vardex.llm import Answer, ModelClient, Request
from vardex.pricing import Usage
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
            "messages": [[m.role, m.text] for m in request.messages],
            "max_tokens": request.max_tokens,
            "effort": request.effort,
            "schema": schema,
            "cache": [request.cache_system, request.cache_conversation],
        }
    )


def query_key(sql: str, max_rows: int) -> str:
    return digest([sql, max_rows])


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
    return Answer(**{**data, "usage": Usage(**data["usage"]), "cost_usd": cost and Decimal(cost)})


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

    def save(self, path: Path) -> None:
        """Write it as JSON lines: what was recorded first, then one line per call or query."""
        head = {
            "model": self.model,
            "warehouse": self.warehouse,
            "dialect": self.dialect,
            "recorded_on": self.recorded_on,
            "tables": [asdict(table) for table in self.tables or []],
        }
        lines = [json.dumps(head, ensure_ascii=False)]
        for key, answers in self.answers.items():
            lines += [json.dumps({"answer": key, **a}, ensure_ascii=False) for a in answers]
        for key, query in self.queries.items():
            lines.append(json.dumps({"query": key, **query}, ensure_ascii=False))
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
            else:
                recording.queries[line.pop("query")] = line
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
