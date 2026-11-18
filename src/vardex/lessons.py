"""Lessons: how each capability's training pairs are made, checked and written.

Answers from the warehouse train the SQL writer, the step every answer depends on and the one
a small model gets wrong most: a pair is a question and the query that answers it, and the
query must pass Vardex's SQL checks and return rows. The pack's worked examples are few, so
more are made from them by putting other values in place of the ones an example names:
another company, another county, another year. Code writes them, not a model.

Extractions train the model that reads documents: a pair is a document and the values it holds, as
the extraction's own checks accept them. They come from the extraction's examples and from
extractions a person approved; none are made by swapping values, because a changed figure in
a document is a document that does not exist.
"""

import difflib
import json
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

import sqlglot
from sqlglot import exp

from vardex.embeddings import Embedder
from vardex.evals import ExtractionCase, words
from vardex.extraction import ExtractionSpec, build_request, example_values, read_values
from vardex.semantic import Example, Knowledge
from vardex.sql import QueryError, Warehouse, check_sql, transpile
from vardex.text_to_sql import Plan
from vardex.text_to_sql import build_request as sql_request
from vardex.traces import Trace
from vardex.tracing import INPUT_MESSAGES, OUTPUT_MESSAGES
from vardex.training import Pair, as_chat

# An input this close to a golden one is the golden one, reworded: "Equinor" for "Equinor
# ASA". Lower, and it starts to catch the same question about another county, which teaches
# the pattern the examples already show, not the answer to the test.
SIMILAR = 0.95
VARIANTS = 12  # pairs generated from one example, at most
TRIES = 40  # values tried per example before giving up on finding more that return rows
YEARS = range(1990, 2101)  # a number in this range, in the question and the query, is a year


def near(text: str, golden: list[str]) -> bool:
    """True when a text is a golden one, or nearly, by its words."""
    said = words(text)
    return any(difflib.SequenceMatcher(None, said, g).ratio() >= SIMILAR for g in golden)


def normal_sql(sql: str, dialect: str) -> str:
    """A query as sqlglot writes it back, so spacing and case do not hide a copy."""
    try:
        return sqlglot.transpile(sql, read=dialect, write=dialect, pretty=False)[0].lower()
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError):
        return " ".join(sql.lower().split())


@dataclass
class SqlLesson:
    """Questions and the queries that answer them, for the SQL writer."""

    knowledge: Knowledge
    warehouse: Warehouse
    golden: list[Any]  # every suite's cases: the agent's questions too
    today: date
    embedder: Embedder | None = None

    capability = "answer"

    def __post_init__(self) -> None:
        dialect = self.knowledge.questions.dialect
        self.questions = [words(c.question) for c in self.golden if hasattr(c, "question")]
        self.queries = {normal_sql(c.sql, dialect) for c in self.golden if getattr(c, "sql", None)}

    def worked(self) -> Iterator[Pair]:
        for example in self.knowledge.questions.examples:
            yield Pair(" ".join(example.question.split()), example.sql.strip(), "example")
            yield from variants(example, self.knowledge, self.warehouse)

    def approved(self, trace: Trace) -> Iterator[Pair]:
        """The question and the query that answered it, when a query ran."""
        if trace.question and trace.sql:
            yield Pair(trace.question, trace.sql, "approved", origin=trace.trace_id)

    def why_not(self, pair: Pair) -> str | None:
        if near(pair.input, self.questions):
            return "its question is a golden question"
        if normal_sql(pair.target, self.knowledge.questions.dialect) in self.queries:
            return "its query is a golden reference query"
        return None

    def chat(self, pair: Pair) -> dict[str, Any]:
        """The request vardex answer sends for the question today, and the plan to run the
        query as the answer."""
        request = sql_request(
            self.knowledge, self.warehouse, pair.input, today=self.today, embedder=self.embedder
        )
        system = "\n\n".join(part for part in (request.system, request.system_tail) if part)
        messages = [{"role": m.role, "content": m.text} for m in request.messages]
        plan = Plan(action="query", sql=pair.target, message=None, assumptions=[])
        return as_chat(system, messages, plan.model_dump_json())


@dataclass
class ExtractionLesson:
    """Documents and the values they hold, for one extraction."""

    spec: ExtractionSpec
    documents: list[str]  # the golden documents: every case's, in the extraction's suites

    capability = "extract"

    def worked(self) -> Iterator[Pair]:
        for example in self.spec.examples:
            values = example_values(self.spec, example)
            yield Pair(example.document.strip(), json.dumps(values, default=str), "example")

    def approved(self, trace: Trace) -> Iterator[Pair]:
        """Each document in the trace whose values passed the checks, read back from the
        call that gave them; it needs traces that keep content, as they do by default."""
        for span in trace.spans:
            if span.name != "extract" or not all(ok for _, ok in span.verdicts()):
                continue
            for call in trace.children(span):
                document = unwrap(text_of(call.get(INPUT_MESSAGES), "user"))
                answer = text_of(call.get(OUTPUT_MESSAGES), "assistant")
                if document and answer and not read_values(self.spec, answer, document)[0]:
                    yield Pair(document, answer, "approved", origin=trace.trace_id)

    def why_not(self, pair: Pair) -> str | None:
        if near(pair.input, [words(document) for document in self.documents]):
            return "its document is a golden document"
        return None

    def chat(self, pair: Pair) -> dict[str, Any]:
        request = build_request(self.spec, pair.input)
        messages = [{"role": m.role, "content": m.text} for m in request.messages]
        return as_chat(request.system, messages, pair.target)


def golden_documents(suites: dict[str, Any], extraction: str, folder: Any) -> list[str]:
    """The text of every document a suite of this extraction grades."""
    return [
        (folder / case.document).read_text(encoding="utf-8")
        for suite in suites.values()
        if getattr(suite, "extraction", None) == extraction
        for case in suite.cases
        if isinstance(case, ExtractionCase)
    ]


def text_of(messages: Any, role: str) -> str:
    """The text of the last message in this role, from a span's messages: a list, or the
    same as JSON when the trace came back from a file."""
    if isinstance(messages, str):
        messages = json.loads(messages)
    texts = [
        part.get("content", "")
        for message in messages or []
        if message.get("role") == role
        for part in message.get("parts", [])
        if part.get("type") == "text"
    ]
    return texts[-1] if texts else ""


def unwrap(text: str) -> str:
    """A document from the message that carried it, without the tags around it."""
    match = re.fullmatch(r"\s*<document>\n?(.*?)\n?</document>\s*", text, re.DOTALL)
    return match.group(1) if match else ""


def literals(sql: str, dialect: str) -> tuple[list[tuple[str, str]], list[int]]:
    """What an example's query names: each text it compares a column with, as the column and
    the text, such as (name, 'TELENOR ASA'); and its years, such as 2024."""
    tree = sqlglot.parse_one(sql, read=dialect)
    texts = [
        (node.left.name, node.right.this)
        for node in tree.find_all(exp.EQ)
        if isinstance(node.left, exp.Column)
        and isinstance(node.right, exp.Literal)
        and node.right.is_string
    ]
    years = [int(node.this) for node in tree.find_all(exp.Literal) if not node.is_string
             and node.this.isdigit() and int(node.this) in YEARS]  # fmt: skip
    return texts, sorted(set(years))


def written(value: str, like: str) -> str:
    """A value from the warehouse, written the way the question wrote the one it replaces.
    A name the register keeps in capitals, TELENOR ASA, is written Telenor ASA when the
    question wrote it so, and short words keep their capitals: Aker BP ASA, not Aker Bp Asa.
    Any other value is written as the warehouse spells it: Møre og Romsdal, company."""
    if like.isupper() or not value.isupper():
        return value.lower() if like.islower() else value
    return " ".join(w if len(w) <= 3 else w.capitalize() for w in value.split())


def in_question(question: str, value: str) -> str | None:
    """The words in the question that stand for a value in the query: the value itself, or
    the value without its last word when that is a legal form, as Orkla for ORKLA ASA."""
    for form in (value, " ".join(value.split()[:-1]) if len(value.split()) > 1 else ""):
        if form:
            # As a whole word: the group in "for the group", never the one in "groups"
            match = re.search(rf"\b{re.escape(form)}\b", question, re.IGNORECASE)
            if match:
                return match.group()
    return None


def columns_of(knowledge: Knowledge) -> list[tuple[str, str]]:
    """Every column whose values a question names: each object's label, and its values."""
    found = []
    for obj in knowledge.ontology.objects:
        if obj.label:
            found.append((obj.table, obj.label))
        found += [(obj.table, column) for column in obj.values]
    return found


def alternatives(warehouse: Warehouse, table: str, column: str, seed: int) -> list[str]:
    """Values from the same column to try in place of one, in an order set by the seed."""
    result = warehouse.run(
        f"select distinct {column} from {table} where {column} is not null", max_rows=5000
    )
    # Sorted first: a database returns distinct values in no promised order, and the same
    # seed must give the same pairs on every run
    values = sorted(str(row[0]) for row in result.rows)
    random.Random(seed).shuffle(values)
    return values


def variants(
    example: Example, knowledge: Knowledge, warehouse: Warehouse, *, limit: int = VARIANTS
) -> Iterator[Pair]:
    """Pairs made from one worked example by changing what it names: the text it filters on,
    for another value of the same column, or its years, all by the same number of years. Each
    is kept only if its query passes the checks and returns rows."""
    dialect = knowledge.questions.dialect
    texts, years = literals(example.sql, dialect)
    seen: set[str] = {words(example.question)}
    made = 0
    swaps: list[Callable[[int], tuple[str, str] | None]] = []
    # Only a column that names things, an object's label or values, is swapped: a filter such
    # as accounts_type = 'group' is part of what the query means, not what it is about
    named = {column: table for table, column in columns_of(knowledge)}
    for column, text in texts:
        found = in_question(example.question, text)
        if found and column in named:
            options = [
                value
                for value in alternatives(warehouse, named[column], column, len(example.question))
                if value != text
            ]
            whole = found.lower() == text.lower()  # Telenor ASA, not Telenor alone

            def swap(i: int, text=text, found=found, options=options, whole=whole):  # type: ignore[no-untyped-def]
                if i >= len(options):
                    return None
                new = options[i]
                said = new if whole else " ".join(new.split()[:-1]) or new
                return (
                    example.question.replace(found, written(said, found)),
                    example.sql.replace(f"'{text}'", "'" + new.replace("'", "''") + "'"),
                )

            swaps.append(swap)
    if years and all(str(year) in example.question for year in years):

        def shift(i: int) -> tuple[str, str] | None:
            delta = (-1, 1, -2)[i] if i < 3 else None
            if delta is None:
                return None
            question, sql = example.question, example.sql
            for year in sorted(years, reverse=delta > 0):
                question = question.replace(str(year), str(year + delta))
                sql = re.sub(rf"\b{year}\b", str(year + delta), sql)
            return question, sql

        swaps.append(shift)
    allowed = set(knowledge.tables)
    for swap in swaps:
        for i in range(TRIES):
            if made >= limit:
                return
            swapped = swap(i)
            if swapped is None:
                break
            question, sql = swapped
            if words(question) in seen:
                continue
            if check_sql(transpile(sql, dialect, warehouse.dialect), warehouse.dialect, allowed):
                continue
            try:
                result = warehouse.run(transpile(sql, dialect, warehouse.dialect), max_rows=10)
            except QueryError:
                continue
            if result.rows:
                seen.add(words(question))
                made += 1
                yield Pair(" ".join(question.split()), sql.strip(), "generated", example.question)
