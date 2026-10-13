"""Text-to-SQL: a question in; an answer out, with the SQL it ran and the rows it used.

The model writes one query against the pack's semantic layer. The engine checks the query,
runs it read-only, and gives the rows back to the model to answer from. Every number in the
answer must trace back to the rows, the query or the question, or the answer is not shown.
"""

from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from vardex.llm import Answer, Effort, Message, ModelClient, Request
from vardex.packs import describe_problem
from vardex.prompts import ANSWER, ANSWER_REPAIR, SQL_REPAIR, WRITE_SQL
from vardex.provenance import Figure, trace
from vardex.semantic import Knowledge
from vardex.sql import QueryError, Result, Table, Warehouse, check_sql, format_value, transpile

DIALECTS = {"duckdb": "DuckDB", "postgres": "PostgreSQL", "snowflake": "Snowflake"}
MAX_VALUES = 50  # list a column's values only when it has at most this many
Kind = Literal["answer", "clarify", "refuse"]


class Plan(BaseModel):
    """What the model decides to do with a question: the shape its first answer must have."""

    action: Literal["query", "clarify", "refuse"] = Field(
        description="query to answer with SQL; clarify to ask one question back; refuse when "
        "the tables cannot answer, or the question asks to change data."
    )
    sql: str | None = Field(
        description="With query: one SELECT statement, one clause per line. Otherwise empty."
    )
    message: str | None = Field(
        description="With clarify: the question to ask. With refuse: why. Otherwise empty."
    )
    assumptions: list[str] = Field(
        description="Each choice the question did not state, such as which year 'last year' is."
    )


@dataclass(frozen=True)
class Reply:
    """An answer to a question, with everything it rests on."""

    kind: Kind
    text: str  # the answer, the question back, or the reason for refusing
    assumptions: list[str]
    result: Result | None  # the query that ran and its rows; None if none ran
    figures: list[Figure]  # every number in the text, and where it came from
    answers: list[Answer]  # the model calls it took


class AnswerError(Exception):
    """Raised when a question gets no usable answer. The message says why."""

    def __init__(self, reason: str, answers: list[Answer], result: Result | None = None) -> None:
        super().__init__(reason)
        self.answers = answers  # so the calls can still be logged and paid for
        self.result = result  # the last query that ran, if any, to show beside the reason


def describe_tables(knowledge: Knowledge, tables: list[Table]) -> str:
    """Each table a model may use: what it holds, and each column with its type and meaning."""
    types = {t.name: {c.name: c.type for c in t.columns} for t in tables}
    parts = []
    for name, doc in knowledge.tables.items():
        lines = [f'<table name="{name}">', " ".join(doc.description.split())]
        for column in doc.columns:
            kind = types.get(name, {}).get(column.name, "unknown type")
            lines.append(f"- {column.name} ({kind}): {' '.join(column.description.split())}")
        parts.append("\n".join([*lines, "</table>"]))
    return "<tables>\n" + "\n".join(parts) + "\n</tables>"


def describe_ontology(knowledge: Knowledge) -> str:
    """The objects, their keys, and the links to join them along."""
    ontology = knowledge.ontology
    lines = ["<objects>"]
    for obj in ontology.objects:
        named = f", named by {obj.label}" if obj.label else ""
        lines.append(
            f"- {obj.name}: {' '.join(obj.description.split())} One row per {obj.name} in "
            f"{obj.table}, identified by {obj.key}{named}."
        )
    if ontology.links:
        lines.append("Links:")
    for link in ontology.links:
        name, column = link.source
        start, end = ontology.object(name), ontology.object(link.to)
        lines.append(
            f"- {name} → {end.name} ({link.name}): {start.table}.{column} = "
            f"{end.table}.{end.key}. {' '.join(link.description.split())} Many {name} to one "
            f"{end.name}."
        )
    return "\n".join([*lines, "</objects>"])


def describe_values(knowledge: Knowledge, warehouse: Warehouse) -> str:
    """The values in each column a question may name, as the warehouse spells them, so a
    filter matches: 'Vestland', not 'VESTLAND' or 'Vestland county'."""
    lines = []
    for obj in knowledge.ontology.objects:
        for column in obj.values:
            result = warehouse.run(
                f"select distinct {column} from {obj.table} where {column} is not null"
                f" order by {column}",
                max_rows=MAX_VALUES,
            )
            if result.truncated:
                lines.append(f"- {obj.table}.{column}: more than {MAX_VALUES} values")
            else:
                values = ", ".join(f"'{format_value(row[0])}'" for row in result.rows)
                lines.append(f"- {obj.table}.{column}: {values}")
    return "<values>\n" + "\n".join(lines) + "\n</values>" if lines else ""


def describe_metrics(knowledge: Knowledge, dialect: str) -> str:
    """Every metric, with its SQL in the warehouse's dialect, its unit and notes."""
    parts = []
    for metric in knowledge.metrics:
        lines = [
            f'<metric name="{metric.name}" table="{metric.table}" unit="{metric.unit}">',
            metric.description,
            f"SQL: {transpile(metric.sql, knowledge.questions.dialect, dialect)}",
            *(f"- {note}" for note in metric.notes),
            "</metric>",
        ]
        parts.append("\n".join(lines))
    return "<metrics>\n" + "\n".join(parts) + "\n</metrics>"


def describe_questions(knowledge: Knowledge, dialect: str) -> str:
    """The pack's glossary, and its worked examples in the warehouse's dialect."""
    questions = knowledge.questions
    parts = []
    if questions.glossary:
        terms = "\n".join(
            f"- {term}: {' '.join(meaning.split())}" for term, meaning in questions.glossary.items()
        )
        parts.append(f"<glossary>\n{terms}\n</glossary>")
    for example in questions.examples:
        sql = transpile(example.sql.strip(), questions.dialect, dialect)
        parts.append(
            f"<example>\n<question>{' '.join(example.question.split())}</question>\n"
            f"<sql>\n{sql}\n</sql>\n</example>"
        )
    return "\n".join(parts)


def system_prompt(knowledge: Knowledge, warehouse: Warehouse, today: date, max_rows: int) -> str:
    """The engine's text-to-SQL prompt, with the pack's semantic layer filled in."""
    context = "\n\n".join(
        part
        for part in (
            describe_tables(knowledge, warehouse.tables()),
            describe_ontology(knowledge),
            describe_values(knowledge, warehouse),
            describe_metrics(knowledge, warehouse.dialect),
            describe_questions(knowledge, warehouse.dialect),
        )
        if part
    )
    return WRITE_SQL.format(
        dialect=DIALECTS.get(warehouse.dialect, warehouse.dialect),
        max_rows=max_rows,
        today=today.isoformat(),
        context=context,
    )


def build_request(
    knowledge: Knowledge,
    warehouse: Warehouse,
    question: str,
    *,
    effort: Effort | None = "medium",
    max_rows: int = 100,
    today: date | None = None,
) -> Request:
    """The request that asks a model to write SQL for one question."""
    return Request(
        system=system_prompt(knowledge, warehouse, today or date.today(), max_rows),
        messages=[Message("user", question)],
        effort=effort,
        output_schema=Plan,
        cache_system=True,  # the same tables and metrics for every question
    )


def rows_as_text(result: Result) -> str:
    """The rows for a model to read: one line per row, values separated by |."""
    lines = [" | ".join(result.columns)]
    lines += [" | ".join(format_value(v) for v in row) for row in result.rows]
    if result.truncated:
        lines.append(f"(only the first {len(result.rows):,} rows; the query returned more)")
    elif not result.rows:
        lines.append("(no rows)")
    return "\n".join(lines)


def plan_and_run(
    client: ModelClient,
    warehouse: Warehouse,
    knowledge: Knowledge,
    request: Request,
    *,
    repairs: int,
    max_rows: int,
) -> tuple[Plan, Result | None, list[Answer]]:
    """Ask for a plan; when it is a query, check it and run it. When either fails, show the
    model what went wrong and ask again, up to `repairs` times."""
    answers: list[Answer] = []
    allowed = set(knowledge.tables)
    for _ in range(repairs + 1):
        answer = client.send(request)
        answers.append(answer)
        if answer.stop_reason != "end":
            raise AnswerError(f"the model stopped early ({answer.stop_reason})", answers)
        try:
            plan = Plan.model_validate_json(answer.text)
        except ValidationError as err:
            problems = [describe_problem(e) for e in err.errors()]
        else:
            if plan.action != "query":
                return plan, None, answers
            problems = check_sql(plan.sql or "", warehouse.dialect, allowed)
            if not problems:
                try:
                    return plan, warehouse.run(plan.sql or "", max_rows=max_rows), answers
                except QueryError as err:
                    problems = [f"the warehouse refused it: {err}"]
        request = replace(
            request,
            messages=[
                *request.messages,
                Message("assistant", answer.text),
                Message("user", SQL_REPAIR.format(problems="\n".join(f"- {p}" for p in problems))),
            ],
        )
    raise AnswerError("no query that could run: " + "; ".join(problems), answers)


def answer_from_rows(
    client: ModelClient,
    question: str,
    plan: Plan,
    result: Result,
    *,
    repairs: int,
) -> tuple[str, list[Figure], list[Answer]]:
    """Ask the model to answer from the rows, and trace every number in what it writes."""
    assumptions = "\n".join(f"- {a}" for a in plan.assumptions) or "- none"
    request = Request(
        system=ANSWER,
        messages=[
            Message(
                "user",
                f"<question>{question}</question>\n<assumptions>\n{assumptions}\n</assumptions>\n"
                f"<sql>\n{result.sql}\n</sql>\n<rows>\n{rows_as_text(result)}\n</rows>",
            )
        ],
        effort="low",
    )
    answers: list[Answer] = []
    for _ in range(repairs + 1):
        answer = client.send(request)
        answers.append(answer)
        figures = trace(answer.text, [result], question)
        untraced = [f.text for f in figures if not f.traced]
        if not untraced:
            return answer.text.strip(), figures, answers
        request = replace(
            request,
            messages=[
                *request.messages,
                Message("assistant", answer.text),
                Message("user", ANSWER_REPAIR.format(figures=", ".join(untraced))),
            ],
        )
    raise AnswerError(
        f"the answer has numbers no query returned: {', '.join(untraced)}", answers, result
    )


def answer(
    client: ModelClient,
    warehouse: Warehouse,
    knowledge: Knowledge,
    question: str,
    *,
    effort: Effort | None = "medium",
    repairs: int = 1,
    max_rows: int = 100,
    today: date | None = None,
) -> Reply:
    """Answer one question from a warehouse: plan a query, run it, answer from its rows."""
    request = build_request(
        knowledge, warehouse, question, effort=effort, max_rows=max_rows, today=today
    )
    plan, result, answers = plan_and_run(
        client, warehouse, knowledge, request, repairs=repairs, max_rows=max_rows
    )
    if result is None:
        kind: Kind = "clarify" if plan.action == "clarify" else "refuse"
        return Reply(kind, plan.message or "", plan.assumptions, None, [], answers)
    try:
        text, figures, more = answer_from_rows(client, question, plan, result, repairs=repairs)
    except AnswerError as err:
        raise AnswerError(str(err), answers + err.answers, result) from err
    return Reply("answer", text, plan.assumptions, result, figures, answers + more)
