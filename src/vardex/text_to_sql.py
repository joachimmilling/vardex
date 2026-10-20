"""Text-to-SQL: a question in; an answer out, with the SQL it ran and the rows it used.

The model writes one query against the pack's semantic layer. The engine checks the query,
runs it read-only, and gives the rows back to the model to answer from. Every number in the
answer must trace back to the rows, the query or the question, or the answer is not shown.
"""

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from vardex.embeddings import Embedder, nearest
from vardex.llm import Answer, Effort, Message, ModelClient, Request
from vardex.packs import describe_problem
from vardex.prompts import ANSWER, ANSWER_REPAIR, NO_ROWS, SQL_REPAIR, WRITE_SQL
from vardex.provenance import Figure, misspelt, trace
from vardex.semantic import Example, Knowledge, Questions
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
    warnings: list[str] = field(default_factory=list)  # such as a name spelt unlike the rows


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


def choose_examples(
    questions: Questions, question: str, embedder: Embedder | None
) -> list[Example]:
    """The worked examples to show for a question: all of them when the pool is no larger
    than show_examples; otherwise the ones whose questions mean the most like this one, in the
    pool's order, or the first ones when the pack asks for them, or there is no embedder."""
    pool = questions.examples
    n = questions.show_examples or len(pool)
    if n >= len(pool):
        return pool
    if questions.choose_examples == "first" or embedder is None:
        return pool[:n]
    # Two questions are compared as two questions, so both are embedded as queries
    vectors = [embedder.embed_query(" ".join(example.question.split())) for example in pool]
    chosen = nearest(embedder.embed_query(question), vectors, n)
    return [pool[i] for i in sorted(chosen)]


def describe_glossary(knowledge: Knowledge) -> str:
    """The pack's own words, and what each means in the tables."""
    glossary = knowledge.questions.glossary
    if not glossary:
        return ""
    terms = "\n".join(
        f"- {term}: {' '.join(meaning.split())}" for term, meaning in glossary.items()
    )
    return f"<glossary>\n{terms}\n</glossary>"


def describe_examples(knowledge: Knowledge, dialect: str, examples: list[Example]) -> str:
    """Worked examples, each a question and its SQL in the warehouse's dialect."""
    parts = [
        f"<example>\n<question>{' '.join(example.question.split())}</question>\n"
        f"<sql>\n{transpile(example.sql.strip(), knowledge.questions.dialect, dialect)}\n</sql>\n"
        "</example>"
        for example in examples
    ]
    return "<examples>\n" + "\n".join(parts) + "\n</examples>" if parts else ""


def system_prompt(knowledge: Knowledge, warehouse: Warehouse, today: date, max_rows: int) -> str:
    """The engine's text-to-SQL prompt, with the pack's semantic layer filled in: the same for
    every question, so the prompt cache can keep it."""
    context = "\n\n".join(
        part
        for part in (
            describe_tables(knowledge, warehouse.tables()),
            describe_ontology(knowledge),
            describe_values(knowledge, warehouse),
            describe_metrics(knowledge, warehouse.dialect),
            describe_glossary(knowledge),
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
    earlier: list[Message] | None = None,
    embedder: Embedder | None = None,
) -> Request:
    """The request that asks a model to write SQL for one question, after any earlier turns:
    a question the model asked back, and the person's answer to it. The worked examples are
    chosen for the question the conversation began with."""
    turns = [*(earlier or []), Message("user", question)]
    examples = choose_examples(knowledge.questions, turns[0].text, embedder)
    return Request(
        system=system_prompt(knowledge, warehouse, today or date.today(), max_rows),
        messages=turns,
        effort=effort,
        output_schema=Plan,
        cache_system=True,  # the same tables and metrics for every question
        # The examples differ from question to question, so they come after the cached part
        system_tail=describe_examples(knowledge, warehouse.dialect, examples),
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


def with_reply(request: Request, said: str, reply: str) -> Request:
    """The request with the model's last answer and our reply to it added."""
    messages = [*request.messages, Message("assistant", said), Message("user", reply)]
    return replace(request, messages=messages)


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
    model what went wrong and ask again, up to `repairs` times. A query that returns no rows
    is shown to the model once, in case a filter is spelt differently in the data."""
    answers: list[Answer] = []
    allowed = set(knowledge.tables)
    failures = 0
    looked = False
    while True:
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
                    result = warehouse.run(plan.sql or "", max_rows=max_rows)
                except QueryError as err:
                    problems = [f"the warehouse refused it: {err}"]
                else:
                    if result.rows or looked:
                        return plan, result, answers
                    looked = True
                    request = with_reply(request, answer.text, NO_ROWS)
                    continue
        failures += 1
        if failures > repairs:
            raise AnswerError("no query that could run: " + "; ".join(problems), answers)
        problems_text = "\n".join(f"- {p}" for p in problems)
        request = with_reply(request, answer.text, SQL_REPAIR.format(problems=problems_text))


def answer_prompt(style: list[str]) -> str:
    """The engine's answer prompt, with the pack's own rules for writing answers after it."""
    if not style:
        return ANSWER
    return ANSWER + "\n\n<style>\n" + "\n".join(f"- {rule}" for rule in style) + "\n</style>"


def answer_from_rows(
    client: ModelClient,
    question: str,
    plan: Plan,
    result: Result,
    *,
    repairs: int,
    style: list[str] | None = None,
) -> tuple[str, list[Figure], list[Answer]]:
    """Ask the model to answer from the rows, and trace every number in what it writes."""
    assumptions = "\n".join(f"- {a}" for a in plan.assumptions) or "- none"
    request = Request(
        system=answer_prompt(style or []),
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
        figures = trace(answer.text, [result], question, plan.assumptions)
        untraced = [f.text for f in figures if not f.traced]
        if not untraced:
            return answer.text.strip(), figures, answers
        repair = ANSWER_REPAIR.format(figures=", ".join(untraced))
        request = with_reply(request, answer.text, repair)
    raise AnswerError(
        f"the answer has numbers no query returned: {', '.join(untraced)}", answers, result
    )


def names_in(knowledge: Knowledge, result: Result) -> list[str]:
    """The names in a result: the values of every column that is an object's label."""
    labels = {obj.label for obj in knowledge.ontology.objects if obj.label}
    columns = [i for i, column in enumerate(result.columns) if column in labels]
    return [row[i] for row in result.rows for i in columns if isinstance(row[i], str)]


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
    earlier: list[Message] | None = None,
    embedder: Embedder | None = None,
) -> Reply:
    """Answer one question from a warehouse: plan a query, run it, answer from its rows.
    `earlier` holds the turns before it, when the model asked a question back. With an
    embedder, the prompt shows the pack's examples most like the question."""
    request = build_request(
        knowledge,
        warehouse,
        question,
        effort=effort,
        max_rows=max_rows,
        today=today,
        earlier=earlier,
        embedder=embedder,
    )
    # What the person asked, in all their turns, for the answer and for tracing its numbers
    question = "\n".join([*(m.text for m in earlier or [] if m.role == "user"), question])
    plan, result, answers = plan_and_run(
        client, warehouse, knowledge, request, repairs=repairs, max_rows=max_rows
    )
    if result is None:
        kind: Kind = "clarify" if plan.action == "clarify" else "refuse"
        return Reply(kind, plan.message or "", plan.assumptions, None, [], answers)
    try:
        text, figures, more = answer_from_rows(
            client, question, plan, result, repairs=repairs, style=knowledge.questions.style
        )
    except AnswerError as err:
        raise AnswerError(str(err), answers + err.answers, result) from err
    warnings = [
        f'"{written}" is not in the rows; they spell it "{name}"'
        for written, name in misspelt(text, names_in(knowledge, result))
    ]
    return Reply("answer", text, plan.assumptions, result, figures, answers + more, warnings)
