"""Capabilities, as the parts that choose between models see them.

Routing, shadow mode and the evals of a candidate model all ask the same three things of
whatever a model is doing, whether it answers a question from the warehouse, reads values from
an invoice or works through a task with tools: run it with this model; say whether the result
failed in a way the engine can see; and say how two results differ. Every capability already
checks its own output deterministically, each in its own way. This module gives those checks
one shape, so that a part that chooses models never has to know which capability it is
choosing for, and a new capability needs one adapter here rather than a change to each of them.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Protocol

from vardex import text_to_sql
from vardex.agent import MAX_STEPS, Run, run_agent
from vardex.embeddings import Embedder
from vardex.extraction import ExtractionError, ExtractionSpec, extract
from vardex.llm import Answer, Effort, Ledger, Message, ModelClient, ModelError
from vardex.semantic import Knowledge
from vardex.sql import Result, Warehouse, compare, same
from vardex.text_to_sql import AnswerError
from vardex.tools import Toolbox

# Two kinds every capability shares: its output failed its checks, after any repairs; or the
# model could not be reached at all. The others are the capability's own: answer, clarify and
# refuse; values and problem.
FAILED = "failed"
UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class Attempt:
    """What one model did with one piece of work."""

    model: str
    kind: str
    text: str  # what a person would read: the answer, the values, or why it failed
    output: Any  # what two attempts are compared on: rows, values
    detail: str  # the output in a few words, for a log: "12 rows", "7 fields"
    answers: list[Answer] = field(default_factory=list)
    reply: Any = None  # the capability's own result, for whoever shows it
    error: Exception | None = None  # what was raised instead, for whoever reports it

    @property
    def seconds(self) -> float:
        return round(sum(a.seconds for a in self.answers), 2)

    @property
    def cost_usd(self) -> Decimal | None:
        """What the calls cost, or None when one of them has no price."""
        costs = [a.cost_usd for a in self.answers]
        return None if None in costs else sum(costs, Decimal(0))  # type: ignore[arg-type]


class Capability(Protocol):
    """One piece of work, ready to be given to a model."""

    name: str
    # Kinds of result after which a larger model may still do better: a refusal, a problem
    # reported with the document. A question back is not among them: it is the person's.
    gives_up: tuple[str, ...]

    def attempt(self, client: ModelClient) -> Attempt:
        """Do the work with this model. A failed check or an unreachable model comes back as
        an attempt of that kind; only an error that every model would hit is raised."""
        ...

    def differ(self, first: Attempt, second: Attempt) -> str:
        """How two attempts differ, in a few words, or empty when they agree."""
        ...


def unreachable(client: ModelClient, err: ModelError, answers: list[Answer]) -> Attempt:
    """A model that could not be reached, or raise when no model could have been: a missing
    key or a refused request fails every attempt the same way."""
    if not err.retryable:
        raise err
    return Attempt(client.model, UNREACHABLE, str(err), None, "", answers, error=err)


@dataclass
class Answering:
    """A question answered from the warehouse: text-to-SQL, as vardex answer runs it."""

    warehouse: Warehouse
    knowledge: Knowledge
    question: str
    effort: Effort | None = "medium"
    repairs: int = 1
    max_rows: int = 100
    today: date | None = None
    earlier: list[Message] = field(default_factory=list)
    embedder: Embedder | None = None

    name = "answer"
    gives_up = ("refuse",)

    def attempt(self, client: ModelClient) -> Attempt:
        ledger = Ledger(client)
        try:
            reply = text_to_sql.answer(
                ledger,
                self.warehouse,
                self.knowledge,
                self.question,
                effort=self.effort,
                repairs=self.repairs,
                max_rows=self.max_rows,
                today=self.today,
                earlier=self.earlier,
                embedder=self.embedder,
            )
        except AnswerError as err:
            rows = rows_of(err.result)
            return Attempt(client.model, FAILED, str(err), err.result, rows, err.answers, error=err)
        except ModelError as err:
            return unreachable(client, err, ledger.answers)
        return Attempt(
            client.model,
            reply.kind,
            reply.text,
            reply.result,
            rows_of(reply.result),
            reply.answers,
            reply=reply,
        )

    def differ(self, first: Attempt, second: Attempt) -> str:
        """The same kind of reply, and for answers the same rows: as many of them, each in
        the other's result. One query may return a column more; neither may return a row
        more. The wording is not compared: two models never phrase an answer alike."""
        if first.kind != second.kind:
            return f"{first.kind} against {second.kind}"
        a, b = first.output, second.output
        if not (isinstance(a, Result) and isinstance(b, Result)):
            return ""
        if len(a.rows) != len(b.rows):
            return f"{len(a.rows)} rows against {len(b.rows)}"
        if compare(a, b, ordered=False) is None or compare(b, a, ordered=False) is None:
            return ""
        return compare(a, b, ordered=False) or "different rows"


def rows_of(result: Result | None) -> str:
    return f"{len(result.rows)} rows" if result else ""


@dataclass
class Extracting:
    """Values read from one document, as vardex extract reads them."""

    spec: ExtractionSpec
    document: str
    effort: Effort | None = "low"
    repairs: int = 1

    name = "extract"
    gives_up = ("problem",)  # the model said the document cannot be used

    def attempt(self, client: ModelClient) -> Attempt:
        ledger = Ledger(client)
        try:
            done = extract(
                ledger, self.spec, self.document, effort=self.effort, repairs=self.repairs
            )
        except ExtractionError as err:
            kind = "problem" if err.reported else FAILED
            return Attempt(client.model, kind, str(err), None, "", err.answers, error=err)
        except ModelError as err:
            return unreachable(client, err, ledger.answers)
        values = {name: done.values.get(name) for name in self.spec.fields}
        found = sum(value is not None for value in values.values())
        text = ", ".join(f"{name} {value}" for name, value in values.items() if value is not None)
        return Attempt(
            client.model,
            "values",
            text,
            values,
            f"{found} of {len(values)} fields",
            done.answers,
            reply=done,
        )

    def differ(self, first: Attempt, second: Attempt) -> str:
        """The same kind of result, and for values the same value in every field: numbers as
        exact decimals, so 842.6 and 842.60 agree. Quotes are not compared: two models may
        quote the same figure from different lines."""
        if first.kind != second.kind:
            return f"{first.kind} against {second.kind}"
        if not (isinstance(first.output, dict) and isinstance(second.output, dict)):
            return ""
        return "; ".join(
            f"{name}: {first.output.get(name)!r} against {second.output.get(name)!r}"
            for name in self.spec.fields
            if not same(first.output.get(name), second.output.get(name))
        )


@dataclass
class Acting:
    """A question worked through by the agent with the pack's tools, on a dry run.

    Only on a dry run: this is how a candidate model is tried on an agent's work, and a
    candidate that wrote to a CRM or drafted into the approval inbox would be acting, which is
    the one thing a candidate may not do. Two runs are compared on how they stopped and which
    tools they reached for, not on their wording."""

    toolbox: Toolbox
    question: str
    max_steps: int = MAX_STEPS
    window: int | None = None

    name = "agent"
    gives_up = ("steps", "truncated", "context")  # it ran out of room, not of answers

    def __post_init__(self) -> None:
        if not self.toolbox.dry_run:
            raise ValueError("an agent is only tried on a dry run")

    def attempt(self, client: ModelClient) -> Attempt:
        ledger = Ledger(client)
        try:
            run = run_agent(
                ledger, self.toolbox, self.question, max_steps=self.max_steps, window=self.window
            )
        except ModelError as err:
            return unreachable(client, err, ledger.answers)
        return ran(client.model, run)

    def differ(self, first: Attempt, second: Attempt) -> str:
        """The same stop, and the same tools, in whatever order and however often."""
        if first.kind != second.kind:
            return f"{first.kind} against {second.kind}"
        used, tried = set(first.output or []), set(second.output or [])
        if used != tried:
            names = [", ".join(sorted(tools)) or "no tools" for tools in (used, tried)]
            return f"{names[0]} against {names[1]}"
        return ""


def ran(model: str, run: Run) -> Attempt:
    """A run of the agent, made already, as an attempt: the live one, in shadow mode."""
    tools = [call.name for call, _ in run.tool_calls]
    return Attempt(
        model,
        run.stop,
        run.answer if run.answered else run.note,
        tools,
        f"{len(tools)} tool calls",
        [step.answer for step in run.steps],
        reply=run,
    )
