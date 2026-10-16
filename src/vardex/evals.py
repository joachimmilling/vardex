"""Evals: how often a capability gets it right, measured on a pack's golden set.

A pack keeps its golden sets in evals/<suite>.yaml: questions a person might ask, each with
the answer a person checked, or what should happen instead. The engine runs every case
through the capability, grades what came back, and reports the share that passed, with the
uncertainty a small sample brings.

A grader is any function that takes a case, what the capability did, and the reference, and
returns a Grade, or None when it does not apply. Most are code; the judge is a model.
"""

import itertools
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vardex import text_to_sql
from vardex.llm import Answer, Effort, Ledger, Message, ModelClient, ModelError, Request
from vardex.packs import Language, PackError, read_yaml
from vardex.prompts import JUDGE
from vardex.semantic import Knowledge
from vardex.sql import QueryError, Result, Warehouse, check_sql, is_number, transpile

Expect = Literal["answer", "clarify", "refuse"]
Kind = Literal["answer", "clarify", "refuse", "error"]
Z = 1.96  # for a 95% interval
SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"


class Case(BaseModel):
    """One question in a golden set, and what a right answer looks like."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SLUG)
    type: str = Field(pattern=SLUG)  # for the report: accuracy by type
    language: Language = "en"
    question: str
    expect: Expect = "answer"
    sql: str | None = None  # with expect: answer, the query whose rows a right answer gives
    ordered: bool = False  # the rows must come in the same order, as in a top five
    critical: bool = False  # the suite fails when this case fails, whatever the rate

    @model_validator(mode="after")
    def sql_with_answers_only(self) -> "Case":
        if (self.expect == "answer") != (self.sql is not None):
            raise ValueError("a case expecting an answer needs sql, and no other case takes it")
        return self


class Suite(BaseModel):
    """The contents of evals/<suite>.yaml in a pack."""

    model_config = ConfigDict(extra="forbid")

    description: str
    capability: Literal["answer"]
    today: date  # the date every run pretends it is, so "last year" never moves
    threshold: float = Field(ge=0, le=1)  # the share of cases that must pass
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "Suite":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every case needs its own id")
        return self


@dataclass(frozen=True)
class Outcome:
    """What the capability did with one case."""

    kind: Kind
    text: str  # the answer, the question back, the reason for refusing, or the error
    result: Result | None  # the query that ran, if any
    answers: list[Answer]


@dataclass(frozen=True)
class Grade:
    grader: str
    passed: bool
    reason: str = ""  # why it failed
    answers: list[Answer] = field(default_factory=list)  # a model grader's own calls


Grader = Callable[[Case, Outcome, Result | None], Grade | None]


@dataclass(frozen=True)
class CaseResult:
    case: Case
    outcome: Outcome
    grades: list[Grade]

    @property
    def passed(self) -> bool:
        return all(grade.passed for grade in self.grades)

    @property
    def answers(self) -> list[Answer]:
        """Every model call the case took, the graders' included."""
        return self.outcome.answers + [a for grade in self.grades for a in grade.answers]


@dataclass(frozen=True)
class Tally:
    """How many cases in a group passed, and how sure that makes us."""

    group: str
    cases: int
    passed: int

    @property
    def rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.passed, self.cases)


@dataclass(frozen=True)
class Report:
    suite: str
    model: str
    results: list[CaseResult]

    def by(self, key: Callable[[Case], str]) -> list[Tally]:
        """One tally per group of cases, in the order the groups first appear."""
        groups: dict[str, list[CaseResult]] = {}
        for result in self.results:
            groups.setdefault(key(result.case), []).append(result)
        return [tally(name, results) for name, results in groups.items()]

    @property
    def total(self) -> Tally:
        return tally("all", self.results)

    @property
    def failed(self) -> list[CaseResult]:
        return [result for result in self.results if not result.passed]

    @property
    def cost_usd(self) -> Decimal | None:
        """What the run cost, graders included, or None if any call had no price."""
        costs = [answer.cost_usd for result in self.results for answer in result.answers]
        return None if None in costs else sum(costs, Decimal(0))

    def gate(self, threshold: float) -> list[str]:
        """Why the run fails its suite's gate: too few cases passed, or a critical case
        failed. Empty means it passes."""
        problems = []
        if self.total.rate < threshold:
            problems.append(f"{self.total.rate:.0%} passed, below the threshold of {threshold:.0%}")
        problems += [f"critical case {r.case.id} failed" for r in self.failed if r.case.critical]
        return problems


def tally(group: str, results: list[CaseResult]) -> Tally:
    return Tally(group, len(results), sum(result.passed for result in results))


def wilson(passed: int, cases: int, z: float = Z) -> tuple[float, float]:
    """The Wilson score interval for a share: where the true pass rate probably lies, given
    how many of a small sample passed. Unlike passed ± 2 standard errors, it never goes below
    0 or above 1, and it stays honest for 5 of 5."""
    if cases == 0:
        return 0.0, 1.0
    share = passed / cases
    centre = (share + z * z / (2 * cases)) / (1 + z * z / cases)
    spread = z * math.sqrt(share * (1 - share) / cases + z * z / (4 * cases * cases))
    spread /= 1 + z * z / cases
    return max(0.0, centre - spread), min(1.0, centre + spread)


def load_suites(pack: Path) -> dict[str, Suite]:
    """Every suite in the pack's evals/ folder, by name."""
    paths = sorted((pack / "evals").glob("*.yaml"))
    return {path.stem: read_yaml(path, Suite, "eval suite") for path in paths}


def recording_path(pack: Path, suite: str) -> Path:
    return pack / "evals" / "recordings" / f"{suite}.jsonl"


def words(text: str) -> str:
    """A question reduced to its words, so a changed comma does not hide a copy."""
    return " ".join(re.findall(r"\w+", text.lower()))


def check_suite(suite: Suite, knowledge: Knowledge) -> list[str]:
    """Everything that makes a suite unfit to measure with: a case the model is shown as an
    example, or reference SQL that would not pass Vardex's own checks."""
    examples = {words(example.question) for example in knowledge.questions.examples}
    problems = []
    for case in suite.cases:
        if words(case.question) in examples:
            problems.append(f"case {case.id}: the question is also an example in questions.yaml")
        if case.sql:
            for problem in check_sql(case.sql, knowledge.questions.dialect, set(knowledge.tables)):
                problems.append(f"case {case.id}: {problem}")
    return problems


def normalise(value: Any) -> Any:
    """A value as it is compared: a whole number exactly, any other number to six significant
    digits, so 0.156199 from one query equals 0.15619915 from another; text without spaces at
    either end."""
    if is_number(value):
        number = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
        if number.is_finite() and number == number.to_integral_value():
            return str(int(number))
        return f"{number:.6g}"
    return value.strip() if isinstance(value, str) else value


def compare(reference: Result, result: Result, ordered: bool) -> str | None:
    """Why a result does not hold the reference's rows, or None when it does. The result may
    have more columns than the reference, in any order and under any names, and more rows: a
    query may return context, such as the next nine groups after the one asked for. Every
    reference row must be in it, and with ordered, as its first rows, in the same order."""
    if len(result.rows) < len(reference.rows):
        return f"{len(result.rows)} rows, expected at least {len(reference.rows)}"
    want = [tuple(normalise(v) for v in row) for row in reference.rows]
    got = [tuple(normalise(v) for v in row) for row in result.rows]

    def holds(j: int, i: int) -> bool:
        """True when result column j holds every value of reference column i."""
        return not Counter(row[i] for row in want) - Counter(row[j] for row in got)

    # For each reference column, the result columns that hold all its values
    columns = range(len(result.columns))
    candidates = [[j for j in columns if holds(j, i)] for i in range(len(reference.columns))]
    missing = [reference.columns[i] for i, found in enumerate(candidates) if not found]
    if missing:
        return f"no column holds the values of {', '.join(missing)}"
    in_rows = False
    for choice in itertools.product(*candidates):
        rows = [tuple(row[j] for j in choice) for row in got]
        if ordered and rows[: len(want)] == want:
            return None
        if not Counter(want) - Counter(rows):
            if not ordered:
                return None
            in_rows = True
    if in_rows:
        return "the right rows, in the wrong order"
    return "the right values, but not in the same rows"


def grade_behaviour(case: Case, outcome: Outcome, reference: Result | None) -> Grade:
    """Did it do what the case expects: answer, ask back or refuse?"""
    if outcome.kind == case.expect:
        return Grade("behaviour", True)
    got = "no answer" if outcome.kind == "error" else f"{outcome.kind}"
    return Grade("behaviour", False, f"expected {case.expect}, got {got}: {outcome.text}")


def grade_rows(case: Case, outcome: Outcome, reference: Result | None) -> Grade | None:
    """Did the query it ran return the reference rows?"""
    if reference is None or outcome.kind != "answer" or outcome.result is None:
        return None
    reason = compare(reference, outcome.result, case.ordered)
    return Grade("rows", reason is None, reason or "")


class Verdict(BaseModel):
    """What a judge returns: its reasoning first, then whether the answer passes."""

    reasoning: str = Field(description="What the answer gets right or wrong, in a sentence or two.")
    passed: bool = Field(description="True when the answer is right.")


class Judge:
    """A grader that asks a model whether an answer's words agree with the reference rows."""

    def __init__(self, client: ModelClient, effort: Effort | None = "low") -> None:
        self.client = client
        self.effort = effort

    def __call__(self, case: Case, outcome: Outcome, reference: Result | None) -> Grade | None:
        if reference is None or outcome.kind != "answer":
            return None
        answer = self.client.send(self.request(case, outcome.text, reference))
        try:
            verdict = Verdict.model_validate_json(answer.text)
        except ValueError:
            return Grade("judge", False, "the judge's verdict did not parse", [answer])
        return Grade("judge", verdict.passed, "" if verdict.passed else verdict.reasoning, [answer])

    def request(self, case: Case, text: str, reference: Result) -> Request:
        message = (
            f"<question>{case.question}</question>\n"
            f"<reference_sql>\n{reference.sql}\n</reference_sql>\n"
            f"<reference_rows>\n{text_to_sql.rows_as_text(reference)}\n</reference_rows>\n"
            f"<answer>\n{text}\n</answer>"
        )
        return Request(
            system=JUDGE,
            messages=[Message("user", message)],
            effort=self.effort,
            output_schema=Verdict,
            cache_system=True,
        )


def answer_case(
    client: ModelClient,
    warehouse: Warehouse,
    knowledge: Knowledge,
    suite: Suite,
    case: Case,
    graders: list[Grader],
    *,
    effort: Effort | None = "medium",
) -> CaseResult:
    """Run one case through text-to-SQL, exactly as vardex answer would, and grade it."""
    reference = None
    if case.sql:
        sql = transpile(case.sql, knowledge.questions.dialect, warehouse.dialect)
        try:
            reference = warehouse.run(sql, max_rows=100)
        except QueryError as err:
            raise PackError(f"case {case.id}: the reference SQL fails: {err}") from err
        if reference.truncated:
            raise PackError(f"case {case.id}: the reference SQL returns more than 100 rows")
    ledger = Ledger(client)
    try:
        reply = text_to_sql.answer(
            ledger, warehouse, knowledge, case.question, effort=effort, today=suite.today
        )
        outcome = Outcome(reply.kind, reply.text, reply.result, reply.answers)
    except text_to_sql.AnswerError as err:
        outcome = Outcome("error", str(err), err.result, err.answers)
    except ModelError as err:
        if not err.retryable:
            raise  # every case would fail the same way, so stop the run
        outcome = Outcome("error", str(err), None, ledger.answers)
    grades = [grade for grader in graders if (grade := grader(case, outcome, reference))]
    return CaseResult(case, outcome, grades)


def run(
    name: str,
    suite: Suite,
    model: str,
    run_case: Callable[[Case], CaseResult],
    *,
    only: list[str] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Report:
    """Run a suite's cases, or only those named, one after another: in order, so a recording
    replays in the same order and the prompt cache stays warm."""
    if only:
        unknown = sorted(set(only) - {case.id for case in suite.cases})
        if unknown:
            raise PackError(f"no case {', '.join(unknown)} in suite {name}")
    results = []
    for case in suite.cases:
        if only and case.id not in only:
            continue
        result = run_case(case)
        results.append(result)
        if on_result:
            on_result(result)
    return Report(name, model, results)


def format_report(report: Report, details: bool = False) -> str:
    """The report as text: the rate by type, by language and in all, then every failure. With
    details, each failure also shows its question, the SQL that ran and the answer."""

    def table(title: str, tallies: list[Tally]) -> list[str]:
        w = max([14, *(len(t.group) + 2 for t in tallies)])  # the first column's width
        lines = [f"{title:<{w}}{'cases':>6}{'passed':>8}{'rate':>7}   95% interval"]
        for t in tallies:
            low, high = t.interval
            lines.append(
                f"{t.group:<{w}}{t.cases:>6}{t.passed:>8}{t.rate:>7.0%}   {low:.0%} to {high:.0%}"
            )
        return lines

    lines = table("type", [*report.by(lambda c: c.type), report.total])
    lines += ["", *table("language", report.by(lambda c: c.language))]
    # An answer takes two calls, a plan and the words; more means a check sent it back
    answered = [r for r in report.results if r.outcome.kind == "answer"]
    again = [r for r in answered if len(r.outcome.answers) > 2]
    if answered:
        lines += [
            "",
            f"Asked again: {len(again)} of {len(answered)} answers took more than two calls",
        ]
    for result in again if details else []:
        before = " ".join(result.outcome.answers[-2].text.split())
        lines.append(f"  {result.case.id}: before: {before}")
    if report.failed:
        lines += ["", "Failed"]
    for result in report.failed:
        lines.append(f"  {result.case.id}")
        for grade in result.grades:
            if not grade.passed:
                lines.append(f"    {grade.grader}: {' '.join(grade.reason.split())}".rstrip())
        if details:
            lines.append(f"    question: {' '.join(result.case.question.split())}")
            if result.outcome.result:
                sql = result.outcome.result.sql.strip().replace("\n", "\n      ")
                lines.append(f"    sql: {sql}")
            lines.append(f"    reply: {' '.join(result.outcome.text.split())}")
    return "\n".join(lines)
