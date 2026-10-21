"""Evals: how often a capability gets it right, measured on a pack's golden set.

A pack keeps its golden sets in evals/<suite>.yaml: questions a person might ask, each with
the answer a person checked, or what should happen instead. The engine runs every case
through the capability, grades what came back, and reports the share that passed, with the
uncertainty a small sample brings.

A grader is any function that takes a case, what the capability did, and the reference, and
returns a Grade, or None when it does not apply. Most are code; the judge is a model.
"""

import itertools
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vardex import rag, text_to_sql
from vardex.documents import DocumentList
from vardex.embeddings import Embedder, Reranker
from vardex.extraction import ExtractionError, ExtractionSpec, extract
from vardex.llm import Answer, Effort, Ledger, Message, ModelClient, ModelError, Request
from vardex.packs import Language, PackError, read_yaml
from vardex.prompts import FAITHFUL, JUDGE
from vardex.retrieval import Filters, Index, Passage, search
from vardex.semantic import Knowledge
from vardex.sql import QueryError, Result, Warehouse, check_sql, is_number, transpile

Expect = Literal["answer", "clarify", "refuse"]
Kind = Literal["answer", "clarify", "refuse", "values", "problem", "passages", "not_found", "error"]
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


class ExtractionCase(BaseModel):
    """One document in a golden set for extraction, and what a right extraction gives."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SLUG)
    type: str = Field(pattern=SLUG)
    language: Language = "en"
    document: str  # a file in the pack's evals/documents/
    expect: Literal["values", "problem"] = "values"
    values: dict[str, Any] = {}  # the fields to grade; a field left out is not graded
    critical: bool = False

    @model_validator(mode="after")
    def values_with_values_only(self) -> "ExtractionCase":
        if (self.expect == "values") != bool(self.values):
            raise ValueError("a case expecting values needs values, and no other case takes them")
        return self

    @property
    def question(self) -> str:
        return f"extract from {self.document}"


class ExtractionSuite(BaseModel):
    """The contents of evals/<suite>.yaml in a pack, for an extraction."""

    model_config = ConfigDict(extra="forbid")

    description: str
    capability: Literal["extract"]
    extraction: str  # the name of an extraction in the pack's extractions/
    threshold: float = Field(ge=0, le=1)
    cases: list[ExtractionCase] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "ExtractionSuite":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every case needs its own id")
        return self


PAGE = r"^[a-z0-9]+(-[a-z0-9]+)*:[1-9][0-9]*$"  # a document's id and a page: report-2025:42
DEPTH = 50  # how far down a search suite looks for a page it missed: every passage ranked


class CaseFilters(BaseModel):
    """The filters a person would set for a question, such as the company it is about."""

    model_config = ConfigDict(extra="forbid")

    about: str | None = None
    kind: str | None = None
    period: str | None = None
    language: Language | None = None
    document: str | None = None


class SearchCase(BaseModel):
    """One question to search the pack's documents for, and the pages that answer it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SLUG)
    type: str = Field(pattern=SLUG)
    language: Language = "en"
    question: str
    pages: list[str] = Field(min_length=1)  # each as document-id:page; finding one is enough
    filters: CaseFilters = CaseFilters()
    critical: bool = False


class SearchSuite(BaseModel):
    """The contents of evals/<suite>.yaml in a pack, for searching documents."""

    model_config = ConfigDict(extra="forbid")

    description: str
    capability: Literal["search"]
    k: int = Field(default=10, ge=1)  # a case passes when one of its pages is in the top k
    threshold: float = Field(ge=0, le=1)
    cases: list[SearchCase] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "SearchSuite":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("every case needs its own id")
        return self


class CiteCase(BaseModel):
    """One question to answer from the pack's documents, and the pages a right answer cites;
    or expect: not_found for a question the documents do not answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SLUG)
    type: str = Field(pattern=SLUG)
    language: Language = "en"
    question: str
    expect: Literal["answer", "not_found"] = "answer"
    pages: list[str] = []  # each as document-id:page; a right answer cites one of them
    filters: CaseFilters = CaseFilters()
    critical: bool = False

    @model_validator(mode="after")
    def pages_with_answers_only(self) -> "CiteCase":
        if (self.expect == "answer") != bool(self.pages):
            raise ValueError("a case expecting an answer needs pages, and no other case takes them")
        return self


class CiteSuite(BaseModel):
    """The contents of evals/<suite>.yaml in a pack, for answers from documents."""

    model_config = ConfigDict(extra="forbid")

    description: str
    capability: Literal["cite"]
    threshold: float = Field(ge=0, le=1)
    cases: list[CiteCase] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "CiteSuite":
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
    values: dict[str, Any] | None = None  # what an extraction gave


@dataclass(frozen=True)
class Grade:
    grader: str
    passed: bool
    reason: str = ""  # why it failed
    answers: list[Answer] = field(default_factory=list)  # a model grader's own calls


AnyCase = Case | ExtractionCase | SearchCase | CiteCase
AnySuite = Suite | ExtractionSuite | SearchSuite | CiteSuite
CAPABILITIES: dict[str, type[BaseModel]] = {
    "answer": Suite, "extract": ExtractionSuite, "search": SearchSuite, "cite": CiteSuite,
}  # fmt: skip
Grader = Callable[[AnyCase, Outcome, Result | None], Grade | None]


@dataclass(frozen=True)
class CaseResult:
    case: AnyCase
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
    results: list[CaseResult]  # every run of every case, each case's runs together

    @property
    def cases(self) -> list[list[CaseResult]]:
        """The runs of each case, one list per case, in order."""
        runs: dict[str, list[CaseResult]] = {}
        for result in self.results:
            runs.setdefault(result.case.id, []).append(result)
        return list(runs.values())

    def by(self, key: Callable[[AnyCase], str]) -> list[Tally]:
        """One tally per group of cases, in the order the groups first appear."""
        groups: dict[str, list[list[CaseResult]]] = {}
        for runs in self.cases:
            groups.setdefault(key(runs[0].case), []).append(runs)
        return [tally(name, cases) for name, cases in groups.items()]

    @property
    def total(self) -> Tally:
        return tally("all", self.cases)

    @property
    def failed(self) -> list[list[CaseResult]]:
        """The cases that failed every run."""
        return [runs for runs in self.cases if not any(r.passed for r in runs)]

    @property
    def flaky(self) -> list[list[CaseResult]]:
        """The cases that passed some runs but not all."""
        return [runs for runs in self.cases if 0 < sum(r.passed for r in runs) < len(runs)]

    @property
    def cost_usd(self) -> Decimal | None:
        """What the run cost, graders included, or None if any call had no price."""
        costs = [answer.cost_usd for result in self.results for answer in result.answers]
        return None if None in costs else sum(costs, Decimal(0))

    def gate(self, threshold: float) -> list[str]:
        """Why the run fails its suite's gate: too few cases passed, or a critical case
        failed in any run. Empty means it passes."""
        problems = []
        if self.total.rate < threshold:
            problems.append(f"{self.total.rate:.0%} passed, below the threshold of {threshold:.0%}")
        problems += [
            f"critical case {runs[0].case.id} failed"
            for runs in self.cases
            if runs[0].case.critical and not all(r.passed for r in runs)
        ]
        return problems


def tally(group: str, cases: list[list[CaseResult]]) -> Tally:
    """A case passes only when every one of its runs passed."""
    return Tally(group, len(cases), sum(all(r.passed for r in runs) for runs in cases))


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


def load_suites(pack: Path) -> dict[str, AnySuite]:
    """Every suite in the pack's evals/ folder, by name, read as the kind of suite its
    capability names: answer, extract, search or cite."""
    suites: dict[str, AnySuite] = {}
    for path in sorted((pack / "evals").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        kind = CAPABILITIES.get(data.get("capability"), Suite)
        suites[path.stem] = read_yaml(path, kind, "eval suite")  # type: ignore[arg-type]
    return suites


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


def grade_behaviour(case: AnyCase, outcome: Outcome, reference: Result | None) -> Grade:
    """Did it do what the case expects: answer, ask back or refuse; give values or report a
    problem?"""
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
    embedder: Embedder | None = None,
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
            ledger,
            warehouse,
            knowledge,
            case.question,
            effort=effort,
            today=suite.today,
            embedder=embedder,
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


def same(got: Any, want: Any) -> bool:
    """An extracted value against the expected one: numbers as exact decimals, so 842.6 from
    YAML equals Decimal("842.6"); anything else as it is."""
    if is_number(got) and is_number(want):
        return Decimal(str(got)) == Decimal(str(want))
    return got == want


def grade_fields(case: AnyCase, outcome: Outcome, reference: Result | None) -> Grade | None:
    """Did the extraction give the expected value for every field the case names?"""
    if not isinstance(case, ExtractionCase) or outcome.values is None:
        return None
    wrong = [
        f"{name} is {outcome.values.get(name)!r}, expected {want!r}"
        for name, want in case.values.items()
        if not same(outcome.values.get(name), want)
    ]
    return Grade("fields", not wrong, "; ".join(wrong))


def check_extraction_suite(suite: ExtractionSuite, spec: ExtractionSpec, pack: Path) -> list[str]:
    """Everything that makes an extraction suite unfit: a missing document, a field the
    extraction does not have, or a document the model is shown as an example."""
    examples = {words(example.document) for example in spec.examples}
    problems = []
    for case in suite.cases:
        path = pack / "evals" / "documents" / case.document
        if not path.is_file():
            problems.append(f"case {case.id}: no document {path}")
        elif words(path.read_text(encoding="utf-8")) in examples:
            problems.append(f"case {case.id}: the document is also an example in the extraction")
        for name in sorted(set(case.values) - set(spec.fields)):
            problems.append(f"case {case.id}: the extraction has no field {name}")
    return problems


def extract_case(
    client: ModelClient,
    pack: Path,
    spec: ExtractionSpec,
    case: ExtractionCase,
    graders: list[Grader],
) -> CaseResult:
    """Run one document through the extraction, exactly as vardex extract would, and grade it."""
    document = (pack / "evals" / "documents" / case.document).read_text(encoding="utf-8")
    ledger = Ledger(client)
    try:
        extraction = extract(ledger, spec, document)
        text = json.dumps(extraction.values, default=str, ensure_ascii=False)
        outcome = Outcome("values", text, None, extraction.answers, extraction.values)
    except ExtractionError as err:
        outcome = Outcome("problem" if err.reported else "error", str(err), None, err.answers)
    except ModelError as err:
        if not err.retryable:
            raise  # every case would fail the same way, so stop the run
        outcome = Outcome("error", str(err), None, ledger.answers)
    grades = [grade for grader in graders if (grade := grader(case, outcome, None))]
    return CaseResult(case, outcome, grades)


def page_of(passage: Passage) -> str:
    """A passage's page, as a case names it: document-id:page."""
    return f"{passage.document.id}:{passage.page}"


def check_document_suite(suite: SearchSuite | CiteSuite, documents: DocumentList) -> list[str]:
    """Every page a case names whose document is not in the pack's list."""
    known = {document.id for document in documents.documents}
    problems = []
    for case in suite.cases:
        for page in case.pages:
            if page.split(":")[0] not in known:
                problems.append(f"case {case.id}: no document {page.split(':')[0]}")
            elif not re.match(PAGE, page):
                problems.append(f"case {case.id}: {page!r} is not document-id:page")
    return problems


def search_case(
    index: Index,
    embedder: Embedder,
    reranker: Reranker | None,
    suite: SearchSuite,
    case: SearchCase,
) -> CaseResult:
    """Search for one case's question, exactly as vardex search would, and grade it: is a
    page that answers it in the top k? Its rank is kept for the mean reciprocal rank. A page
    missed is looked for further down, among every passage the search ranked, to say how far
    it was."""
    filters = Filters(**case.filters.model_dump())
    found = search(index, embedder, case.question, filters, limit=DEPTH, reranker=reranker)
    pages = [page_of(passage) for passage in found.passages]
    first = next((n for n, page in enumerate(pages, start=1) if page in case.pages), None)
    rank = first if first and first <= suite.k else None
    top = ", ".join(dict.fromkeys(pages[: suite.k]))
    outcome = Outcome("passages", top, None, [], {"rank": rank, "found_at": first})
    if rank:
        reason = ""
    elif first:
        reason = f"found at rank {first}, below the top {suite.k}: got {top}"
    else:
        reason = f"not among the {len(pages)} passages ranked: got {top}"
    return CaseResult(case, outcome, [Grade("pages", rank is not None, reason)])


def grade_citations(case: AnyCase, outcome: Outcome, reference: Result | None) -> Grade | None:
    """Does the answer cite one of the pages that answer the question?"""
    if not isinstance(case, CiteCase) or outcome.kind != "answer" or outcome.values is None:
        return None
    cited = outcome.values["cited"]
    if set(cited) & set(case.pages):
        return Grade("citations", True)
    return Grade(
        "citations",
        False,
        f"cites {', '.join(cited) or 'nothing'}, not one of {', '.join(case.pages)}",
    )


class FaithfulnessJudge:
    """A grader that asks a model whether every claim in an answer is in the passages it
    cites. The references are the passages, not a person's answer."""

    def __init__(self, client: ModelClient, effort: Effort | None = "low") -> None:
        self.client = client
        self.effort = effort

    def __call__(self, case: AnyCase, outcome: Outcome, reference: Result | None) -> Grade | None:
        if not isinstance(case, CiteCase) or outcome.kind != "answer" or outcome.values is None:
            return None
        answer = self.client.send(self.request(case, outcome.text, outcome.values["passages"]))
        try:
            verdict = Verdict.model_validate_json(answer.text)
        except ValueError:
            return Grade("faithful", False, "the judge's verdict did not parse", [answer])
        reason = "" if verdict.passed else verdict.reasoning
        return Grade("faithful", verdict.passed, reason, [answer])

    def request(self, case: CiteCase, text: str, passages: list[Passage]) -> Request:
        message = (
            f"<question>{case.question}</question>\n{rag.passages_text(passages)}\n"
            f"<answer>\n{text}\n</answer>"
        )
        return Request(
            system=FAITHFUL,
            messages=[Message("user", message)],
            effort=self.effort,
            output_schema=Verdict,
            cache_system=True,
        )


def cite_case(
    client: ModelClient,
    index: Index,
    embedder: Embedder,
    reranker: Reranker | None,
    case: CiteCase,
    graders: list[Grader],
) -> CaseResult:
    """Answer one case from the documents, exactly as vardex cite would, and grade it."""
    ledger = Ledger(client)
    filters = Filters(**case.filters.model_dump())
    try:
        reply = rag.answer(ledger, index, embedder, case.question, filters, reranker=reranker)
    except rag.CiteError as err:
        outcome = Outcome("error", str(err), None, err.answers)
    except ModelError as err:
        if not err.retryable:
            raise  # every case would fail the same way, so stop the run
        outcome = Outcome("error", str(err), None, ledger.answers)
    else:
        pages = sorted({f"{c.quote.document}:{c.quote.page}" for c in reply.citations})
        numbers = {c.number for c in reply.citations}
        cited = [p for n, p in enumerate(reply.search.passages, start=1) if n in numbers]
        values = {"cited": pages, "passages": cited}
        outcome = Outcome(reply.kind, reply.text, None, reply.answers, values)
    grades = [grade for grader in graders if (grade := grader(case, outcome, None))]
    return CaseResult(case, outcome, grades)


def run(
    name: str,
    suite: AnySuite,
    model: str,
    run_case: Callable[[AnyCase], CaseResult],
    *,
    only: list[str] | None = None,
    repeat: int = 1,
    on_result: Callable[[CaseResult], None] | None = None,
) -> Report:
    """Run a suite's cases, or only those named, one after another, each `repeat` times in a
    row: in order, so a recording replays in the same order and the prompt cache stays warm."""
    if only:
        unknown = sorted(set(only) - {case.id for case in suite.cases})
        if unknown:
            raise PackError(f"no case {', '.join(unknown)} in suite {name}")
    results = []
    for case in suite.cases:
        if only and case.id not in only:
            continue
        for _ in range(repeat):
            result = run_case(case)
            results.append(result)
            if on_result:
                on_result(result)
    return Report(name, model, results)


def new_case_id(question: str, taken: set[str]) -> str:
    """An id for a case, from the first six words of its question, made unique: How many
    employees does Mowi ASA have? becomes how-many-employees-does-mowi-asa."""
    text = question.lower().translate(str.maketrans({"æ": "ae", "ø": "o", "å": "a"}))
    slug = "-".join(re.findall(r"[a-z0-9]+", text)[:6]) or "case"
    found, number = slug, 2
    while found in taken:
        found, number = f"{slug}-{number}", number + 1
    return found


def case_text(case: Case, indent: str = "  ") -> str:
    """A case as it is written in a suite file, under cases:, its dash after `indent`."""
    lines = [f"- id: {case.id}", f"  type: {case.type}"]
    if case.language != "en":
        lines.append(f"  language: {case.language}")
    lines.append(f"  question: {json.dumps(case.question, ensure_ascii=False)}")
    if case.expect != "answer":
        lines.append(f"  expect: {case.expect}")
    if case.ordered:
        lines.append("  ordered: true")
    if case.critical:
        lines.append("  critical: true")
    if case.sql:
        lines.append("  sql: |")
        lines += [f"    {line}".rstrip() for line in case.sql.strip().splitlines()]
    return "".join(f"{indent}{line}\n" for line in lines)


def add_case(path: Path, case: Case, knowledge: Knowledge) -> None:
    """Add a case to the end of a suite file, as text, so its comments stay. The file must
    still load and pass check_suite, or nothing is written."""
    text = path.read_text(encoding="utf-8").rstrip("\n")
    dash = re.search(r"^( *)- id:", text, re.MULTILINE)  # follow the file's own indentation
    text += "\n\n" + case_text(case, dash.group(1) if dash else "  ")
    try:
        suite = Suite.model_validate(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as err:
        raise PackError(f"the case does not fit {path}: {err}") from err
    problems = check_suite(suite, knowledge)
    if problems:
        raise PackError("; ".join(problems))
    path.write_text(text, encoding="utf-8")


def verdicts(report: Report) -> dict[str, dict[str, Any]]:
    """Each case's verdict, to keep with a recording: whether it passed, and every grade of
    every run."""
    return {
        runs[0].case.id: {
            "passed": all(result.passed for result in runs),
            "grades": [
                {"run": number, "grader": g.grader, "passed": g.passed, "reason": g.reason}
                for number, result in enumerate(runs, start=1)
                for g in result.grades
            ],
        }
        for runs in report.cases
    }


@dataclass(frozen=True)
class Comparison:
    """How a run differs from an earlier one, case by case."""

    broken: list[str]  # passed before, fail now
    fixed: list[str]  # failed before, pass now
    new: list[str]  # not in the earlier run
    removed: list[str]  # only in the earlier run

    @property
    def p_value(self) -> float:
        return mcnemar(len(self.broken), len(self.fixed))


def compare_runs(before: dict[str, bool], report: Report) -> Comparison:
    """The cases that changed between an earlier run's verdicts and this report."""
    now = {runs[0].case.id: all(r.passed for r in runs) for runs in report.cases}
    both = [case for case in now if case in before]
    return Comparison(
        broken=[case for case in both if before[case] and not now[case]],
        fixed=[case for case in both if not before[case] and now[case]],
        new=[case for case in now if case not in before],
        removed=[case for case in before if case not in now],
    )


def mcnemar(broken: int, fixed: int) -> float:
    """McNemar's exact test: the chance of a split at least this lopsided between cases that
    broke and cases that were fixed, if the change made no difference and each changed case
    went either way with even odds."""
    changed = broken + fixed
    if changed == 0:
        return 1.0
    tail = sum(math.comb(changed, k) for k in range(min(broken, fixed) + 1))
    return min(1.0, 2 * tail / 2**changed)


def format_comparison(comparison: Comparison) -> str:
    lines = []
    for title, cases in [
        ("Broken", comparison.broken),
        ("Fixed", comparison.fixed),
        ("New", comparison.new),
        ("Removed", comparison.removed),
    ]:
        lines.append(f"{title} since the baseline: {len(cases)}")
        lines += [f"  {case}" for case in cases]
    lines.append(f"McNemar's exact test: p = {comparison.p_value:.2g}")
    return "\n".join(lines)


def format_report(report: Report, details: bool = False) -> str:
    """The report as text: the rate by type, by language and in all, then every failure and
    every flaky case. With details, each failure also shows its question, the SQL that ran and
    the answer."""

    def table(title: str, tallies: list[Tally]) -> list[str]:
        w = max([14, *(len(t.group) + 2 for t in tallies)])  # the first column's width
        lines = [f"{title:<{w}}{'cases':>6}{'passed':>8}{'rate':>7}   95% interval"]
        for t in tallies:
            low, high = t.interval
            lines.append(
                f"{t.group:<{w}}{t.cases:>6}{t.passed:>8}{t.rate:>7.0%}   {low:.0%} to {high:.0%}"
            )
        return lines

    def failures(runs: list[CaseResult]) -> list[str]:
        lines = []
        for number, result in enumerate(runs, start=1):
            if result.passed:
                continue
            run = f"run {number}, " if len(runs) > 1 else ""
            for grade in result.grades:
                if not grade.passed:
                    reason = " ".join(grade.reason.split())
                    lines.append(f"    {run}{grade.grader}: {reason}".rstrip())
            if details:
                lines.append(f"    question: {' '.join(result.case.question.split())}")
                if result.outcome.result:
                    sql = result.outcome.result.sql.strip().replace("\n", "\n      ")
                    lines.append(f"    sql: {sql}")
                lines.append(f"    reply: {' '.join(result.outcome.text.split())}")
        return lines

    lines = table("type", [*report.by(lambda c: c.type), report.total])
    lines += ["", *table("language", report.by(lambda c: c.language))]
    found = [r.outcome.values for r in report.results if r.outcome.kind == "passages"]
    if found:  # a search suite: the share found is its recall, and how high it was found
        mrr = sum(1 / v["rank"] for v in found if v["rank"]) / len(found)
        lower = sum(1 for v in found if not v["rank"] and v.get("found_at"))
        never = sum(1 for v in found if not v.get("found_at"))
        lines += [
            "",
            f"Mean reciprocal rank: {mrr:.2f}",
            f"Missed: {lower} ranked lower, {never} not ranked at all",
        ]
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
    for runs in report.failed:
        lines += [f"  {runs[0].case.id}", *failures(runs)]
    if report.flaky:
        lines += ["", "Flaky"]
    for runs in report.flaky:
        passes = sum(result.passed for result in runs)
        lines += [f"  {runs[0].case.id}: passed {passes} of {len(runs)}", *failures(runs)]
    return "\n".join(lines)
