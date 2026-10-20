"""Answers from documents: a question in; an answer out, every claim cited down to the page.

Retrieval-augmented generation, in three steps: search the pack's documents for the passages
most likely to answer the question, give the model those passages and nothing else, and check
what it writes. Every sentence that states a fact cites passages by number; every cited
passage comes with the words it was cited for, which must be in it, character for character;
and every number in the answer must be in those words or in the question. An answer that
fails a check is sent back once, and is not shown when it fails again.
"""

import re
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from vardex.embeddings import Embedder, Reranker
from vardex.extraction import squeeze
from vardex.llm import Answer, Effort, Message, ModelClient, Request
from vardex.packs import describe_problem
from vardex.prompts import CITE, CITE_REPAIR
from vardex.provenance import Figure, Quote, trace_quotes
from vardex.retrieval import Filters, Index, Passage, Search, search

MARKER = re.compile(r"\[(\d+)\]")  # a citation in the text, such as [2]
Kind = Literal["answer", "not_found"]


class Cited(BaseModel):
    """The words a passage is cited for."""

    passage: int = Field(description="The number of the passage.")
    words: str = Field(description="The words that support the answer, copied exactly.")


class CitedAnswer(BaseModel):
    """What the model must return: whether the passages answer the question, the answer with
    its citation marks, and the words each cited passage is cited for."""

    found: bool = Field(description="True when the passages answer the question.")
    answer: str = Field(
        description="The answer, with [n] after each sentence that states a fact; or, when "
        "found is false, what the passages lack."
    )
    citations: list[Cited] = Field(description="One entry per passage cited in the answer.")


@dataclass(frozen=True)
class Citation:
    """A passage an answer cites, by its number in the answer, and the words cited."""

    number: int  # as in [2] in the answer
    quote: Quote


@dataclass(frozen=True)
class Reply:
    """An answer from documents, with everything it rests on."""

    kind: Kind
    text: str  # the answer, with its citation marks, or what the passages lack
    citations: list[Citation]  # in the order the answer first cites them
    figures: list[Figure]  # every number in the answer, and the quotes that hold it
    search: Search  # the passages the model was given
    answers: list[Answer]


class CiteError(Exception):
    """Raised when a question gets no answer that passes the checks. The message says why."""

    def __init__(self, reason: str, answers: list[Answer], found: Search | None = None) -> None:
        super().__init__(reason)
        self.answers = answers  # so the calls can still be logged and paid for
        self.search = found


def passages_text(passages: list[Passage]) -> str:
    """The passages as the model sees them: numbered, each with its document and page."""
    parts = [
        f'<passage number="{n}" document="{p.document.title}" page="{p.page}">\n'
        f"{p.text}\n</passage>"
        for n, p in enumerate(passages, start=1)
    ]
    return "<passages>\n" + "\n".join(parts) + "\n</passages>"


def build_request(question: str, passages: list[Passage], effort: Effort | None) -> Request:
    """The request that asks a model to answer a question from these passages only."""
    return Request(
        system=CITE,
        messages=[Message("user", f"{passages_text(passages)}\n\n<question>{question}</question>")],
        effort=effort,
        output_schema=CitedAnswer,
    )


def check(
    cited: CitedAnswer, passages: list[Passage], question: str
) -> tuple[list[str], list[Citation], list[Figure]]:
    """Everything wrong with an answer, as readable sentences, with its quotes and figures.
    No problems means every claim is cited, every quote is in its passage, and every number is
    in a quote or the question."""
    if not cited.found:
        return [], [], []
    problems = []
    marks = [int(n) for n in MARKER.findall(cited.answer)]
    if not marks:
        problems.append("the answer cites no passage: put [n] after each sentence")
    words = {}
    for citation in cited.citations:
        if not 1 <= citation.passage <= len(passages):
            problems.append(f"there is no passage {citation.passage}")
        elif squeeze(citation.words) not in squeeze(passages[citation.passage - 1].text):
            problems.append(f"passage {citation.passage} does not say {citation.words!r}")
        else:
            words.setdefault(citation.passage, []).append(citation.words)
    for n in dict.fromkeys(marks):
        if not 1 <= n <= len(passages):
            problems.append(f"[{n}] cites no passage: there are {len(passages)}")
        elif n not in {c.passage for c in cited.citations}:
            problems.append(f"[{n}] has no words copied from passage {n}")
    citations = [
        Citation(n, Quote(p.document.id, p.document.title, p.page, text, p.link))
        for n in dict.fromkeys(marks)
        if n in words
        for p in [passages[n - 1]]
        for text in words[n]
    ]
    quotes = [citation.quote for citation in citations]
    # The cited documents' titles count as part of the question: "2025" in the title of
    # the annual report for 2025 needs no quote.
    titles = " ".join(dict.fromkeys(q.title for q in quotes))
    figures = trace_quotes(MARKER.sub("", cited.answer), quotes, f"{question}\n{titles}")
    untraced = [f.text for f in figures if not f.traced]
    if untraced:
        problems.append(f"these numbers are in none of the words you copied: {', '.join(untraced)}")
    return problems, citations, figures


def answer(
    client: ModelClient,
    index: Index,
    embedder: Embedder,
    question: str,
    filters: Filters | None = None,
    *,
    reranker: Reranker | None = None,
    passages: int = 8,
    effort: Effort | None = "low",
    repairs: int = 1,
) -> Reply:
    """Answer a question from the pack's documents: search, answer from the passages found,
    and check every citation and number."""
    found = search(index, embedder, question, filters, limit=passages, reranker=reranker)
    if not found.passages:  # nothing to read: no model call, and nothing to cite
        return Reply(
            "not_found", "No passage matches the question and its filters.", [], [], found, []
        )
    request = build_request(question, found.passages, effort)
    answers: list[Answer] = []
    for _ in range(repairs + 1):
        answer = client.send(request)
        answers.append(answer)
        if answer.stop_reason != "end":
            raise CiteError(f"the model stopped early ({answer.stop_reason})", answers, found)
        try:
            cited = CitedAnswer.model_validate_json(answer.text)
        except ValidationError as err:
            problems = [describe_problem(e) for e in err.errors()]
        else:
            problems, citations, figures = check(cited, found.passages, question)
            if not problems:
                kind: Kind = "answer" if cited.found else "not_found"
                return Reply(kind, cited.answer.strip(), citations, figures, found, answers)
        repair = CITE_REPAIR.format(problems="\n".join(f"- {p}" for p in problems))
        request = replace(
            request,
            messages=[
                *request.messages,
                Message("assistant", answer.text),
                Message("user", repair),
            ],
        )
    raise CiteError("no answer passed the checks: " + "; ".join(problems), answers, found)
