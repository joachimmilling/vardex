"""Evals for documents: does search find the pages, and are answers faithful to them?"""

import json

import pytest
from fakes import FakeEmbedder, FakeModel, FakeReranker, MemoryIndex
from pydantic import ValidationError
from test_rag import PASSAGES, REPORT, reply

from vardex.documents import DocumentList
from vardex.evals import (
    CiteCase,
    CiteSuite,
    FaithfulnessJudge,
    SearchCase,
    SearchSuite,
    check_document_suite,
    cite_case,
    format_report,
    grade_behaviour,
    grade_citations,
    load_suites,
    run,
    search_case,
)


@pytest.fixture
def index():
    index = MemoryIndex()
    texts = [(p.page, p.text) for p in PASSAGES]
    index.put(REPORT, "f", texts, FakeEmbedder().embed_documents([t for _, t in texts]))
    return index


def search_suite(*cases, k=1):
    return SearchSuite(
        description="Search", capability="search", k=k, threshold=0.5, cases=list(cases)
    )


def test_a_search_case_passes_when_its_page_is_in_the_top_k(index):
    case = SearchCase(
        id="dividend", type="lookup", question="the board dividend",
        pages=["demo-laks-annual-report-2025:40"],
    )  # fmt: skip
    result = search_case(index, FakeEmbedder(), None, search_suite(case), case)
    assert result.passed
    assert result.outcome.values == {"rank": 1, "found_at": 1}


def test_one_page_that_answers_is_enough_and_none_fails_the_case(index):
    case = SearchCase(
        id="either", type="lookup", question="harvest volumes rose",
        pages=["demo-laks-annual-report-2025:40", "demo-laks-annual-report-2025:12"],
    )  # fmt: skip
    result = search_case(index, FakeEmbedder(), FakeReranker(), search_suite(case), case)
    assert result.passed and result.outcome.values == {"rank": 1, "found_at": 1}


def test_a_missed_page_says_how_far_down_it_was(index):
    lower = SearchCase(
        id="lower", type="lookup", question="harvest volumes rose",
        pages=["demo-laks-annual-report-2025:40"],
    )  # fmt: skip
    result = search_case(index, FakeEmbedder(), FakeReranker(), search_suite(lower), lower)
    assert not result.passed
    assert result.outcome.values == {"rank": None, "found_at": 2}
    assert result.grades[0].reason == (
        "found at rank 2, below the top 1: got demo-laks-annual-report-2025:12"
    )
    never = lower.model_copy(update={"pages": ["demo-laks-annual-report-2025:99"]})
    result = search_case(index, FakeEmbedder(), FakeReranker(), search_suite(never), never)
    assert result.grades[0].reason.startswith("not among the 2 passages ranked")


def test_the_report_gives_recall_and_mean_reciprocal_rank(index):
    cases = [
        SearchCase(id=f"c{n}", type="lookup", question=q, pages=[page])
        for n, (q, page) in enumerate(
            [
                ("harvest volumes", "demo-laks-annual-report-2025:12"),
                ("harvest volumes", "demo-laks-annual-report-2025:40"),  # found second
            ]
        )
    ]
    suite = search_suite(*cases, k=2)
    report = run(
        "search",
        suite,
        "fake-embedder",
        lambda c: search_case(index, FakeEmbedder(), None, suite, c),
    )
    text = format_report(report)
    assert "all                2       2   100%" in text  # the share found: recall at 2
    assert "Mean reciprocal rank: 0.75" in text
    assert "Missed: 0 ranked lower, 0 not ranked at all" in text


def test_a_case_names_pages_of_documents_in_the_pack():
    documents = DocumentList(about="Company", documents=[REPORT])
    case = SearchCase(id="x", type="lookup", question="q", pages=["unknown-report:3"])
    assert check_document_suite(search_suite(case), documents) == [
        "case x: no document unknown-report"
    ]


def test_a_cite_case_expecting_an_answer_needs_pages():
    with pytest.raises(ValidationError, match="needs pages"):
        CiteCase(id="x", type="lookup", question="q")
    CiteCase(id="x", type="refuse", question="q", expect="not_found")


def verdict(passed):
    return json.dumps({"reasoning": "Every claim is in passage 1.", "passed": passed})


def test_a_cited_answer_is_graded_by_its_pages_and_the_judge(index):
    case = CiteCase(
        id="dividend", type="lookup", question="What dividend does the board propose?",
        pages=["demo-laks-annual-report-2025:40"],
    )  # fmt: skip
    model = FakeModel(reply(True, "NOK 4.20 per share [1].", [(1, "NOK 4.20")]), verdict(True))
    graders = [grade_behaviour, grade_citations, FaithfulnessJudge(model)]
    result = cite_case(model, index, FakeEmbedder(), None, case, graders)
    assert [(g.grader, g.passed) for g in result.grades] == [
        ("behaviour", True), ("citations", True), ("faithful", True),
    ]  # fmt: skip
    judged = model.requests[1].messages[0].text
    assert "NOK 4.20 per share [1]." in judged
    assert "dividend of NOK 4.20" in judged  # the judge reads the passage cited


def test_citing_the_wrong_page_fails_even_when_the_judge_is_satisfied(index):
    case = CiteCase(
        id="harvest", type="lookup", question="What dividend does the board propose?",
        pages=["demo-laks-annual-report-2025:12"],
    )  # fmt: skip
    model = FakeModel(reply(True, "NOK 4.20 per share [1].", [(1, "NOK 4.20")]), verdict(True))
    graders = [grade_behaviour, grade_citations, FaithfulnessJudge(model)]
    result = cite_case(model, index, FakeEmbedder(), None, case, graders)
    assert not result.passed
    assert result.grades[1].reason == (
        "cites demo-laks-annual-report-2025:40, not one of demo-laks-annual-report-2025:12"
    )


def test_a_question_the_documents_cannot_answer_must_be_said_so(index):
    case = CiteCase(
        id="salary", type="refuse", question="What does the CEO earn?", expect="not_found"
    )
    model = FakeModel(reply(False, "The passages do not say."))
    result = cite_case(model, index, FakeEmbedder(), None, case, [grade_behaviour, grade_citations])
    assert result.passed
    assert len(model.requests) == 1  # the judge has nothing to read


def test_suites_are_read_as_their_capability_says(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "search.yaml").write_text(
        "description: d\ncapability: search\nthreshold: 0.8\ncases:\n"
        "  - {id: a, type: t, question: q, pages: ['r:1']}\n"
    )
    (tmp_path / "evals" / "cite.yaml").write_text(
        "description: d\ncapability: cite\nthreshold: 0.8\ncases:\n"
        "  - {id: a, type: t, question: q, expect: not_found}\n"
    )
    suites = load_suites(tmp_path)
    assert isinstance(suites["search"], SearchSuite)
    assert isinstance(suites["cite"], CiteSuite)
