"""Answers from documents: cited, quoted, and every number traced to the words cited."""

import json

import pytest
from fakes import FakeEmbedder, FakeModel, MemoryIndex

from vardex.documents import DocumentSpec
from vardex.rag import CitedAnswer, CiteError, answer, check, passages_text
from vardex.retrieval import Filters, Passage

REPORT = DocumentSpec(
    id="demo-laks-annual-report-2025",
    title="Demo Laks AS Annual Report 2025",  # a made-up company
    about="999999999",
    kind="annual-report",
    period="2025",
    language="en",
    url="https://reports.demo.no/annual-2025.pdf",
)
PASSAGES = [
    Passage(1, REPORT, 12, "Harvest volumes rose to 61 400 tonnes, a record for the group."),
    Passage(2, REPORT, 40, "The board proposes a dividend of NOK 4.20 per share."),
]


def reply(found=True, text="", citations=()):
    return json.dumps(
        {
            "found": found,
            "answer": text,
            "citations": [{"passage": n, "words": w} for n, w in citations],
        }
    )


def checked(text, citations, question="How much salmon did Demo Laks harvest?"):
    cited = CitedAnswer.model_validate_json(reply(True, text, citations))
    return check(cited, PASSAGES, question)


def test_passages_are_numbered_with_their_document_and_page():
    text = passages_text(PASSAGES)
    assert '<passage number="2" document="Demo Laks AS Annual Report 2025" page="40">' in text


def test_a_cited_answer_passes_and_links_to_its_page():
    problems, citations, figures = checked(
        "Demo Laks harvested 61,400 tonnes in 2025 [1].", [(1, "rose to 61 400 tonnes")]
    )
    assert problems == []
    assert citations[0].number == 1
    assert citations[0].quote.link == "https://reports.demo.no/annual-2025.pdf#page=12"
    assert [f.text for f in figures if f.quotes] == ["61,400"]


def test_the_year_may_come_from_the_documents_title():
    problems, _, _ = checked("It harvested 61 400 tonnes in 2025 [1].", [(1, "61 400 tonnes")])
    assert problems == []


@pytest.mark.parametrize(
    "text, citations, problem",
    [
        ("Demo Laks harvested a record.", [], "cites no passage"),
        ("A record [3].", [(3, "a record")], "there is no passage 3"),
        ("A record [1].", [(1, "a record harvest")], "passage 1 does not say"),
        ("A record [1][2].", [(1, "a record")], "[2] has no words copied"),
        ("It harvested 62,000 tonnes [1].", [(1, "61 400 tonnes")], "62,000"),
        ("It paid NOK 4.20 [1].", [(1, "61 400 tonnes")], "4.20"),
    ],
)
def test_what_the_checks_send_back(text, citations, problem):
    problems, _, _ = checked(text, citations)
    assert any(problem in p for p in problems), problems


def test_not_found_needs_no_citations():
    cited = CitedAnswer.model_validate_json(reply(False, "The passages do not say."))
    assert check(cited, PASSAGES, "What is the CEO's salary?") == ([], [], [])


@pytest.fixture
def index():
    index = MemoryIndex()
    texts = [(p.page, p.text) for p in PASSAGES]
    index.put(REPORT, "f", texts, FakeEmbedder().embed_documents([t for _, t in texts]))
    return index


def test_an_answer_is_searched_for_written_and_checked(index):
    model = FakeModel(reply(True, "The board proposes NOK 4.20 per share [1].", [(1, "NOK 4.20")]))
    result = answer(model, index, FakeEmbedder(), "What dividend does the board propose?")
    assert result.kind == "answer"
    assert result.citations[0].quote.page == 40
    sent = model.requests[0]
    assert sent.output_schema is CitedAnswer
    assert "The board proposes a dividend of NOK 4.20 per share." in sent.messages[0].text


def test_a_failed_check_is_sent_back_once(index):
    model = FakeModel(
        reply(True, "NOK 4.50 per share [1].", [(1, "NOK 4.20")]),
        reply(True, "NOK 4.20 per share [1].", [(1, "NOK 4.20")]),
    )
    result = answer(model, index, FakeEmbedder(), "What dividend does the board propose?")
    assert result.text == "NOK 4.20 per share [1]."
    assert "4.50" in model.requests[1].messages[-1].text


def test_an_answer_that_fails_twice_is_not_shown(index):
    wrong = reply(True, "NOK 5 per share [1].", [(1, "NOK 4.20")])
    with pytest.raises(CiteError, match="no answer passed the checks") as err:
        answer(FakeModel(wrong, wrong), index, FakeEmbedder(), "Dividend?")
    assert len(err.value.answers) == 2


def test_a_question_the_documents_do_not_answer(index):
    model = FakeModel(reply(False, "The passages do not give the chief executive's salary."))
    result = answer(model, index, FakeEmbedder(), "What does the chief executive earn?")
    assert (result.kind, result.citations) == ("not_found", [])


def test_no_passage_to_read_needs_no_model(index):
    result = answer(FakeModel(), index, FakeEmbedder(), "Salmon?", Filters(about="000000000"))
    assert (result.kind, result.answers) == ("not_found", [])


def test_a_date_that_reads_as_no_number_is_traced_as_written():
    passages = [Passage(1, REPORT, 5, "The contract announced 18.04.18 was worth EUR 124 million.")]
    cited = CitedAnswer.model_validate_json(
        reply(True, "It was announced on 18.04.18 [1].", [(1, "announced 18.04.18")])
    )
    assert check(cited, passages, "When?")[0] == []
    wrong = CitedAnswer.model_validate_json(
        reply(True, "It was announced on 19.04.18 [1].", [(1, "announced 18.04.18")])
    )
    assert "19.04.18" in check(wrong, passages, "When?")[0][0]
