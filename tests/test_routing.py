"""Routing and shadow mode, for an answer and an extraction, with fake models."""

import json

import duckdb
import pytest
from fakes import FakeModel
from test_extraction import GOOD, PAGE, SPEC
from test_text_to_sql import KNOWLEDGE, QUESTION, SQL, TABLES, TODAY, plan

from vardex import routing
from vardex.capabilities import FAILED, UNREACHABLE, Answering, Extracting
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.llm import ModelError
from vardex.shadow import ShadowLog, run_shadow, summary

ANSWERED = "DEMO LAKS AS had the highest operating margin in 2025: 23.6%."
BROKEN = plan(sql="select nonsense from nowhere")


@pytest.fixture
def warehouse(tmp_path):
    with duckdb.connect(tmp_path / "w.duckdb") as con:
        con.execute(TABLES)
    connection = DuckDBWarehouse(tmp_path / "w.duckdb")
    yield connection
    connection.close()


def small_and_large(small_replies, large_replies):
    small, large = FakeModel(*small_replies), FakeModel(*large_replies)
    small.model, large.model = "gemma4:12b", "claude-sonnet-5-5"
    return small, large


def asking(warehouse):
    return Answering(warehouse, KNOWLEDGE, QUESTION, today=TODAY)


def test_work_the_small_model_does_never_reaches_the_large_one(warehouse):
    small, large = small_and_large([plan(), ANSWERED], [])
    routed = routing.run(routing.Route(small, large), asking(warehouse))
    assert routed.attempt.kind == "answer"
    assert routed.attempt.model == "gemma4:12b"
    assert routed.escalated == ""
    assert not large.requests


def test_an_answer_that_fails_its_checks_goes_to_the_large_model(warehouse):
    small, large = small_and_large([BROKEN, BROKEN], [plan(), ANSWERED])
    routed = routing.run(routing.Route(small, large), asking(warehouse))
    assert routed.attempt.text == ANSWERED
    assert routed.escalated.startswith("it failed the checks")
    # Every call is in the result, the small model's two included, so all of them are paid for
    assert [a.model for a in routed.answers] == ["gemma4:12b"] * 2 + ["claude-sonnet-5-5"] * 2


def test_a_small_model_that_gives_up_is_asked_no_further_than_the_large_one(warehouse):
    gave_up = plan("refuse", sql=None, message="The tables cannot answer this.")
    small, large = small_and_large([gave_up], [plan(), ANSWERED])
    routed = routing.run(routing.Route(small, large), asking(warehouse))
    assert routed.attempt.kind == "answer"
    assert routed.escalated.startswith("it gave up")


def test_a_question_back_is_the_persons_to_answer_not_the_large_models(warehouse):
    small, large = small_and_large([plan("clarify", sql=None, message="Which year?")], [])
    routed = routing.run(routing.Route(small, large), asking(warehouse))
    assert routed.attempt.kind == "clarify"
    assert not large.requests


def test_a_small_model_that_cannot_be_reached_is_routed_round(warehouse):
    class Down(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("Could not reach http://localhost:11434/v1", retryable=True)

    routed = routing.run(routing.Route(Down(), FakeModel(plan(), ANSWERED)), asking(warehouse))
    assert routed.escalated.startswith("it could not be reached")
    assert routed.attempt.kind == "answer"


def test_a_model_no_attempt_can_reach_is_raised_not_routed_round(warehouse):
    class NoKey(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("No API key")  # every attempt would fail the same way

    with pytest.raises(ModelError):
        routing.run(routing.Route(NoKey(), FakeModel()), asking(warehouse))


def test_when_both_fail_the_result_carries_both_models_calls_and_the_error(warehouse):
    small, large = small_and_large([BROKEN, BROKEN], [BROKEN, BROKEN])
    routed = routing.run(routing.Route(small, large), asking(warehouse))
    assert routed.attempt.kind == FAILED
    assert routed.attempt.error is not None
    assert len(routed.answers) == 4


def test_an_extraction_is_routed_by_its_own_checks():
    # The small model quotes a figure that is not on the page: the quote check fails it twice
    made_up = json.dumps(GOOD | {"revenue_quote": "1 480 999"})
    small, large = small_and_large([made_up, made_up], [json.dumps(GOOD)])
    routed = routing.run(routing.Route(small, large), Extracting(SPEC, PAGE))
    assert routed.attempt.kind == "values"
    assert routed.attempt.model == "claude-sonnet-5-5"
    assert routed.escalated.startswith("it failed the checks")
    assert routed.attempt.detail == "7 of 7 fields"


def test_a_document_the_small_model_finds_unusable_goes_to_the_large_one():
    problem = json.dumps({key: None for key in GOOD} | {"problem": "No accounts on this page."})
    small, large = small_and_large([problem], [json.dumps(GOOD)])
    routed = routing.run(routing.Route(small, large), Extracting(SPEC, PAGE))
    assert routed.escalated.startswith("it gave up")
    assert routed.attempt.kind == "values"


def test_shadow_keeps_both_answers_and_finds_where_they_agree(warehouse, tmp_path):
    work = asking(warehouse)
    live = work.attempt(FakeModel(plan(), ANSWERED))
    candidate = FakeModel(plan(), "Demo Laks AS, with 23.6%.")
    candidate.model = "gemma4:12b"
    entry = run_shadow(candidate, work, live, input=QUESTION, trace_id="a" * 32)
    assert entry.agree
    assert (entry.capability, entry.shadow.model) == ("answer", "gemma4:12b")
    assert entry.shadow.detail == "2 rows"
    assert entry.shadow.cost_usd == "0.008"  # two calls at the fake's price
    log = ShadowLog(tmp_path / "shadow.jsonl")
    log.add(entry)
    assert log.read() == [entry]
    assert log.read(capability="extract") == []
    assert "1 shadowed (answer), 1 alike: 100%" in summary(log.read())[0]


def test_a_candidate_that_fails_is_recorded_never_raised(warehouse):
    work = asking(warehouse)
    live = work.attempt(FakeModel(plan(), ANSWERED))
    entry = run_shadow(FakeModel(BROKEN, BROKEN), work, live, input=QUESTION, trace_id="b" * 32)
    assert not entry.agree
    assert entry.why == f"answer against {FAILED}"
    assert "Where they differ" in summary([entry])

    class NoKey(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("No API key")

    entry = run_shadow(NoKey(), work, live, input=QUESTION, trace_id="c" * 32)
    assert entry.shadow.kind == UNREACHABLE


def test_answers_that_return_other_rows_do_not_agree(warehouse):
    work = asking(warehouse)
    live = work.attempt(FakeModel(plan(), ANSWERED))
    other = work.attempt(FakeModel(plan(sql=SQL.replace("desc", "asc")), "DEMO DATA AS: 5.0%."))
    assert work.differ(live, other) == ""  # the same rows, in another order, hold each other
    fewer = work.attempt(FakeModel(plan(sql=SQL + "\nlimit 1"), ANSWERED))
    assert work.differ(live, fewer) == "2 rows against 1"  # a top one is not the list
    oslo = "select name from companies where county = 'Oslo'"
    assert work.differ(live, work.attempt(FakeModel(plan(sql=oslo), "DEMO DATA AS."))) != ""


def test_extractions_agree_on_values_not_on_quotes():
    work = Extracting(SPEC, PAGE)
    live = work.attempt(FakeModel(json.dumps(GOOD)))
    # The same figure, written with a decimal place: an exact decimal, so they agree
    alike = work.attempt(FakeModel(json.dumps(GOOD | {"revenue": 1480312.0})))
    assert work.differ(live, alike) == ""
    group = work.attempt(FakeModel(json.dumps(GOOD | {"accounts": "group"})))
    assert work.differ(live, group) == "accounts: 'company' against 'group'"
