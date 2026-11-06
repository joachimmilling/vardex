"""Reading traces back, adding them up, and turning what people said into candidate cases."""

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import httpx2
import pytest

from vardex.dashboard import (
    GROUPINGS,
    by_day,
    figures,
    format_figures,
    percentile,
    write_dashboard,
)
from vardex.feedback import (
    Candidates,
    Feedback,
    FeedbackError,
    FeedbackLog,
    candidate_from,
    sample,
)
from vardex.traces import (
    FileTraces,
    LangfuseTraces,
    SpanRecord,
    Trace,
    TraceError,
    as_traces,
    traces,
)

NOW = datetime(2026, 10, 27, 9, 0, tzinfo=UTC)


def span(trace, sid, name, parent=None, seconds=1.0, status="ok", **attributes):
    return SpanRecord(
        trace_id=trace,
        span_id=sid,
        parent_id=parent,
        name=name,
        start=NOW,
        end=NOW + timedelta(seconds=seconds),
        status=status,
        attributes=attributes,
    )


def chat(trace, sid, parent, cost="0.004", **more):
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 1000,
        "gen_ai.usage.output_tokens": 200,
        "vardex.cost_usd": cost,
        "gen_ai.response.model": "claude-sonnet-5-5",
        **more,
    }
    return span(trace, sid, "chat claude-sonnet-5-5", parent, seconds=2.0, **attributes)


def answered(trace="a" * 32, seconds=4.0, repaired=False, failed=False, question="How many?"):
    """One traced question: a root, an attempt or two, and a model call in each."""
    root = span(
        trace,
        "r",
        "vardex answer",
        seconds=seconds,
        status="error" if failed else "ok",
        **{
            "vardex.command": "answer",
            "vardex.question": question,
            "vardex.outcome": "error" if failed else "answer",
            "vardex.sql": "select 1",
            "vardex.answer": "One.",
            "vardex.pack.version": "v1",
        },
    )
    spans = [root, span(trace, "w1", "write_sql", "r", **{"vardex.attempt": 1})]
    spans.append(chat(trace, "c1", "w1"))
    if repaired:
        spans += [span(trace, "w2", "write_sql", "r", **{"vardex.attempt": 2})]
        spans.append(chat(trace, "c2", "w2"))
    return Trace(spans)


def test_a_trace_adds_up_its_calls_and_knows_when_it_needed_a_repair():
    trace = answered(repaired=True)
    assert len(trace.calls) == 2
    assert trace.cost_usd == Decimal("0.008")
    assert trace.input_tokens == 2000
    assert trace.repaired
    assert trace.command == "answer"
    assert trace.model == "claude-sonnet-5-5"
    assert [(depth, s.name) for depth, s in trace.walk()][:3] == [
        (0, "vardex answer"),
        (1, "write_sql"),
        (2, "chat claude-sonnet-5-5"),
    ]


def test_a_call_with_no_price_is_counted_as_unpriced_not_as_free():
    trace = Trace([span("t" * 32, "r", "vardex ask"), chat("t" * 32, "c", "r", cost=None)])
    assert trace.cost_usd == 0
    assert trace.unpriced == 1


def test_file_traces_read_a_window_and_find_a_trace_by_its_start(tmp_path):
    folder = tmp_path / "traces"
    folder.mkdir()
    lines = [
        record(answered("ab" + "0" * 30).spans[0]),
        record(answered("ac" + "0" * 30).spans[0]),
        "{ cut short by a crash",
    ]
    (folder / f"{NOW.date().isoformat()}.jsonl").write_text("\n".join(lines) + "\n")
    store = FileTraces(folder)
    assert len(store.spans(NOW - timedelta(hours=1), NOW + timedelta(hours=1))) == 2
    assert store.spans(NOW + timedelta(hours=1)) == []
    assert store.trace("ab").trace_id.startswith("ab")
    with pytest.raises(TraceError, match="names 2 traces"):
        store.trace("a")
    with pytest.raises(TraceError, match="no trace"):
        store.trace("ff")


def record(s: SpanRecord) -> str:
    return json.dumps(
        {
            "trace_id": s.trace_id,
            "span_id": s.span_id,
            "parent_id": s.parent_id,
            "name": s.name,
            "start": s.start.isoformat(),
            "end": s.end.isoformat(),
            "status": s.status,
            "attributes": s.attributes,
            "events": [],
        }
    )


def test_spans_are_grouped_into_traces_oldest_first():
    early = answered("e" * 32).spans
    late = [
        SpanRecord(**{**s.__dict__, "start": s.start + timedelta(hours=1)})
        for s in answered("l" * 32).spans
    ]
    found = as_traces(late + early)
    assert [t.trace_id[0] for t in found] == ["e", "l"]


def test_percentile_is_a_value_that_occurred():
    assert percentile([], 0.95) == 0
    assert percentile([1.0, 9.0], 0.5) == 1.0
    assert percentile([float(n) for n in range(1, 101)], 0.95) == 95.0
    assert percentile([2.0, 3.0, 40.0], 0.95) == 40.0  # with few values, p95 is the worst


def test_figures_count_failures_repairs_cost_and_feedback_per_group():
    found = [
        answered("1" * 32, seconds=3),
        answered("2" * 32, seconds=5, repaired=True),
        answered("3" * 32, seconds=30, failed=True),
    ]
    (row,) = figures(found, GROUPINGS["command"], {"1" * 32: "good", "3" * 32: "bad"})
    assert (row.group, row.requests, row.failed, row.repaired) == ("answer", 3, 1, 1)
    assert row.cost_usd == Decimal("0.016")
    assert row.cost_per_request == Decimal("0.016") / 3
    assert (row.p50_seconds, row.p95_seconds) == (5.0, 30.0)
    assert (row.good, row.bad) == (1, 1)
    table = format_figures([row])
    assert "1 (33%)" in table and "$0.0160" in table


def test_a_day_with_no_requests_is_a_zero_not_a_gap():
    labels, days = by_day([answered()], date(2026, 10, 25), date(2026, 10, 27))
    assert labels == ["25 Oct", "26 Oct", "27 Oct"]
    assert [d.requests for d in days] == [0, 0, 1]


def test_the_dashboard_page_has_its_tables_and_charts(tmp_path):
    page = write_dashboard(
        tmp_path / "dashboard.html", "demo", [answered()], date(2026, 10, 21), date(2026, 10, 27)
    )
    html = page.read_text()
    assert "<h2>By command</h2>" in html and "<h2>By pack version</h2>" in html
    assert html.count('role="img"') == 2  # cost per day, and the slowest requests
    assert "prefers-color-scheme: dark" in html


def test_a_person_who_changes_their_mind_is_believed(tmp_path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    log.add(Feedback("a" * 32, "bad", "kari"))
    log.add(Feedback("a" * 32, "good", "kari", note="I misread it"))
    assert log.verdicts() == {"a" * 32: "good"}
    assert len(log.all()) == 2  # both are kept: what was said, and when


def test_flagging_the_same_answer_twice_makes_one_candidate(tmp_path):
    store = Candidates(tmp_path / "candidates.jsonl")
    first, new = store.add(candidate_from(answered(), "flagged", "wrong year"))
    again, new_again = store.add(candidate_from(answered(), "flagged"))
    assert new and not new_again
    assert again.id == first.id == "a" * 12
    assert first.question == "How many?" and first.sql == "select 1"
    assert first.asked_on == "2026-10-27"


def test_a_candidate_is_decided_once(tmp_path):
    store = Candidates(tmp_path / "candidates.jsonl")
    candidate, _ = store.add(candidate_from(answered(), "sampled"))
    store.decide(candidate, "promoted", "kari", case="how-many")
    assert store.open() == []
    assert store.get("aaa").case == "how-many"
    with pytest.raises(FeedbackError, match="already promoted"):
        store.decide(store.get("aaa"), "dismissed", "ola")
    with pytest.raises(FeedbackError, match="no candidate"):
        store.get("bbb")


def test_a_trace_with_no_question_cannot_be_a_candidate():
    trace = Trace([span("t" * 32, "r", "vardex ingest")])
    with pytest.raises(FeedbackError, match="no question"):
        candidate_from(trace, "flagged")


def test_a_sample_is_random_repeatable_and_only_of_questions_answered():
    found = [answered(str(n) * 32, question=f"Q{n}") for n in range(1, 10)]
    found.append(Trace([span("x" * 32, "r", "vardex ingest")]))
    picked = sample(found, 3, seed=7)
    assert len(picked) == 3
    assert [t.trace_id for t in picked] == [t.trace_id for t in sample(found, 3, seed=7)]
    assert all(t.question for t in sample(found, 50, seed=1))


def observation(oid, parent=None, **more):
    return {
        "id": oid,
        "traceId": "f" * 32,
        "parentObservationId": parent,
        "name": "vardex answer" if parent is None else "chat claude-sonnet-5-5",
        "type": "SPAN" if parent is None else "GENERATION",
        "startTime": "2026-10-27T09:00:00.000Z",
        "endTime": "2026-10-27T09:00:04.000Z",
        "level": "DEFAULT",
        **more,
    }


def langfuse(pages, seen):
    """A Langfuse that answers with the pages given, one per request, and keeps each query."""

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx2.Response(200, json=pages[len(seen) - 1])

    return LangfuseTraces(
        "https://cloud.langfuse.com",
        "pk",
        "sk",
        http=lambda: httpx2.Client(
            base_url="https://cloud.langfuse.com", transport=httpx2.MockTransport(handler)
        ),
    )


def test_langfuse_observations_come_back_as_the_spans_that_were_sent():
    root = observation(
        "r", metadata={"attributes": {"vardex.question": "Hvor mange?", "vardex.command": "answer"}}
    )
    call = observation(
        "c",
        parent="r",
        model="claude-sonnet-5-5",
        usageDetails={"input": 6200, "output": 50, "cache_read_input_tokens": 6000},
        metadata={"attributes": {"vardex.cost_usd": "0.004", "vardex.attempt": "1"}},
    )
    seen = []
    store = langfuse([{"data": [root], "meta": {"cursor": "next"}}, {"data": [call]}], seen)
    found = traces(store, NOW - timedelta(hours=1))
    assert seen[0]["fromStartTime"] == "2026-10-27T08:00:00.000Z"
    assert seen[0]["expandMetadata"] == "attributes"  # or the attributes are cut at 200
    assert seen[1]["cursor"] == "next"
    (trace,) = found
    assert trace.question == "Hvor mange?" and trace.command == "answer"
    (chat_span,) = trace.calls  # a generation is a model call, whatever it was named
    assert chat_span.get("gen_ai.usage.input_tokens") == 6200
    assert chat_span.get("gen_ai.usage.cache_read.input_tokens") == 6000
    assert chat_span.get("gen_ai.request.model") == "claude-sonnet-5-5"
    assert chat_span.get("vardex.attempt") == 1
    assert trace.cost_usd == Decimal("0.004")


def test_langfuse_attributes_flattened_into_the_metadata_are_read_too():
    # The shape Langfuse's cloud answers with: every attribute a key of the metadata itself
    root = observation(
        "r",
        metadata={
            "attributes.vardex.command": "answer",
            "attributes.vardex.outcome": "answer",
            "resourceAttributes.vardex.pack": "norwegian-companies",
            "resourceAttributes.vardex.pack.version": "b37856758845",
        },
    )
    call = observation(
        "c",
        parent="r",
        model="claude-sonnet-5-5",
        usageDetails={"input": 911, "output": 340},
        metadata={
            "attributes.vardex.cost_usd": 0.005222,
            "attributes.gen_ai.usage.input_tokens": 911,
            "resourceAttributes.vardex.pack": "norwegian-companies",
        },
    )
    (trace,) = traces(langfuse([{"data": [root, call]}], []), NOW - timedelta(hours=1))
    assert trace.command == "answer" and trace.outcome == "answer"
    assert trace.pack == "norwegian-companies" and trace.pack_version == "b37856758845"
    assert trace.cost_usd == Decimal("0.005222")


def test_langfuse_finds_a_trace_by_the_first_characters_of_its_id():
    seen = []
    store = langfuse([{"data": [observation("r")]}], seen)
    assert store.trace("ffff").trace_id == "f" * 32
    assert "traceId" not in seen[0] and "fromStartTime" in seen[0]


def test_langfuse_finds_a_trace_by_its_whole_id_and_says_so_when_it_cannot():
    seen = []
    store = langfuse([{"data": [observation("r")]}, {"data": []}], seen)
    assert store.trace("f" * 32).trace_id == "f" * 32
    assert seen[0]["traceId"] == "f" * 32
    with pytest.raises(TraceError, match="no trace"):
        store.trace("e" * 32)


def test_langfuse_that_is_down_is_a_trace_error():
    def handler(request):
        return httpx2.Response(503)

    store = LangfuseTraces(
        "https://cloud.langfuse.com",
        "pk",
        "sk",
        http=lambda: httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(TraceError, match="did not answer"):
        store.spans(NOW)
