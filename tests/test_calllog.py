import json
from datetime import UTC, datetime
from decimal import Decimal

from vardex.calllog import CallRecord, log_call
from vardex.llm import Answer
from vardex.pricing import Usage

ANSWER = Answer(
    text="Nine digits.",
    model="claude-haiku-4-5",
    usage=Usage(input_tokens=1, output_tokens=3, cache_write_tokens=5, cache_read_tokens=2),
    thinking_tokens=None,
    stop_reason="end_turn",
    seconds=0.5,
    cost_usd=Decimal("0.000017225"),
)


def test_log_call_creates_the_folder_and_appends(tmp_path):
    path = tmp_path / "logs" / "calls.jsonl"
    log_call(ANSWER, path)
    log_call(ANSWER, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["cache_write_tokens"] == 5


def test_cost_round_trips_exactly(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_call(ANSWER, path)
    record = CallRecord.model_validate_json(path.read_text())
    assert record.cost_usd == Decimal("0.000017225")


def test_unknown_cost_is_logged_as_null(tmp_path):
    path = tmp_path / "calls.jsonl"
    log_call(Answer(**{**ANSWER.__dict__, "cost_usd": None}), path)
    assert json.loads(path.read_text())["cost_usd"] is None


def test_record_keeps_the_time_it_is_given():
    when = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    assert CallRecord.from_answer(ANSWER, time=when).time == when
