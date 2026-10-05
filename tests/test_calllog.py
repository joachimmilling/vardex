import json
from decimal import Decimal

from vardex.calllog import log_call
from vardex.llm import Answer
from vardex.pricing import Usage


def make_answer(cost):
    return Answer(
        text="Nine digits.",
        model="claude-sonnet-5-5",
        usage=Usage(input_tokens=40, output_tokens=300),
        thinking_tokens=250,
        stop_reason="end_turn",
        seconds=2.5,
        cost_usd=cost,
    )


def test_each_call_appends_one_line(tmp_path):
    path = tmp_path / "logs" / "calls.jsonl"
    log_call(make_answer(Decimal("0.00308")), path)
    log_call(make_answer(None), path)

    first, second = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert first["model"] == "claude-sonnet-5-5"
    assert first["output_tokens"] == 300
    assert first["thinking_tokens"] == 250
    assert first["cost_usd"] == "0.00308"  # a string, so the exact Decimal survives
    assert second["cost_usd"] is None
