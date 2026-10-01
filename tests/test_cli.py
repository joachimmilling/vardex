import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vardex.cli import app
from vardex.llm import Answer
from vardex.pricing import Usage

runner = CliRunner()
COST_ARGS = ["cost", "--per-day", "1", "--input", "100", "--output", "10"]
FIRST_PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run every command in an empty folder, so calls are never logged to the repository."""
    monkeypatch.chdir(tmp_path)


def test_validate_accepts_the_first_pack():
    result = runner.invoke(app, ["validate", str(FIRST_PACK)])
    assert result.exit_code == 0
    assert "norwegian-companies" in result.output


def test_validate_rejects_a_folder_without_a_pack(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 1


def test_packs_lists_valid_invalid_and_empty_folders(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "pack.yaml").write_text(
        "name: good\ndescription: A valid pack.\nrecipes: [analyst]\n"
    )
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "pack.yaml").write_text("name: Bad Name\nrecipes: []\n")
    (tmp_path / "empty").mkdir()

    result = runner.invoke(app, ["packs", str(tmp_path)])

    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert any(line.startswith("OK") and "good 0.1.0" in line for line in lines)
    assert any(line.startswith("ERROR") and "bad" in line for line in lines)
    assert any(line.startswith("ERROR") and "empty" in line for line in lines)
    assert "not a valid pack" in result.output
    assert "No pack.yaml found" in result.output


def test_packs_with_only_valid_packs_succeeds():
    result = runner.invoke(app, ["packs", str(FIRST_PACK.parent)])
    assert result.exit_code == 0
    assert "norwegian-companies" in result.output
    assert "ERROR" not in result.output


def test_packs_in_an_empty_folder(tmp_path):
    result = runner.invoke(app, ["packs", str(tmp_path)])
    assert result.exit_code == 0
    assert "No packs found" in result.output


def test_ask_without_a_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("vardex.config.load_dotenv", lambda: None)  # ignore any local .env
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_cost_lists_every_model_with_the_monthly_total():
    result = runner.invoke(app, ["cost", "--per-day", "50", "--input", "8000", "--output", "1500"])
    assert result.exit_code == 0
    assert "claude-haiku-4-5" in result.output
    assert "$46.50" in result.output  # Sonnet 5.5


def test_cost_with_batch_halves_the_monthly_total():
    args = ["cost", "--per-day", "50", "--input", "8000", "--output", "1500", "--batch"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "$23.25" in result.output  # Sonnet 5.5, half of $46.50


def test_cost_with_model_shows_only_that_model():
    result = runner.invoke(app, [*COST_ARGS, "--model", "claude-sonnet-5-5"])
    assert result.exit_code == 0
    assert "claude-sonnet-5-5" in result.output
    assert "claude-haiku-4-5" not in result.output


def test_cost_with_model_and_batch():
    args = ["cost", "--per-day", "50", "--input", "8000", "--output", "1500"]
    result = runner.invoke(app, [*args, "--model", "claude-sonnet-5-5", "--batch"])
    assert result.exit_code == 0
    assert "$23.25" in result.output


def test_cost_with_an_unpriced_model_fails_clearly():
    result = runner.invoke(app, [*COST_ARGS, "--model", "gpt-imaginary"])
    assert result.exit_code == 1
    assert "No price for 'gpt-imaginary'" in result.stderr


def test_cost_refuses_more_cached_tokens_than_input_tokens():
    args = ["cost", "--per-day", "1", "--input", "100", "--output", "10", "--cached", "200"]
    assert runner.invoke(app, args).exit_code == 1


def test_ask_with_stats_and_another_model(monkeypatch):
    seen = {}

    def fake_ask(question, settings, effort=None, on_text=None):
        seen["model"] = settings.model
        return Answer(
            text="Nine digits.",
            model=settings.model,
            usage=Usage(input_tokens=40, output_tokens=300),
            thinking_tokens=250,
            stop_reason="end_turn",
            seconds=2.5,
            cost_usd=Decimal("0.0015"),
        )

    monkeypatch.setattr("vardex.cli.ask_model", fake_ask)
    result = runner.invoke(app, ["ask", "--stats", "--model", "claude-haiku-4-5", "What?"])
    assert result.exit_code == 0
    assert seen["model"] == "claude-haiku-4-5"
    assert "Nine digits." in result.output
    assert "250 thinking" in result.output
    assert "$0.0015" in result.output


def test_tokens_counts_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr("vardex.cli.count_tokens", lambda text, settings: 120)
    file = tmp_path / "report.txt"
    file.write_text("Driftsinntektene økte med 12 prosent.", encoding="utf-8")
    result = runner.invoke(app, ["tokens", str(file)])
    assert result.exit_code == 0
    assert "120" in result.output
    assert "24.00" in result.output  # 120 tokens / 5 words


def fake_answer(question, settings, effort=None, on_text=None):
    return Answer(
        text="Nine digits.",
        model="claude-sonnet-5-5",
        usage=Usage(input_tokens=40, output_tokens=300, cache_read_tokens=7),
        thinking_tokens=None,
        stop_reason="end_turn",
        seconds=1.25,
        cost_usd=Decimal("0.0030014"),
    )


def test_ask_appends_one_json_line_per_call(monkeypatch):
    monkeypatch.setattr("vardex.cli.ask_model", fake_answer)
    runner.invoke(app, ["ask", "What?"])
    runner.invoke(app, ["ask", "What?"])

    lines = Path("logs/calls.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["model"] == "claude-sonnet-5-5"
    assert record["input_tokens"] == 40
    assert record["output_tokens"] == 300
    assert record["cache_read_tokens"] == 7
    assert record["seconds"] == 1.25
    assert record["cost_usd"] == "0.0030014"  # a string, so the cost stays exact
    assert record["time"].endswith("Z")


def test_ask_still_prints_the_answer_when_the_log_cannot_be_written(monkeypatch):
    monkeypatch.setattr("vardex.cli.ask_model", fake_answer)
    Path("logs").write_text("a file where the log folder should be")

    result = runner.invoke(app, ["ask", "What?"])

    assert result.exit_code == 0
    assert "Nine digits." in result.output
    assert "could not write the call log" in result.output


def test_cost_warns_on_stderr_when_prices_are_stale(monkeypatch):
    monkeypatch.setattr("vardex.cli.today", lambda: date(2027, 1, 1))
    result = runner.invoke(app, COST_ARGS)
    assert result.exit_code == 0
    assert "prices were last checked" in result.stderr
    assert "platform.claude.com/docs/en/about-claude/pricing" in result.stderr
    assert "prices were last checked" not in result.stdout


def test_cost_does_not_warn_when_prices_are_fresh(monkeypatch):
    monkeypatch.setattr("vardex.cli.today", lambda: date(2026, 10, 1))
    result = runner.invoke(app, COST_ARGS)
    assert "prices were last checked" not in result.output


def test_ask_with_stats_warns_when_prices_are_stale(monkeypatch):
    monkeypatch.setattr("vardex.cli.ask_model", fake_answer)
    monkeypatch.setattr("vardex.cli.today", lambda: date(2027, 1, 1))
    assert "prices were last checked" in runner.invoke(app, ["ask", "--stats", "What?"]).stderr
    assert "prices were last checked" not in runner.invoke(app, ["ask", "What?"]).output


def test_ask_with_stream_prints_pieces_as_they_come_and_the_first_text_time(monkeypatch):
    def fake_stream(question, settings, effort=None, on_text=None):
        for piece in ["Nine ", "digits."]:
            on_text(piece)
        return replace(fake_answer(question, settings), first_text_seconds=0.4)

    monkeypatch.setattr("vardex.cli.ask_model", fake_stream)
    result = runner.invoke(app, ["ask", "--stream", "--stats", "What?"])

    assert result.exit_code == 0
    assert result.stdout == "Nine digits.\n"
    assert "first text 0.4 s" in result.stderr


def test_ask_without_stream_does_not_stream(monkeypatch):
    seen = {}

    def fake(question, settings, effort=None, on_text=None):
        seen["on_text"] = on_text
        return fake_answer(question, settings)

    monkeypatch.setattr("vardex.cli.ask_model", fake)
    result = runner.invoke(app, ["ask", "--stats", "What?"])
    assert seen["on_text"] is None
    assert result.stdout == "Nine digits.\n"
    assert "first text" not in result.stderr
