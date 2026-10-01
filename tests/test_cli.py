from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from vardex.cli import app
from vardex.llm import Answer
from vardex.pricing import Usage

runner = CliRunner()
FIRST_PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"


def test_validate_accepts_the_first_pack():
    result = runner.invoke(app, ["validate", str(FIRST_PACK)])
    assert result.exit_code == 0
    assert "norwegian-companies" in result.output


def test_validate_rejects_a_folder_without_a_pack(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code == 1


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


def test_cost_refuses_more_cached_tokens_than_input_tokens():
    args = ["cost", "--per-day", "1", "--input", "100", "--output", "10", "--cached", "200"]
    assert runner.invoke(app, args).exit_code == 1


def test_ask_with_stats_and_another_model(monkeypatch):
    seen = {}

    def fake_ask(question, settings, effort=None):
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
