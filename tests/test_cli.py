from pathlib import Path

from typer.testing import CliRunner

from vardex.cli import app

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
