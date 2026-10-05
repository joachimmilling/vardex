import json
from pathlib import Path

from fakes import FakeModel
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


def test_cost_refuses_more_cached_tokens_than_input_tokens():
    args = ["cost", "--per-day", "1", "--input", "100", "--output", "10", "--cached", "200"]
    assert runner.invoke(app, args).exit_code == 1


def test_cost_for_one_model_at_batch_prices():
    args = ["cost", "--per-day", "50", "--input", "8000", "--output", "1500"]
    result = runner.invoke(app, [*args, "--model", "claude-sonnet-5-5", "--batch"])
    assert result.exit_code == 0
    assert "$23.25" in result.output  # half of $46.50
    assert "claude-haiku-4-5" not in result.output


def test_cost_refuses_a_model_without_a_price():
    args = ["cost", "--per-day", "1", "--input", "100", "--output", "10", "--model", "gpt-x"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "No price for 'gpt-x'" in result.output


def test_cost_warns_when_prices_are_old(monkeypatch):
    monkeypatch.setattr("vardex.pricing.PRICES_CHECKED", "2020-01-01")
    result = runner.invoke(app, ["cost", "--per-day", "1", "--input", "100", "--output", "10"])
    assert result.exit_code == 0
    assert "Warning: prices were checked" in result.output


def test_ask_with_stats_and_another_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the call log is written to the current folder
    seen = {}

    def connect(settings):
        seen["model"] = settings.model
        return FakeModel("Nine digits.")

    monkeypatch.setattr("vardex.cli.connect", connect)
    result = runner.invoke(app, ["ask", "--stats", "--model", "claude-haiku-4-5", "What?"])
    assert result.exit_code == 0
    assert seen["model"] == "claude-haiku-4-5"
    assert "Nine digits." in result.output
    assert "150 thinking" in result.output
    assert "$0.0040" in result.output


def test_ask_streams_the_answer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel("Nine digits."))
    result = runner.invoke(app, ["ask", "--stream", "--stats", "What?"])
    assert result.exit_code == 0
    assert result.output.startswith("Nine digits.\n")
    assert "first text after 0.5 s" in result.output


def test_ask_logs_every_call(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel("One.", "Two."))
    runner.invoke(app, ["ask", "One?"])
    runner.invoke(app, ["ask", "Two?"])
    lines = (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cost_usd"] == "0.004"


def test_extract_prints_one_line_per_document_and_reports_failures(monkeypatch, tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("Demo Bygg AS. Alle tal i heile tusen kroner. Driftsresultat 5 927.")
    bad = tmp_path / "bad.txt"
    bad.write_text("Til våre aksjonærer: 2025 ble et år med store endringer.")
    values = {
        "problem": None,
        "company_name": "Demo Bygg AS",
        "fiscal_year": 2025,
        "accounts": "company",
        "currency": "NOK",
        "unit": "thousands",
        "unit_quote": "Alle tal i heile tusen kroner",
        "revenue": None,
        "revenue_quote": None,
        "operating_profit": 5927,
        "operating_profit_quote": "5 927",
    }
    no_figures = values | {"problem": "The page has no figures."}
    model = FakeModel(json.dumps(values), json.dumps(no_figures))
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["extract", str(FIRST_PACK), "key-figures", str(good), str(bad)])

    assert result.exit_code == 1
    first = json.loads(result.output.splitlines()[0])
    assert first["file"] == "good.txt"
    assert first["operating_profit"] == "5927"  # an exact string, never a float
    assert "bad.txt: the model reports: The page has no figures." in result.output
    assert "1 of 2 documents failed." in result.output
    assert len((tmp_path / "logs" / "calls.jsonl").read_text().splitlines()) == 2


def test_extract_with_an_unknown_extraction_fails_before_any_call(tmp_path):
    page = tmp_path / "page.txt"
    page.write_text("…")
    result = runner.invoke(app, ["extract", str(FIRST_PACK), "invoices", str(page)])
    assert result.exit_code == 1
    assert "No extraction 'invoices'" in result.output


def test_tokens_counts_a_file(monkeypatch, tmp_path):
    model = FakeModel()
    model.count_tokens = lambda text: 120
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    file = tmp_path / "report.txt"
    file.write_text("Driftsinntektene økte med 12 prosent.", encoding="utf-8")
    result = runner.invoke(app, ["tokens", str(file)])
    assert result.exit_code == 0
    assert "120" in result.output
    assert "24.00" in result.output  # 120 tokens / 5 words
