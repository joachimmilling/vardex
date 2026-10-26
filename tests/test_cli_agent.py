"""The agent through the CLI: what a person sees, and what a program reads."""

import json

import duckdb
import pytest
from fakes import FakeModel, Turn
from typer.testing import CliRunner

from vardex import cli
from vardex.llm import ModelError

runner = CliRunner()

MODELS = """
models:
  - name: companies
    description: One row per company.
    columns:
      - {name: orgnr, description: Organisation number.}
      - {name: name, description: Registered name.}
      - {name: revenue, description: Revenue in NOK.}
"""
METRICS = """
metrics:
  - {name: revenue, description: Revenue., table: companies, sql: sum(revenue), unit: NOK}
"""
ONTOLOGY = """
objects:
  - name: Company
    description: A company.
    table: companies
    key: orgnr
    label: name
actions:
  - name: adjust_credit_limit
    description: Propose a new credit limit.
    object: Company
    connector: crm
    confirm: Set {label}'s credit limit to {amount} NOK, because {reason}.
    fields:
      - {name: amount, description: The new limit., type: amount}
      - {name: reason, description: Why., type: text}
"""
PERMISSIONS = """
budgets:
  per_run: 0.50
tools:
  describe_tool: read
  list_tables: read
  describe_table: read
  run_sql: read
  make_chart: read
  adjust_credit_limit: draft
"""
TABLES = """
create table companies (orgnr varchar, name varchar, revenue decimal(18, 0));
insert into companies values ('100000008', 'DEMO LAKS AS', 1480312000);
"""


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A small pack with a warehouse, an action and a permissions file."""
    folder = tmp_path / "pack"
    files = {
        "pack.yaml": "name: demo\ndescription: A demo pack.\nrecipes: [analyst]\n",
        "warehouse/models/models.yml": MODELS,
        "warehouse/metrics.yaml": METRICS,
        "warehouse/ontology.yaml": ONTOLOGY,
        "permissions.yaml": PERMISSIONS,
    }
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    data = tmp_path / "data"
    (data / "demo").mkdir(parents=True)
    with duckdb.connect(data / "demo" / "warehouse.duckdb") as con:
        con.execute(TABLES)
    monkeypatch.setenv("VARDEX_DATA_DIR", str(data))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return folder


def model(*replies):
    """Make the CLI use a prepared model instead of a real one."""
    fake = FakeModel(*replies)
    return fake


def use(monkeypatch, fake):
    monkeypatch.setattr(cli, "connect", lambda settings: fake)


def test_the_agent_answers_a_question_with_its_tools(pack, monkeypatch):
    fake = model(
        Turn("Looking at the tables.", [("list_tables", {})]),
        Turn("", [("run_sql", {"sql": "select name, revenue from companies"})]),
        "DEMO LAKS AS had revenue of 1,480,312,000 NOK.",
    )
    use(monkeypatch, fake)
    result = runner.invoke(cli.app, ["agent", str(pack), "What was revenue?"])
    assert result.exit_code == 0, result.output
    assert "1,480,312,000 NOK" in result.stdout


def test_json_gives_a_program_the_whole_run(pack, monkeypatch):
    fake = model(
        Turn("", [("run_sql", {"sql": "select name from companies"})]),
        "DEMO LAKS AS.",
    )
    use(monkeypatch, fake)
    result = runner.invoke(cli.app, ["agent", str(pack), "Who?", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["ok"] is True
    assert body["answer"] == "DEMO LAKS AS."
    assert body["steps"] == 2
    assert [call["name"] for call in body["tool_calls"]] == ["run_sql"]
    assert body["tool_calls"][0]["permission"] == "read"
    assert body["usage"]["input_tokens"] == 2000
    assert body["cost_usd"] == "0.008"


def test_a_failure_in_json_carries_a_code_a_program_can_branch_on(pack, monkeypatch):
    class Failing(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("the API is down", retryable=False)

    use(monkeypatch, Failing("unused"))
    result = runner.invoke(cli.app, ["agent", str(pack), "Anything?", "--json"])
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body == {"ok": False, "error": {"code": "model_failed", "message": "the API is down"}}


def test_a_pack_with_no_permissions_file_has_no_agent(pack, monkeypatch):
    (pack / "permissions.yaml").unlink()
    use(monkeypatch, model("never asked"))
    result = runner.invoke(cli.app, ["agent", str(pack), "Anything?", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "pack_invalid"


def test_a_run_that_does_not_answer_exits_one(pack, monkeypatch):
    turns = [Turn("Again.", [("list_tables", {})])] * 30
    use(monkeypatch, model(*turns))
    result = runner.invoke(cli.app, ["agent", str(pack), "Go on", "--steps", "3"])
    assert result.exit_code == 1
    assert "Stopped after 3 steps" in result.output


def test_dry_run_says_what_would_change_and_changes_nothing(pack, monkeypatch):
    fake = model(
        Turn(
            "",
            [
                (
                    "adjust_credit_limit",
                    {"key": "100000008", "amount": 500000, "reason": "three good years"},
                )
            ],
        ),
        "I would have proposed a new limit.",
    )
    use(monkeypatch, fake)
    result = runner.invoke(
        cli.app, ["agent", str(pack), "Raise their limit", "--dry-run", "--json"]
    )
    body = json.loads(result.stdout)
    call = body["tool_calls"][0]
    assert "Dry run, nothing written" in call["content"]
    assert call["wrote"] is False


def test_the_tools_command_shows_every_tool_and_its_level(pack):
    result = runner.invoke(cli.app, ["tools", str(pack)])
    assert result.exit_code == 0
    assert "run_sql" in result.stdout
    assert "adjust_credit_limit" in result.stdout
    assert "draft" in result.stdout


def test_check_catches_a_tool_with_no_permission_level(pack):
    text = (pack / "permissions.yaml").read_text(encoding="utf-8")
    (pack / "permissions.yaml").write_text(text.replace("  run_sql: read\n", ""), encoding="utf-8")
    result = runner.invoke(cli.app, ["check", str(pack)])
    assert result.exit_code == 1
    assert "tool run_sql has no permission level" in result.output


def test_check_catches_a_level_for_a_tool_that_is_not_there(pack):
    with (pack / "permissions.yaml").open("a", encoding="utf-8") as file:
        file.write("  delete_everything: act\n")
    result = runner.invoke(cli.app, ["check", str(pack)])
    assert result.exit_code == 1
    assert "delete_everything, which is not a tool here" in result.output


def test_check_passes_a_pack_whose_permissions_match_its_tools(pack):
    result = runner.invoke(cli.app, ["check", str(pack)])
    assert result.exit_code == 0, result.output
    assert "1 actions" in result.stdout
