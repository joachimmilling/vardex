import json
from pathlib import Path

import duckdb
import pytest
from fakes import FakeModel
from typer.testing import CliRunner

from vardex.cli import app
from vardex.semantic import Knowledge, Metric, ModelDoc, Ontology, Questions
from vardex.warehouse import Build, WarehouseError

runner = CliRunner()
PACK = str(Path(__file__).parent.parent / "packs" / "norwegian-companies")
SQL = "select name, employees from dim_companies order by employees desc"


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VARDEX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VARDEX_WAREHOUSE_URL", raising=False)
    return tmp_path


def make_warehouse(data: Path) -> Path:
    """A warehouse holding two fictional companies, where the pack's warehouse would be."""
    path = data / "norwegian-companies" / "warehouse.duckdb"
    path.parent.mkdir()
    with duckdb.connect(path) as con:
        con.execute("create table dim_companies (name varchar, employees integer)")
        con.execute("insert into dim_companies values ('DEMO LAKS AS', 120), ('DEMO FISK AS', 80)")
    return path


def test_sql_prints_the_result(data):
    make_warehouse(data)
    result = runner.invoke(app, ["sql", PACK, SQL])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "name          employees",
        "------------  ---------",
        "DEMO LAKS AS        120",
        "DEMO FISK AS         80",
        "(2 rows)",
    ]


def test_sql_on_another_warehouse(data):
    path = make_warehouse(data).rename(data / "elsewhere.duckdb")
    result = runner.invoke(app, ["sql", PACK, "select count(*) as n", "--warehouse", str(path)])
    assert result.exit_code == 0
    assert "(1 row)" in result.output


def test_sql_cannot_write(data):
    make_warehouse(data)
    result = runner.invoke(app, ["sql", PACK, "create table t (x integer)"])
    assert result.exit_code == 1
    assert "read-only" in result.output


def test_sql_before_any_ingest(data):
    result = runner.invoke(app, ["sql", PACK, "select 1"])
    assert result.exit_code == 1
    assert "Run vardex ingest first" in result.output


def test_check_names_what_does_not_hold(data):
    make_warehouse(data)
    result = runner.invoke(app, ["check", PACK])
    assert result.exit_code == 1
    assert "model fct_accounts is not in the warehouse" in result.output


def test_answer_shows_the_answer_the_sql_the_rows_and_where_each_number_came_from(
    data, monkeypatch
):
    make_warehouse(data)
    plan = {
        "action": "query",
        "sql": SQL,
        "message": None,
        "assumptions": ["Employees are today's figures."],
    }
    model = FakeModel(json.dumps(plan), "Demo Laks AS has the most employees: 120.")
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.setattr("vardex.cli.load_knowledge", lambda pack: knowledge())

    result = runner.invoke(app, ["answer", PACK, "Who has the most employees?"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "Demo Laks AS has the most employees: 120."
    assert "- Employees are today's figures." in lines
    assert SQL in result.stdout
    assert "DEMO LAKS AS        120" in lines
    assert "  120              row 1, employees = 120" in lines


def test_answer_that_fails_its_checks_is_red_and_shows_the_sql(data, monkeypatch):
    make_warehouse(data)
    plan = {"action": "query", "sql": SQL, "message": None, "assumptions": []}
    made_up = "Demo Laks AS has 150 employees."
    model = FakeModel(json.dumps(plan), made_up, made_up)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.setattr("vardex.cli.load_knowledge", lambda pack: knowledge())

    result = runner.invoke(app, ["answer", PACK, "Who has the most employees?"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "the answer has numbers no query returned: 150" in result.stderr
    assert SQL in result.stderr


def test_answer_show_prompt_calls_nothing(data, monkeypatch):
    make_warehouse(data)
    monkeypatch.setattr("vardex.cli.load_knowledge", lambda pack: knowledge())
    result = runner.invoke(app, ["answer", PACK, "Who?", "--show-prompt"])
    assert result.exit_code == 0
    assert "one SQL query for DuckDB" in result.output
    assert '<table name="dim_companies">' in result.output


def knowledge() -> Knowledge:
    """The semantic layer of a pack whose only model is dim_companies."""
    doc = ModelDoc.model_validate(
        {
            "name": "dim_companies",
            "description": "Companies.",
            "columns": [
                {"name": "name", "description": "Name."},
                {"name": "employees", "description": "Employees."},
            ],
        }
    )
    metric = Metric(
        name="employees",
        description="Employees.",
        table="dim_companies",
        sql="sum(employees)",
        unit="persons",
    )
    return Knowledge({"dim_companies": doc}, [metric], Ontology(), Questions())


def test_ingest_prints_the_warehouse_and_fails_when_data_is_stale(data, monkeypatch):
    built = Build(path=data / "w.duckdb", warnings=["test x: 1 failing row"], stale=["source a"])
    monkeypatch.setattr("vardex.cli.build", lambda *args, **kwargs: built)
    result = runner.invoke(app, ["ingest", PACK])
    assert result.exit_code == 1  # the warehouse is in place, but a scheduler should notice
    assert str(data / "w.duckdb") in result.stdout
    assert "Warning: test x: 1 failing row" in result.stderr


def test_a_failed_build_is_red(data, monkeypatch):
    def fails(*args, **kwargs):
        raise WarehouseError("dbt build failed")

    monkeypatch.setattr("vardex.cli.build", fails)
    result = runner.invoke(app, ["ingest", PACK])
    assert result.exit_code == 1
    assert "dbt build failed" in result.output


def test_a_question_back_is_answered_at_the_keyboard(data, monkeypatch):
    make_warehouse(data)
    back = {"action": "clarify", "sql": None, "message": "Most of what?", "assumptions": []}
    plan = {"action": "query", "sql": SQL, "message": None, "assumptions": []}
    model = FakeModel(json.dumps(back), json.dumps(plan), "Demo Laks AS: 120.")
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.setattr("vardex.cli.load_knowledge", lambda pack: knowledge())
    monkeypatch.setattr("vardex.cli.interactive", lambda: True)

    result = runner.invoke(app, ["answer", PACK, "Who has the most?"], input="Employees.\n")

    assert result.exit_code == 0
    assert "Most of what?" in result.stdout
    assert [(m.role, m.text) for m in model.requests[1].messages] == [
        ("user", "Who has the most?"),
        ("assistant", json.dumps(back)),
        ("user", "Employees."),
    ]
    assert "Who has the most?\nEmployees." in model.requests[2].messages[0].text


def test_a_question_back_without_a_keyboard_ends_with_2(data, monkeypatch):
    make_warehouse(data)
    back = {"action": "clarify", "sql": None, "message": "Most of what?", "assumptions": []}
    model = FakeModel(json.dumps(back))
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.setattr("vardex.cli.load_knowledge", lambda pack: knowledge())
    monkeypatch.setattr("vardex.cli.interactive", lambda: True)

    result = runner.invoke(app, ["answer", PACK, "Who has the most?", "--no-input"])

    assert result.exit_code == 2
    assert result.stdout.startswith("Most of what?")
    assert len(model.requests) == 1
