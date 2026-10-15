"""Text-to-SQL with a fake model, on a real DuckDB file and, when configured, PostgreSQL."""

import json
from datetime import date

import duckdb
import pytest
from conftest import create_tables
from fakes import FakeModel

from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.postgres_warehouse import PostgresWarehouse
from vardex.semantic import Knowledge, Metric, ModelDoc, Ontology, Questions, check
from vardex.text_to_sql import AnswerError, Plan, answer

TABLES = """
create table companies (orgnr varchar(9), name varchar(100), county varchar(30));
insert into companies values ('100000008', 'DEMO LAKS AS', 'Vestland'),
                             ('100000016', 'DEMO DATA AS', 'Oslo');
create table accounts (orgnr varchar(9), fiscal_year integer, period_end date,
                       revenue decimal(18, 0), operating_profit decimal(18, 0));
insert into accounts values ('100000008', 2025, '2025-12-31', 1480312000, 350000000),
                            ('100000016', 2025, '2025-12-31', 900000000, 45000000);
"""
KNOWLEDGE = Knowledge(
    tables={
        "companies": ModelDoc.model_validate(
            {
                "name": "companies",
                "description": "One row per company.",
                "columns": [
                    {"name": "orgnr", "description": "Organisation number."},
                    {"name": "name", "description": "Registered name."},
                    {"name": "county", "description": "County, such as Vestland."},
                ],
            }
        ),
        "accounts": ModelDoc.model_validate(
            {
                "name": "accounts",
                "description": "One row per company and fiscal year.",
                "columns": [
                    {"name": "orgnr", "description": "The company."},
                    {"name": "fiscal_year", "description": "The year it ends in."},
                    {"name": "period_end", "description": "The last day of the year."},
                    {"name": "revenue", "description": "Revenue in NOK."},
                    {"name": "operating_profit", "description": "Operating profit in NOK."},
                ],
            }
        ),
    },
    metrics=[
        Metric(
            name="operating_margin",
            description="Operating profit as a share of revenue.",
            table="accounts",
            sql="sum(operating_profit) / nullif(sum(revenue), 0)",
            unit="share",
            notes=["The margin of the totals, not the average of margins."],
        )
    ],
    ontology=Ontology.model_validate(
        {
            "objects": [
                {
                    "name": "Company",
                    "description": "A company.",
                    "table": "companies",
                    "key": "orgnr",
                    "label": "name",
                    "values": ["county"],
                }
            ]
        }
    ),
    questions=Questions.model_validate(
        {
            "dialect": "duckdb",
            "glossary": {"last year": "The last complete fiscal year."},
            "examples": [
                {
                    "question": "How many sets of accounts end in each year?",
                    "sql": "select year(period_end) as year, count(*) as accounts\n"
                    "from accounts group by year(period_end)",
                }
            ],
        }
    ),
)
SQL = """\
select c.name, sum(a.operating_profit) / nullif(sum(a.revenue), 0) as operating_margin
from accounts as a
join companies as c on c.orgnr = a.orgnr
where a.fiscal_year = 2025
group by c.name
order by operating_margin desc"""
QUESTION = "Which company had the highest operating margin last year?"
TODAY = date(2026, 10, 12)


def plan(action="query", sql=SQL, message=None, assumptions=("Last year is 2025.",)) -> str:
    return json.dumps(
        {"action": action, "sql": sql, "message": message, "assumptions": list(assumptions)}
    )


@pytest.fixture(params=["duckdb", "postgres"])
def warehouse(request, tmp_path):
    if request.param == "duckdb":
        with duckdb.connect(tmp_path / "warehouse.duckdb") as con:
            con.execute(TABLES)
        connection = DuckDBWarehouse(tmp_path / "warehouse.duckdb")
    else:
        url = request.getfixturevalue("postgres_url")
        create_tables(url, TABLES)
        connection = PostgresWarehouse(url)
    yield connection
    connection.close()


def ask(model: FakeModel, warehouse, **kwargs):
    return answer(model, warehouse, KNOWLEDGE, QUESTION, today=TODAY, **kwargs)


def test_the_same_question_on_each_warehouse(warehouse):
    model = FakeModel(plan(), "Demo Laks AS had the highest operating margin in 2025: 23.6%.")
    reply = ask(model, warehouse)

    assert reply.kind == "answer"
    assert reply.result.sql == SQL
    assert [row[0] for row in reply.result.rows] == ["DEMO LAKS AS", "DEMO DATA AS"]
    assert [(f.text, f.traced) for f in reply.figures] == [("2025", True), ("23.6%", True)]
    assert (reply.figures[1].cells[0].row, reply.figures[1].cells[0].column) == (
        1,
        "operating_margin",
    )
    assert reply.assumptions == ["Last year is 2025."]
    assert len(reply.answers) == 2


def test_the_request_holds_the_semantic_layer(warehouse):
    model = FakeModel(plan(), "Demo Laks AS: 23.6%.")
    ask(model, warehouse, max_rows=20)
    request = model.requests[0]
    assert request.output_schema is Plan
    assert request.cache_system
    dialect = "DuckDB" if warehouse.dialect == "duckdb" else "PostgreSQL"
    for expected in [
        f"one SQL query for {dialect}",
        "Today is 2026-10-12.",
        "Return at most 20 rows.",
        '<table name="accounts">',
        "- county (",
        "One row per Company in companies, identified by orgnr, named by name.",
        '<metric name="operating_margin" table="accounts" unit="share">',
        "- The margin of the totals, not the average of margins.",
        "- last year: The last complete fiscal year.",
        "- companies.county: 'Oslo', 'Vestland'",
    ]:
        assert expected in request.system, expected
    assert "<rows>" in model.requests[1].messages[0].text


@pytest.mark.parametrize("action", ["clarify", "refuse"])
def test_a_question_back_or_a_refusal_runs_nothing(warehouse, action):
    model = FakeModel(plan(action, sql=None, message="Company or group accounts?"))
    reply = ask(model, warehouse)
    assert (reply.kind, reply.text, reply.result) == (action, "Company or group accounts?", None)
    assert len(model.requests) == 1


def test_a_query_that_fails_a_check_is_written_again(warehouse):
    model = FakeModel(plan(sql="drop table companies"), plan(), "Demo Laks AS: 23.6%.")
    reply = ask(model, warehouse)
    assert reply.kind == "answer"
    repair = model.requests[1].messages[-1].text
    assert "- only a query may run, not DROP" in repair
    assert companies_left(warehouse) == 2


def test_a_query_the_warehouse_refuses_is_written_again(warehouse):
    model = FakeModel(plan(sql="select profit from accounts"), plan(), "Demo Laks AS: 23.6%.")
    ask(model, warehouse)
    assert "the warehouse refused it" in model.requests[1].messages[-1].text


def test_after_the_last_repair_it_gives_up(warehouse):
    model = FakeModel(plan(sql="select * from secrets"), plan(sql="select * from secrets"))
    with pytest.raises(AnswerError) as caught:
        ask(model, warehouse, repairs=1)
    assert "secrets is not one of the tables you may use" in str(caught.value)
    assert len(caught.value.answers) == 2


def test_an_answer_that_does_not_match_the_schema_is_written_again(warehouse):
    model = FakeModel('{"action": "guess"}', plan(), "Demo Laks AS: 23.6%.")
    assert ask(model, warehouse).kind == "answer"
    assert "action: must be one of query, clarify, refuse" in model.requests[1].messages[-1].text


def test_a_number_no_query_returned_is_repaired_or_rejected(warehouse):
    made_up = "The average margin was 14.3%."
    model = FakeModel(plan(), made_up, "Demo Laks AS had 23.6%.")
    reply = ask(model, warehouse)
    assert reply.text == "Demo Laks AS had 23.6%."
    assert "14.3%" in model.requests[2].messages[-1].text

    model = FakeModel(plan(), made_up, made_up)
    with pytest.raises(AnswerError) as caught:
        ask(model, warehouse)
    assert "the answer has numbers no query returned: 14.3%" in str(caught.value)
    assert caught.value.result.sql == SQL  # so the SQL and rows can still be shown
    assert len(caught.value.answers) == 3  # the plan and two answers, all paid for


def test_a_model_that_stops_early_gets_no_answer(warehouse):
    model = FakeModel(plan(), stop_reason="max_tokens")
    with pytest.raises(AnswerError) as caught:
        ask(model, warehouse)
    assert "stopped early (max_tokens)" in str(caught.value)


def companies_left(warehouse) -> int:
    return warehouse.run("select count(*) from companies").rows[0][0]


def test_a_column_with_too_many_values_is_not_listed(warehouse, monkeypatch):
    monkeypatch.setattr("vardex.text_to_sql.MAX_VALUES", 1)
    model = FakeModel(plan(), "Demo Laks AS: 23.6%.")
    ask(model, warehouse)
    assert "- companies.county: more than 1 values" in model.requests[0].system


def test_the_packs_sql_is_rewritten_for_the_warehouse(warehouse):
    assert check(warehouse, KNOWLEDGE) == []
    model = FakeModel(plan(), "Demo Laks AS: 23.6%.")
    ask(model, warehouse)
    if warehouse.dialect == "postgres":
        assert "EXTRACT(YEAR FROM period_end)" in model.requests[0].system
        assert "SQL: CAST(SUM(operating_profit) AS DOUBLE PRECISION)" in model.requests[0].system
    else:
        assert "year(period_end)" in model.requests[0].system


def test_no_rows_gets_a_second_look(warehouse):
    misspelt = SQL.replace("where a.fiscal_year = 2025", "where c.county = 'VESTLAND'")
    model = FakeModel(plan(sql=misspelt), plan(), "Demo Laks AS had 23.6%.")
    reply = ask(model, warehouse, repairs=0)  # a second look is not a repair
    assert "The query returned no rows." in model.requests[1].messages[-1].text
    assert reply.result.sql == SQL


def test_no_rows_can_be_the_answer(warehouse):
    nobody = SQL.replace("where a.fiscal_year = 2025", "where a.fiscal_year = 1990")
    model = FakeModel(plan(sql=nobody), plan(sql=nobody), "No company filed accounts for 1990.")
    reply = ask(model, warehouse)
    assert reply.result.rows == []
    assert len(model.requests) == 3  # one second look, then the answer


def test_a_misspelt_name_is_a_warning_not_a_rejection(warehouse):
    model = FakeModel(plan(), "Demo Lacs AS had the highest margin, 23.6%.")
    reply = ask(model, warehouse)
    assert reply.kind == "answer"
    assert reply.warnings == ['"Demo Lacs AS" is not in the rows; they spell it "DEMO LAKS AS"']
