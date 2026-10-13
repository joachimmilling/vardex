from decimal import Decimal

import pytest

from vardex.sql import Result, check_sql, format_table, transpile

TABLES = {"dim_companies", "fct_accounts"}


def problems(sql: str, dialect: str = "duckdb") -> list[str]:
    return check_sql(sql, dialect, TABLES)


def test_a_query_on_allowed_tables_passes():
    assert problems("select name from dim_companies where employees > 200 limit 10") == []
    assert (
        problems(
            "with big as (select orgnr from dim_companies where employees > 200) "
            "select a.revenue from fct_accounts as a join big using (orgnr)"
        )
        == []
    )
    assert problems("select 1 from dim_companies union all select 2 from fct_accounts") == []


@pytest.mark.parametrize(
    ("sql", "problem"),
    [
        ("drop table dim_companies", "only a query may run, not DROP"),
        ("delete from dim_companies", "only a query may run, not DELETE"),
        ("insert into dim_companies select * from dim_companies", "not INSERT"),
        ("attach 'other.duckdb' as other", "not ATTACH"),
        ("copy (select 1) to 'out.csv'", "not COPY"),
        ("pragma version", "not PRAGMA"),
        ("set threads = 1", "not SET"),
        ("select 1; drop table dim_companies", "write exactly one statement, not 2"),
        ("select * from raw.units", "name the table units without a schema or database"),
        ("select * from stg_units", "stg_units is not one of the tables you may use"),
        ("select * from read_csv('/etc/hosts')", "read only from tables"),
        ("select * from 'data/secret.csv'", "data/secret.csv is not one of the tables"),
        ("select * into copy from dim_companies", "SELECT … INTO and FOR UPDATE"),
        ("select * from dim_companies for update", "SELECT … INTO and FOR UPDATE"),
    ],
)
def test_anything_but_one_query_on_allowed_tables_is_refused(sql, problem):
    found = problems(sql)
    assert found, f"{sql!r} should be refused"
    assert any(problem in p for p in found), found


def test_sql_that_does_not_parse():
    assert problems("select (1 from dim_companies")[0].startswith("the SQL does not parse")


def test_the_dialect_decides_what_parses():
    top = "select top 5 name from dim_companies"  # SQL Server's way to limit rows
    assert problems(top, "tsql") == []
    assert problems(top, "duckdb")[0].startswith("the SQL does not parse")


def test_rows_as_a_table():
    result = Result(
        sql="select …",
        columns=["name", "revenue"],
        rows=[("DEMO LAKS AS", Decimal("1480312000")), ("DEMO DATA AS", None)],
        truncated=True,
        warehouse="DuckDB test",
        seconds=0.01,
    )
    assert format_table(result).splitlines() == [
        "name             revenue",
        "------------  ----------",
        "DEMO LAKS AS  1480312000",
        "DEMO DATA AS",
        "(2 rows; more not fetched)",
    ]


def test_sql_is_rewritten_for_another_dialect():
    assert transpile("year(period_end)", "duckdb", "postgres") == "EXTRACT(YEAR FROM period_end)"
    # DuckDB's / divides with decimals; PostgreSQL's divides two integers into an integer
    assert transpile("7 / 2", "duckdb", "postgres") == "CAST(7 AS DOUBLE PRECISION) / 2"
    assert transpile("year(d)", "duckdb", "duckdb") == "year(d)"
