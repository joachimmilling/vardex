from pathlib import Path

import duckdb
import pytest

from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.sql import QueryError


@pytest.fixture
def warehouse(tmp_path) -> DuckDBWarehouse:
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute("create table companies (orgnr varchar, name varchar, employees integer)")
        con.execute("insert into companies values ('100000008', 'DEMO LAKS AS', 120)")
        con.execute("insert into companies values ('100000016', 'DEMO DATA AS', 300)")
        con.execute("create schema raw")
        con.execute("create table raw.units (x varchar)")
    connection = DuckDBWarehouse(path, timeout_seconds=1)
    yield connection
    connection.close()


def test_tables_lists_the_main_schema_only(warehouse):
    [table] = warehouse.tables()
    assert table.name == "companies"
    assert [(c.name, c.type) for c in table.columns] == [
        ("orgnr", "VARCHAR"),
        ("name", "VARCHAR"),
        ("employees", "INTEGER"),
    ]


def test_run_returns_rows_and_where_they_came_from(warehouse):
    result = warehouse.run("select name, employees from companies order by employees desc")
    assert result.columns == ["name", "employees"]
    assert result.rows == [("DEMO DATA AS", 300), ("DEMO LAKS AS", 120)]
    assert not result.truncated
    assert result.warehouse.startswith("DuckDB ")


def test_run_fetches_at_most_max_rows(warehouse):
    result = warehouse.run("select * from range(10)", max_rows=3)
    assert len(result.rows) == 3
    assert result.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "drop table companies",
        "insert into companies values ('1', 'X', 1)",
        "select * from read_csv('/etc/hosts')",
        "copy companies to 'stolen.csv'",
        "set enable_external_access = true",
        "install httpfs",
    ],
)
def test_it_cannot_write_or_reach_outside_the_file(warehouse, sql):
    with pytest.raises(QueryError):
        warehouse.run(sql)


def test_a_slow_query_is_stopped_and_the_connection_still_works(warehouse):
    with pytest.raises(QueryError) as caught:
        warehouse.run("select count(*) from range(100000000000) as t(x) where x % 7 = 3")
    assert "more than 1 s" in str(caught.value)
    assert warehouse.run("select count(*) from companies").rows == [(2,)]


def test_no_file_no_warehouse(tmp_path: Path):
    with pytest.raises(QueryError) as caught:
        DuckDBWarehouse(tmp_path / "warehouse.duckdb")
    assert "Run vardex ingest first" in str(caught.value)
