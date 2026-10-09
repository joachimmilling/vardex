from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from vardex.cli import app
from vardex.warehouse import Build, WarehouseError

runner = CliRunner()
PACK = str(Path(__file__).parent.parent / "packs" / "norwegian-companies")


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("VARDEX_DATA_DIR", str(tmp_path))
    return tmp_path


def test_sql_prints_the_result(data):
    path = data / "norwegian-companies" / "warehouse.duckdb"
    path.parent.mkdir()
    with duckdb.connect(path) as con:
        con.execute("create table dim_companies as select 'DEMO LAKS AS' as name")
    result = runner.invoke(app, ["sql", PACK, "select name from dim_companies"])
    assert result.exit_code == 0
    assert "DEMO LAKS AS" in result.output


def test_sql_cannot_write(data):
    path = data / "norwegian-companies" / "warehouse.duckdb"
    path.parent.mkdir()
    duckdb.connect(path).close()
    result = runner.invoke(app, ["sql", PACK, "create table t (x integer)"])
    assert result.exit_code == 1
    assert "read-only" in result.output


def test_sql_before_any_ingest(data):
    result = runner.invoke(app, ["sql", PACK, "select 1"])
    assert result.exit_code == 1
    assert "Run vardex ingest first" in result.output


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
