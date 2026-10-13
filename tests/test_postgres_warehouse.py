"""PostgresWarehouse, against a fake connection and, when one is configured, a real database."""

from contextlib import nullcontext
from types import SimpleNamespace

import psycopg
import pytest
from conftest import create_tables

from vardex.postgres_warehouse import PostgresWarehouse
from vardex.sql import QueryError


class FakeCursor:
    def __init__(self, connection, name):
        self.connection = connection
        self.name = name
        self.description = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        self.connection.executed.append((self.name, sql))
        if self.connection.error:
            raise self.connection.error
        self.description = [SimpleNamespace(name="n")]

    def fetchmany(self, size):
        return [(i,) for i in range(min(size, 5))]


class FakeConnection:
    """Stands in for a psycopg connection, and keeps what was asked of it."""

    def __init__(self, error=None):
        self.error = error
        self.executed = []
        self.read_only = False
        self.info = SimpleNamespace(user="reader", host="db.demo.no", port=5432, dbname="dw")

    def execute(self, sql):
        self.executed.append((None, sql))
        return SimpleNamespace(fetchall=lambda: [("t", "a", "integer"), ("t", "b", "text")])

    def transaction(self):
        return nullcontext()

    def cursor(self, name=None):
        return FakeCursor(self, name)

    def close(self):
        pass


def connect_to(fake: FakeConnection):
    def connect(url, **kwargs):
        assert kwargs == {"autocommit": True, "connect_timeout": 10}
        return fake

    return connect


def test_it_reads_only_with_a_time_limit_and_names_itself_without_a_password():
    fake = FakeConnection()
    warehouse = PostgresWarehouse(
        "postgresql://reader:secret@db.demo.no/dw", timeout_seconds=5, connect=connect_to(fake)
    )
    assert fake.read_only
    assert (None, "set statement_timeout = 5000") in fake.executed
    assert warehouse.name == "PostgreSQL reader@db.demo.no:5432/dw"
    assert "secret" not in warehouse.name


def test_a_query_runs_in_a_server_side_cursor_and_stops_at_max_rows():
    fake = FakeConnection()
    warehouse = PostgresWarehouse("postgresql://dw", connect=connect_to(fake))
    result = warehouse.run("select n from t", max_rows=3)
    assert fake.executed[-1] == ("vardex", "select n from t")  # a named cursor is server-side
    assert result.rows == [(0,), (1,), (2,)]
    assert result.truncated


def test_tables_are_read_from_the_information_schema():
    warehouse = PostgresWarehouse("postgresql://dw", connect=connect_to(FakeConnection()))
    [table] = warehouse.tables()
    assert (table.name, [c.name for c in table.columns]) == ("t", ["a", "b"])


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (psycopg.errors.QueryCanceled("canceling statement"), "ran for more than 30 s"),
        (psycopg.errors.InsufficientPrivilege("permission denied"), "permission denied"),
    ],
)
def test_database_errors_become_query_errors(error, message):
    warehouse = PostgresWarehouse("postgresql://dw", connect=connect_to(FakeConnection(error)))
    with pytest.raises(QueryError) as caught:
        warehouse.run("select 1")
    assert message in str(caught.value)


def test_a_failed_connection_is_a_query_error():
    def refuse(url, **kwargs):
        raise psycopg.OperationalError("connection refused")

    with pytest.raises(QueryError) as caught:
        PostgresWarehouse("postgresql://dw", connect=refuse)
    assert "could not connect to PostgreSQL: connection refused" in str(caught.value)


def test_a_real_database(postgres_url):
    create_tables(
        postgres_url,
        "create table companies (orgnr varchar(9), name text);"
        "insert into companies values ('100000008', 'DEMO LAKS AS');",
    )
    warehouse = PostgresWarehouse(postgres_url, timeout_seconds=1)
    try:
        assert [t.name for t in warehouse.tables()] == ["companies"]
        assert warehouse.run("select name from companies").rows == [("DEMO LAKS AS",)]
        for sql in [
            "drop table companies",
            "select 1; drop table companies",
            "with gone as (delete from companies returning *) select * from gone",
            "select pg_sleep(3)",
        ]:
            with pytest.raises(QueryError):
                warehouse.run(sql)
        assert warehouse.run("select count(*) from companies").rows == [(1,)]
    finally:
        warehouse.close()
