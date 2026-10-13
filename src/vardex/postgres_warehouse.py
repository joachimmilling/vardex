"""The Warehouse for PostgreSQL, through psycopg: a warehouse someone else runs.

With duckdb_warehouse.py, the only module in Vardex that queries through a database driver;
a test enforces it. Connect with an account that may only read: everything here makes writing
hard, but only the database's own permissions make it impossible.
"""

import time
from collections.abc import Callable
from typing import Any

import psycopg

from vardex.sql import Column, QueryError, Result, Table


class PostgresWarehouse:
    """Queries one PostgreSQL database, in read-only transactions, with a time limit."""

    dialect = "postgres"

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 30.0,
        connect: Callable[..., Any] = psycopg.connect,
    ) -> None:
        try:
            self.con = connect(url, autocommit=True, connect_timeout=10)
            # Every query runs in a transaction that cannot write, and stops after the limit.
            self.con.read_only = True
            self.con.execute(f"set statement_timeout = {int(timeout_seconds * 1000)}")
        except psycopg.Error as err:
            raise QueryError(f"could not connect to PostgreSQL: {err}") from err
        info = self.con.info
        self.name = f"PostgreSQL {info.user}@{info.host}:{info.port}/{info.dbname}"
        self.timeout_seconds = timeout_seconds

    def tables(self) -> list[Table]:
        """The tables and views in the current schema that this account may read."""
        try:
            rows = self.con.execute(
                "select table_name, column_name, data_type from information_schema.columns"
                " where table_schema = current_schema() order by table_name, ordinal_position"
            ).fetchall()
        except psycopg.Error as err:
            raise QueryError(str(err).strip()) from err
        tables: dict[str, list[Column]] = {}
        for table, column, data_type in rows:
            tables.setdefault(table, []).append(Column(column, data_type))
        return [Table(name, columns) for name, columns in tables.items()]

    def run(self, sql: str, max_rows: int = 1000) -> Result:
        """Run one query and fetch at most max_rows rows.

        The query runs in a server-side cursor, which PostgreSQL declares with DECLARE …
        CURSOR FOR: that accepts exactly one SELECT or VALUES, so a second statement or a
        write is refused before anything runs, and rows stream instead of all arriving at once.
        """
        started = time.perf_counter()
        try:
            with self.con.transaction(), self.con.cursor(name="vardex") as cursor:
                cursor.execute(sql)
                rows = cursor.fetchmany(max_rows + 1)
                columns = [d.name for d in cursor.description or []]
        except psycopg.errors.QueryCanceled as err:
            raise QueryError(f"the query ran for more than {self.timeout_seconds:g} s") from err
        except psycopg.Error as err:
            raise QueryError(str(err).strip()) from err
        return Result(
            sql=sql,
            columns=columns,
            rows=rows[:max_rows],
            truncated=len(rows) > max_rows,
            warehouse=self.name,
            seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        self.con.close()
