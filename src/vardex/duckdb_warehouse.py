"""The Warehouse for a DuckDB file, such as the one `vardex ingest` builds for a pack.

With postgres_warehouse.py, the only module in Vardex that queries through a database driver;
a test enforces it. warehouse.py also imports duckdb, but only to build the file.
"""

import threading
import time
from pathlib import Path

import duckdb

from vardex.sql import Column, QueryError, Result, Table, one_at_a_time

# Read-only, and nothing outside the file: no reading or writing other files, no URLs, no
# extensions. Locked, so a query cannot turn any of it back on.
SETTINGS = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}


class DuckDBWarehouse:
    """Queries one DuckDB file, read-only, with a time limit on every query."""

    dialect = "duckdb"

    def __init__(self, path: Path, *, timeout_seconds: float = 30.0) -> None:
        if not path.is_file():
            raise QueryError(f"No warehouse at {path}. Run vardex ingest first.")
        self.gate = threading.RLock()  # one query at a time: see one_at_a_time
        self.name = f"DuckDB {path}"
        self.timeout_seconds = timeout_seconds
        self.con = duckdb.connect(path, read_only=True, config=SETTINGS)

    @one_at_a_time
    def tables(self) -> list[Table]:
        """The tables and views in the file's main schema, with their columns, by name."""
        rows = self.con.execute(
            "select table_name, column_name, data_type from information_schema.columns"
            " where table_schema = 'main' order by table_name, ordinal_position"
        ).fetchall()
        tables: dict[str, list[Column]] = {}
        for table, column, data_type in rows:
            tables.setdefault(table, []).append(Column(column, data_type))
        return [Table(name, columns) for name, columns in tables.items()]

    @one_at_a_time
    def run(self, sql: str, max_rows: int = 1000) -> Result:
        """Run one query and fetch at most max_rows rows. A query that runs past the time
        limit is interrupted."""
        timer = threading.Timer(self.timeout_seconds, self.con.interrupt)
        started = time.perf_counter()
        timer.start()
        try:
            cursor = self.con.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(max_rows + 1)
        except duckdb.InterruptException as err:
            raise QueryError(f"the query ran for more than {self.timeout_seconds:g} s") from err
        except duckdb.Error as err:
            raise QueryError(str(err)) from err
        finally:
            timer.cancel()
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
