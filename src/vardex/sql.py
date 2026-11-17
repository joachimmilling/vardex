"""Talking to a warehouse, without depending on any one database.

Everything in Vardex that queries data goes through a Warehouse. Only the modules that
implement one for a database import that database's driver.
"""

import functools
import itertools
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import sqlglot
from sqlglot import exp

from vardex.tracing import DB_QUERY, DB_ROWS, DB_SYSTEM, span

# Each dialect as OpenTelemetry names the database in db.system.name
SYSTEMS = {"duckdb": "duckdb", "postgres": "postgresql"}


def one_at_a_time[F: Callable[..., Any]](method: F) -> F:
    """For a method that uses a connection two threads must not use at once.

    A database connection is one conversation with the server: two threads talking over it
    at the same time interleave their statements and transactions, and the driver fails in
    ways that look nothing like the cause. The agent runs a turn's tool calls side by side
    and a workflow its cases, so a warehouse or an index is shared between threads. The
    object keeps a `gate`, an RLock, and every method that talks to the server holds it.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self.gate:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def traced_query[F: Callable[..., Any]](method: F) -> F:
    """For a Warehouse's `run`: a span for each query, with its text and how many rows came
    back. The span is named after the database, as OpenTelemetry names a query it cannot
    summarise, and it includes any wait for the connection, because that wait is real."""

    @functools.wraps(method)
    def wrapper(self: Any, sql: str, max_rows: int = 1000) -> Any:
        system = SYSTEMS.get(self.dialect, "other_sql")
        attributes = {DB_SYSTEM: system, DB_QUERY: sql, "vardex.max_rows": max_rows}
        with span(system, attributes, kind="client") as query:
            result = method(self, sql, max_rows)
            query.set(DB_ROWS, len(result.rows))
            query.set("vardex.truncated", result.truncated)
            return result

    return wrapper  # type: ignore[return-value]


class QueryError(Exception):
    """Raised when a query fails: a typo, a missing column, a refused write or a timeout."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # in the database's own words, such as DECIMAL(18,0) or numeric


@dataclass(frozen=True)
class Table:
    name: str
    columns: list[Column]


@dataclass(frozen=True)
class Result:
    """The rows a query returned, and where they came from."""

    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool  # the query returned more rows than were fetched
    warehouse: str  # which warehouse ran it, such as DuckDB data/…/warehouse.duckdb
    seconds: float


class Warehouse(Protocol):
    """What Vardex needs from a warehouse. Any class with these members will do.

    A warehouse is read-only by construction: whatever SQL it is given, it cannot change data.
    """

    name: str  # what and where it is, for people; never a password
    dialect: str  # the SQL dialect, in sqlglot's names: duckdb, postgres, snowflake, …

    def tables(self) -> list[Table]:
        """The tables a query can use, with their columns."""
        ...

    def run(self, sql: str, max_rows: int = 1000) -> Result:
        """Run one query and fetch at most max_rows rows. Raises QueryError if it fails."""
        ...

    def close(self) -> None: ...


# Statements that change something, or reach outside the warehouse. Only a query is allowed.
WRITES = (exp.Into, exp.Lock)


def check_sql(sql: str, dialect: str, tables: set[str]) -> list[str]:
    """Everything that stops a query from being run, as readable sentences. Empty means it may
    run: one query that reads only the named tables, by their names alone."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as err:
        reason = err.errors[0]["description"] if err.errors else str(err)
        return [f"the SQL does not parse: {reason}"]
    except sqlglot.errors.TokenError as err:  # a quote that never closes, before any parsing
        return [f"the SQL does not parse: {err}"]
    if len(statements) != 1:
        return [f"write exactly one statement, not {len(statements)}"]
    query = statements[0]
    if not isinstance(query, exp.Query):
        return [f"only a query may run, not {query.key.upper()}"]
    problems = []
    if any(query.find_all(*WRITES)):
        problems.append("SELECT … INTO and FOR UPDATE are not allowed")
    named = {cte.alias_or_name.lower() for cte in query.find_all(exp.CTE)}
    for table in query.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            problems.append(f"read only from tables, not from {table.this.sql(dialect)}")
        elif table.db or table.catalog:
            problems.append(f"name the table {table.name} without a schema or database")
        elif table.name.lower() not in tables | named:
            problems.append(f"{table.name} is not one of the tables you may use")
    return list(dict.fromkeys(problems))  # each problem once, in order


def transpile(sql: str, read: str, write: str) -> str:
    """SQL written in one dialect, rewritten for another: year(d) in DuckDB is
    EXTRACT(YEAR FROM d) in PostgreSQL, and DuckDB's / always divides with decimals, where
    PostgreSQL's divides two integers into an integer. Unchanged when the two are the same."""
    if read == write:
        return sql
    return sqlglot.transpile(sql, read=read, write=write, pretty="\n" in sql.strip())[0]


def is_number(value: Any) -> bool:
    """True for an int, a float or a Decimal; False for a bool, which Python counts as an int."""
    return isinstance(value, int | float | Decimal) and not isinstance(value, bool)


def same(got: Any, want: Any) -> bool:
    """An extracted value against the expected one: numbers as exact decimals, so 842.6 from
    YAML equals Decimal("842.6"); anything else as it is."""
    if is_number(got) and is_number(want):
        return Decimal(str(got)) == Decimal(str(want))
    return got == want


def normalise(value: Any) -> Any:
    """A value as it is compared: a whole number exactly, any other number to six significant
    digits, so 0.156199 from one query equals 0.15619915 from another; text without spaces at
    either end."""
    if is_number(value):
        number = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
        if number.is_finite() and number == number.to_integral_value():
            return str(int(number))
        return f"{number:.6g}"
    return value.strip() if isinstance(value, str) else value


def compare(reference: Result, result: Result, ordered: bool) -> str | None:
    """Why a result does not hold the reference's rows, or None when it does. The result may
    have more columns than the reference, in any order and under any names, and more rows: a
    query may return context, such as the next nine groups after the one asked for. Every
    reference row must be in it, and with ordered, as its first rows, in the same order."""
    if len(result.rows) < len(reference.rows):
        return f"{len(result.rows)} rows, expected at least {len(reference.rows)}"
    want = [tuple(normalise(v) for v in row) for row in reference.rows]
    got = [tuple(normalise(v) for v in row) for row in result.rows]

    def holds(j: int, i: int) -> bool:
        """True when result column j holds every value of reference column i."""
        return not Counter(row[i] for row in want) - Counter(row[j] for row in got)

    # For each reference column, the result columns that hold all its values
    columns = range(len(result.columns))
    candidates = [[j for j in columns if holds(j, i)] for i in range(len(reference.columns))]
    missing = [reference.columns[i] for i, found in enumerate(candidates) if not found]
    if missing:
        return f"no column holds the values of {', '.join(missing)}"
    in_rows = False
    for choice in itertools.product(*candidates):
        rows = [tuple(row[j] for j in choice) for row in got]
        if ordered and rows[: len(want)] == want:
            return None
        if not Counter(want) - Counter(rows):
            if not ordered:
                return None
            in_rows = True
    if in_rows:
        return "the right rows, in the wrong order"
    return "the right values, but not in the same rows"


def format_value(value: Any) -> str:
    """A value as text: empty for NULL, and numbers as the database returned them."""
    return "" if value is None else str(value)


def format_table(result: Result) -> str:
    """The rows as an aligned text table, numbers on the right, with a line on how many."""
    cells = [[format_value(v) for v in row] for row in result.rows]
    numeric = [
        all(is_number(row[i]) for row in result.rows if row[i] is not None)
        for i in range(len(result.columns))
    ]
    widths = [
        max([len(name), *(len(row[i]) for row in cells)]) for i, name in enumerate(result.columns)
    ]

    def line(values: list[str]) -> str:
        padded = zip(values, widths, numeric, strict=True)
        return "  ".join(v.rjust(w) if right else v.ljust(w) for v, w, right in padded).rstrip()

    lines = [line(result.columns), line(["-" * w for w in widths])]
    lines += [line(row) for row in cells]
    count = f"{len(result.rows):,} row{'' if len(result.rows) == 1 else 's'}"
    lines.append(f"({count}; more not fetched)" if result.truncated else f"({count})")
    return "\n".join(lines)
