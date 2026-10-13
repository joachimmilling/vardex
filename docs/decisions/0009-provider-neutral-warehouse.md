# 0009 Query warehouses through one provider-neutral connection; PostgreSQL second

Status: Accepted
Date: 2026-10-12

## Context

Text-to-SQL must run where the data is. For a pack built on public data, that is the DuckDB file
Vardex builds. Inside a company it is the warehouse the data team already runs, such as
PostgreSQL, Snowflake, BigQuery or Databricks, and copying that data into a file is neither
allowed nor wanted. Each database has its own driver, errors, types and SQL dialect.

## Options

- Call DuckDB directly everywhere, and add other databases when someone asks.
- SQLAlchemy, which has dialects for most databases behind one API.
- Vardex's own small `Warehouse` protocol in `vardex.sql`: a name, a dialect, `tables()`,
  `run(sql, max_rows)` and `close()`, with one module per database that imports its driver.

## Decision

The engine queries only through `Warehouse`. `DuckDBWarehouse` and `PostgresWarehouse` are the
first two implementations; `cli.connect_warehouse` is the one place that chooses one. A test
fails if any other engine module imports a database driver (the warehouse builder may import
DuckDB to build its file). PostgreSQL comes second: it is the database companies most often
already run, it runs in Docker for development and in CI as a service, and its read-only roles
and server-side cursors make the read-only promise checkable.

## Consequences

- A company points Vardex at its own warehouse with `--warehouse` or `VARDEX_WAREHOUSE_URL`;
  no data is copied.
- Each implementation owns its read-only setup, time limit and error translation, so `QueryError`
  means the same everywhere.
- Each new database is one module of about 80 lines, plus its dialect name for sqlglot.
- SQLAlchemy's connection pooling and type system are not available; Vardex has not needed them.

## When to revisit

When a third and fourth database are added and their modules repeat each other, or when
connection pooling is needed by a long-running server.
