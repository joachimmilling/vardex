# 0004 Use DuckDB as the warehouse

Status: Accepted
Date: 2026-10-05

## Context

Each pack needs a SQL warehouse: about a million registered entities, tens of thousands of
accounts and a few statistics tables for the first pack, queried by people and, later, by a
model writing SQL. Vardex should run on a laptop, in CI and in a container without a database
server to install, secure or pay for.

## Options

- PostgreSQL: a server, row-oriented, the usual choice for an application database.
- A cloud warehouse such as Snowflake or BigQuery: powerful, but an account, a bill and a
  network round trip for every query, and tests that need credentials.
- DuckDB: an analytical database inside the Python process, stored in one file.

## Decision

Each pack's warehouse is one DuckDB file, `data/<pack>/warehouse.duckdb`. It is rebuilt by
`vardex ingest` and opened read-only by everything that queries it.

## Consequences

- No server. A full rebuild of the first pack takes seconds; tests build real warehouses in a
  temporary folder.
- DuckDB reads CSV, JSON and Parquet directly, including gzipped files, so loading needs no code.
- One process writes at a time. Vardex builds into a new file and swaps it in, so readers are
  never blocked, but two builds of the same pack must not run at once.
- Data size is bounded by one machine's disk, which is far beyond what packs need now.

## When to revisit

When several services must write to the warehouse at the same time, when data outgrows one
machine, or when a customer requires their existing warehouse. dbt models (0005) move to
another database with small changes.
