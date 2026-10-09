# 0005 Transform and test with dbt Core 1.x, run from Vardex

Status: Accepted
Date: 2026-10-05

## Context

Raw data from registers and statistics agencies must become clean, documented tables, and
every build must be tested before anyone queries it: keys unique, codes valid, units right.
The SQL belongs to the pack; running it belongs to the engine.

## Options

- SQL files run in order by Vardex's own code, with tests written in Python.
- dbt Core 1.x, the Python package, with the dbt-duckdb adapter.
- dbt 2.0, the Rust rewrite released in September 2026, distributed as a binary with DuckDB
  built in.

## Decision

Each pack holds a dbt project in `warehouse/`. Vardex runs it with dbt Core 1.12 from PyPI, in a
separate process, and reads the results from dbt's `run_results.json` and `sources.json`. A
failing test stops the build; a test marked `severity: warn` is reported and does not.

## Consequences

- Pack authors write models, tests and documentation in the format most data teams know, and can
  run dbt on the pack by hand.
- Dependencies grow by dbt and its adapter, all locked in `uv.lock` and installed by `uv sync`.
- Each dbt command takes a second or two to start, which is small next to fetching data.

## When to revisit

When dbt 2.0 can be installed and locked like a Python dependency, or when dbt Core 1.x stops
receiving fixes. The pack's SQL and YAML are meant to run on both.
