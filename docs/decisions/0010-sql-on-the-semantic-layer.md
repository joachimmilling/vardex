# 0010 Let the model write SQL against the semantic layer, checked and read-only

Status: Accepted
Date: 2026-10-12

## Context

A model answering a question about data can be given the data, given tools that compute fixed
metrics, or asked to write SQL. Reading data in a prompt does not scale past a few thousand rows
and leaves the arithmetic to the model. Fixed metric tools answer only the questions someone
foresaw. SQL answers almost anything, and almost anything includes the wrong thing: raw tables
with unclear columns, writes, and queries that read files or run for an hour.

## Options

- Give the model raw tables and let it write any SQL.
- Generate SQL from metric definitions only, as dbt's MetricFlow does, with the model choosing
  metrics and filters.
- Let the model write SQL, but only against the models the pack describes, with the pack's
  metrics, ontology, glossary and examples in its prompt; check the SQL before it runs, and run
  it on a connection that cannot write.

## Decision

The model writes one query per question against the described models. `check_sql` parses it
with sqlglot in the warehouse's dialect and refuses anything but one query that reads only those
tables, by name. The warehouse connection is read-only by construction: a read-only DuckDB file
with external access off, or a PostgreSQL account that may only select, in read-only
transactions and server-side cursors. Every query has a time limit and a row limit. A query that
fails a check or the database is shown to the model with the reason, and written again once.

## Consequences

- Questions nobody foresaw can be answered, and every answer shows the SQL it ran.
- What the model may see is what the pack describes; an undescribed table is invisible.
- The check is a filter on shape, not a proof of safety. The database account is the boundary,
  so the account must be read-only even if the check has a bug.
- A wrong query that runs is not caught here. Evals measure how often that happens.

## When to revisit

When evals show the model choosing wrong joins or filters often enough that generating SQL from
metrics, for the questions they cover, would be more accurate.
