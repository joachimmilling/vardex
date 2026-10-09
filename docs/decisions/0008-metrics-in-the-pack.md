# 0008 Define metrics in the pack, in Vardex's own small format

Status: Accepted
Date: 2026-10-05

## Context

Questions will be answered from the warehouse by people and, soon, by a model writing SQL. A
number such as "operating margin" has traps: which accounts, which currency, ratio of sums or
average of ratios. If every query decides for itself, two answers to the same question differ.

## Options

- No definitions: rely on column descriptions in the dbt project.
- dbt's semantic layer with MetricFlow: semantic models, measures and metrics in dbt's YAML,
  queried through MetricFlow.
- A small `warehouse/metrics.yaml` per pack: for each metric a name, a description, a model, an
  aggregate SQL expression, a unit and notes on how to use it.

## Decision

Each pack defines its metrics in `warehouse/metrics.yaml`. Vardex validates the file and, on
every ingest, runs each metric's SQL against the new warehouse; a metric that does not run
fails the build.

## Consequences

- One definition per number, readable by a person and by a model, owned by the pack.
- Metrics cannot drift from the models: a renamed column fails the next ingest.
- Vardex does not generate SQL from metrics as MetricFlow does; a query must use the expression
  and follow the notes.

## When to revisit

When packs need joins, time grains or derived metrics generated for them, or when users bring
an existing dbt semantic layer. The format is close enough to convert.
