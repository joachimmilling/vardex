# 0007 Declare sources in YAML, with two generic kinds of fetch

Status: Accepted
Date: 2026-10-05

## Context

Each pack fetches from its own sources: registers, statistics agencies, internal systems. The
engine must fetch, retry, land and schedule them without knowing any of them (0001).

## Options

- Python connectors written in each pack, called by the engine.
- An integration tool such as Airbyte or dlt in front of Vardex.
- Sources declared in `warehouse/ingest.yaml`, fetched by the engine with a small number of
  generic kinds.

## Decision

A source is a name, a URL and a schedule (`refresh_days`). With `keys`, an SQL query over the
raw tables already loaded, the URL is fetched once per key and every answer, success or not,
is landed as one JSON line. Without it, the URL is downloaded as one CSV or JSON file. All
parsing is the job of the pack's SQL.

## Consequences

- A pack author adds a source with a few lines of YAML; retries, landing, scheduling and
  freshness come with it.
- Sources that need authentication, paging or POST requests cannot be declared yet.
- Order matters: a source with keys must come after the sources its query reads.

## When to revisit

When a pack needs a source that the two kinds cannot fetch, add a third kind to the engine, or
consider an integration tool for that pack.
