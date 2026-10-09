# 0001 Keep the engine free of any domain; put each domain in a pack

Status: Accepted
Date: 2026-09-30

## Context

Vardex is meant to build several kinds of AI app for many companies. Each company brings its
own data, terms, prompts and rules. If that knowledge lands in the engine, every new customer
means engine changes, and one customer's details can leak into another's install.

## Options

- One application per domain, copied and changed for each new one.
- A plugin system where domains are Python packages that hook into the engine.
- An engine that knows no domain, and packs: folders of YAML, SQL and prompts that the engine
  reads and validates.

## Decision

The engine in `src/vardex` never imports a pack and never mentions one by name. A pack is a
folder with a `pack.yaml` manifest; everything about its domain lives there. A test fails the
build if the engine names a pack or imports from `packs/`.

## Consequences

- A new domain is a new folder, not a fork. A company can keep its pack in a private repository.
- Every pack file needs a schema and readable errors, because pack authors are not engine
  developers.
- Some features need more design: the engine must offer a general mechanism (extractions,
  sources, metrics) where a single app could have hard-coded one case.

## When to revisit

If packs need logic that YAML and SQL cannot express, consider letting packs ship Python, with
the same rule that the engine never imports them by name.
