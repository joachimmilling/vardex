# 0037 Connectors are plug-ins, found by entry point

Status: Accepted
Date: 2026-11-09

## Context

Every outside system sits behind a Vardex interface (0002, 0009), but the engine still chose
among them by name: `connect` built an Anthropic client, `connect_warehouse` tested a URL for
`postgresql://`. Adding a provider meant editing the engine, and installing Vardex installed
every SDK and driver, so a company that uses one warehouse and one provider got all of them,
and the command-line tool imported all of them before doing anything.

## Options

- Keep the `if`s, and add one per connector.
- A registry in the engine that connector modules add themselves to on import.
- Entry points: each connector is named in a package's metadata under a group
  (`vardex.providers`, `vardex.warehouses`, `vardex.judges`, `vardex.trainers`) and loaded by
  name when a setting asks for it; its SDK or driver is an optional extra.

## Decision

Entry points, read by `vardex.plugins`. A model is named `provider/model`; a warehouse by its
URL's scheme or its file's suffix. Vardex's own connectors are declared in its own
pyproject.toml exactly as another package would declare its own. `anthropic`, `openai`,
`ollama`, `duckdb`, `postgresql` and `mlx` are extras. A test checks that starting the
command-line tool imports no SDK and no driver, and another that `connect` and
`connect_warehouse` name none in their code.

## Consequences

- A provider, warehouse, judge or trainer is added by installing a package, with no change to
  the engine. A company's own connector can stay in a private package.
- `pip install vardex` installs no connector at all; a setting that names one whose extra is
  missing fails with the extra to install.
- A name without a provider still means Anthropic, so existing settings keep working; that
  default is the one place a provider is named, in config.py.
- Entry points are read from installed metadata: a connector added to pyproject.toml works
  only after `uv sync` reinstalls the project.

## When to revisit

When a pack needs to ship a connector of its own (packs as packages are planned): the pack's
package can declare it the same way, but loading then depends on which packs are installed.
