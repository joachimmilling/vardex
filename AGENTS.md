# Instructions for coding agents

Vardex is an open-source framework for enterprise AI apps: an engine (src/vardex),
recipes, and domain packs (packs/). Each release is described in CHANGELOG.md.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint and format: `uv run ruff check --fix && uv run ruff format`
- Run the CLI: `uv run vardex --help`

## Rules
- The engine in src/vardex never imports a pack or mentions one by name.
  tests/test_architecture.py enforces this. Never weaken or skip that test.
- Validate all external data (files, API responses, model output) with Pydantic.
- Tests never call real APIs. Pass fake clients instead.
- Never read, print or commit .env. Use .env.example for variable names.
- Add dependencies with `uv add`, never by editing pyproject.toml by hand or with pip.
- Every change comes with tests. Run the full test suite and ruff before finishing.
- Describe every change a user would notice under "Unreleased" in CHANGELOG.md.