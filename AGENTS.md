# Instructions for coding agents

Vardex is an open-source framework for enterprise AI apps: an engine (src/vardex),
recipes, and domain packs (packs/). Each release is described in CHANGELOG.md, and each
main design choice in docs/decisions/.

## Commands
- Install: `uv sync && uv run pre-commit install`
- Test: `uv run pytest`
- Lint and format: `uv run ruff check --fix && uv run ruff format`
- Run the CLI: `uv run vardex --help`
- Build a pack's warehouse: `uv run vardex ingest packs/norwegian-companies`

## Rules
- The engine in src/vardex never imports a pack or mentions one by name.
  tests/test_architecture.py enforces this. Never weaken or skip that test.
- Only src/vardex/anthropic_client.py imports anthropic. Everything else talks to models
  through ModelClient in vardex.llm. tests/test_architecture.py enforces this too.
- Prompts are code. Engine prompts live in src/vardex/prompts.py, domain prompts in packs.
  Describe every prompt change in CHANGELOG.md.
- A pack's sources, SQL models, tests and metrics live in its warehouse/ folder. Every column of
  a model in models/marts is described; tests/test_norwegian_companies.py checks it.
- Validate all external data (files, API responses, model output) with Pydantic.
- Tests never call real APIs. Pass fake model clients, and HTTP clients built on
  httpx2.MockTransport.
- Never read, print or commit .env or anything in data/.
- Add dependencies with `uv add`, never by editing pyproject.toml by hand or with pip.
- Every change comes with tests. Run the full test suite and ruff before finishing.
- Describe every change a user would notice under "Unreleased" in CHANGELOG.md.
- A change to a decision in docs/decisions/ gets a new record that supersedes the old one.
