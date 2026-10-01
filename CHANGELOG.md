# Changelog

All notable changes to Vardex are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `vardex packs` lists every pack in a folder (default `packs/`), with the error for each
  invalid one. It exits with 1 if any pack is invalid.
- A pre-commit configuration that runs ruff lint and format on every commit. Install it with
  `uv run pre-commit install`.
- `vardex ask` appends one JSON line per call to `logs/calls.jsonl`: time, model, tokens,
  seconds and exact cost. If the log cannot be written, it warns and still prints the answer.

### Changed

- Invalid pack.yaml files are reported as one readable line per problem, such as
  `languages → item 2: must be one of en, nb, nn (got False)`.

## [0.2.0] - YYYY-MM-DD

### Added

- `vardex ask --stats` shows the tokens, thinking tokens, time and cost of every answer.
- `vardex ask --model` and `--effort`.
- `vardex tokens` counts the tokens in a text file, and what sending it costs.
- `vardex cost` estimates the monthly cost of a feature on every model.
- A price table for the current Claude models, with exact decimal arithmetic.
- Examples: temperature, similarity, repeating a question, and a sample report in English and
  Norwegian.

### Changed

- `vardex.llm.ask` returns an `Answer` with usage, time and cost instead of a plain string.
- Answers cut off by `max_tokens` now print a warning.

## [0.1.0] - 2026-09-30

### Added

- The `vardex` command-line tool, with `version`, `ask` and `validate`.
- Pack format version 0: a `pack.yaml` with name, version, description, languages and recipes,
  checked by `vardex validate`.
- The first pack, `norwegian-companies`, with its manifest.
- `vardex ask`, which sends a question to Claude.
- Tests, linting and continuous integration on every push, and a Dockerfile.