# Changelog

All notable changes to Vardex are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `vardex chat`: type a line, and the model answers it with the whole conversation so far,
  streamed as it is written. An empty line or the end of input ends the chat, and the number of
  turns and the total cost are printed; `--stats` also shows each turn. A failed call prints the
  error and changes nothing, so the next line can carry on. The conversation is cached with
  automatic caching, through the new `Request.cache_conversation`. `Conversation` in `vardex.llm`
  does the same for code.
- Worked examples in extractions: an `examples` list in `extractions/<name>.yaml`, each a document
  and the values it should give. The model sees them in the system prompt, after the pack's
  instructions, inside `<examples>` tags. Each must pass the same checks as a real answer when the
  pack loads. The `key-figures` extraction has one, with a negative operating profit in TNOK.
- `vardex extract --show-prompt` prints the system prompt, each message and the output schema
  for the first file, exactly as they would be sent, and calls no model, so it needs no API key.
- `vardex extract --repairs N` (default 1, at most 3, 0 turns it off): when an answer fails the
  schema or the checks, the model gets its answer and the list of problems back in a new turn,
  with the new `REPAIR` prompt, and answers again. A problem the model reports itself is never
  repaired. Every call is logged and priced, and `--stats` shows one line per call. If a repair
  call fails, the document fails, and the calls before it are still logged.

### Changed

- `Extraction.answers` and `ExtractionError.answers` hold every answer, in order, instead of
  `answer`.

## [0.3.0] - 2026-10-01

### Added

- `vardex extract PACK NAME FILES...` extracts structured values from documents. A pack describes
  the fields in `extractions/<name>.yaml`; each document gives one JSON line of checked values, or
  a red reason why not.
- The `key-figures` extraction in the `norwegian-companies` pack: revenue, operating profit, year,
  currency, unit and accounts from a page of an annual report, with the words each came from.
- Checks on extracted values: quotes must appear in the document, amounts must match their quotes,
  and required values must be present.
- `ModelClient`, a provider-neutral interface to models, and `AnthropicClient`, its implementation
  for Claude, with a 120-second timeout, two retries and prompt caching.
- `--stats` shows cache reads and writes.
- Examples: ten sample report pages, and one round of tool calling by hand.

### Changed

- `vardex.llm.ask` takes a `ModelClient` instead of settings. Only `anthropic_client.py` imports
  `anthropic`, and a test enforces it.
- The engine's prompts live in `vardex.prompts`. They name no product and no audience; both belong
  to the app built on Vardex.
- `--effort` is ignored on Claude Haiku 4.5 instead of failing.
## [0.2.1] - 2026-10-01

### Added

- `vardex packs` lists every pack in a folder (default `packs/`), with the error for each
  invalid one. It exits with 1 if any pack is invalid.
- A pre-commit configuration that runs ruff lint and format on every commit. Install it with
  `uv run pre-commit install`.
- `vardex ask` appends one JSON line per call to `logs/calls.jsonl`: time, model, tokens,
  seconds and exact cost. If the log cannot be written, it warns and still prints the answer.
- `vardex cost` and `vardex ask --stats` warn in yellow when the price table was last checked
  more than 90 days ago, with the link to check the prices.
- `vardex cost --batch` applies the 50% Message Batches API discount (checked 2026-10-01), and
  `vardex cost --model` shows only one model. A model without a price fails with a clear error.
- `vardex ask --stream` prints the answer as it is written. With `--stats`, it also shows the
  time to the first piece of text.

### Changed

- Invalid pack.yaml files are reported as one readable line per problem, such as
  `languages → item 2: must be one of en, nb, nn (got False)`.

## [0.2.0] - 2026-10-01

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