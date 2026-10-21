# Changelog

All notable changes to Vardex are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- A search suite looks 50 passages deep, grades on the top k as before, and says how far down a
  missed page was: `found at rank 26, below the top 10` or `not among the 50 passages ranked`.
  The report adds `Missed: 2 ranked lower, 0 not ranked at all`, which separates a ranking
  problem from a reading one.

## [0.7.0] - 2026-10-20

### Added

- `vardex index PACK` fetches the documents in a pack's `documents/sources.yaml`, reads them
  page by page and stores their passages, vectors and stemmed words in an index in PostgreSQL
  with pgvector (`VARDEX_INDEX_URL`). Only new or changed documents are read; `--rebuild` reads
  them all. A page with no text, as on a scan, is read by the model from a picture, and every
  number in the reading is checked against Tesseract's OCR; numbers the two disagree on are
  reported.
- `vardex search PACK QUERY` searches the documents by meaning and by words, fuses the two
  rankings by reciprocal rank fusion and reranks the best 20, with filters `--about`, `--kind`,
  `--period`, `--language` and `--document`.
- `vardex cite PACK QUESTION` answers from the best eight passages. Every claim cites a
  passage, every quote is checked word for word against its passage, and every number is traced
  to a quote; each source is printed with its document, page, quote and a link to the page.
- `vardex extract` reads PDFs, scanned or not, with `--pages`, and gives every quoted value the
  page its quote is on.
- `Embedder` and `Reranker` interfaces, with open-weight models run locally through fastembed:
  `intfloat/multilingual-e5-large` and `BAAI/bge-reranker-v2-m3` (`VARDEX_EMBEDDING_MODEL`,
  `VARDEX_RERANK_MODEL`). `Index`, with `PostgresIndex`. `Quote` and `trace_quotes` in
  provenance. Messages can carry pictures.
- Eval suites with `capability: search` (recall at k and mean reciprocal rank, no model) and
  `capability: cite` (behaviour, the pages cited, and a faithfulness judge with the new
  `FAITHFUL` prompt). Recordings also hold vectors, rerank scores and searches, so CI replays
  them with no index and no embedding model. The norwegian-companies pack has fifteen documents
  and both suites.
- The engine prompts `READ_PAGE`, `READ_PAGE_REPAIR`, `CITE`, `CITE_REPAIR` and `FAITHFUL`.
- Decision records 0017 to 0023.

### Changed

- The text-to-SQL prompt shows the `show_examples` worked examples whose questions mean the most
  like the question, from the pack's pool (`choose_examples: similar`), or the first ones. The
  norwegian-companies pack has a pool of 24 and shows four.
- `ANSWER` no longer names a currency; a pack's `style` in `questions.yaml` is added to it. The
  norwegian-companies pack's style asks for amounts with their currency.
- `vardex eval` loads the semantic layer only for text-to-SQL suites, so a pack of documents
  alone needs none.
- CI runs PostgreSQL with pgvector. The architecture test also finds a pack's name written with
  underscores.

## [0.6.1] - 2026-10-16

### Added

- `vardex eval --repeat N` runs every case N times in a row. A case passes only when every run
  passed (pass^k), and the report lists the flaky cases, with each failed run's reasons.
- A recording keeps each case's verdict. `vardex eval --baseline FILE` compares a run with a
  recording case by case: the cases broken and fixed, new and removed, and McNemar's exact test.
- `vardex add-case PACK SUITE QUESTION --type` adds a case to a golden set: it answers the
  question live and adds it, with its SQL, only when you confirm the answer; with `--expect
  clarify|refuse` it adds the case without running anything.
- Suites with `capability: extract` measure an extraction: cases name a document in the
  pack's `evals/documents/` and the values a right extraction gives, or `expect: problem`,
  graded in code by `behaviour` and `fields`. The norwegian-companies pack has `key-figures`,
  with ten pages. `ExtractionError.reported` tells a problem the model reported from a failed
  check.

## [0.6.0] - 2026-10-16

### Added

- `vardex eval PACK [SUITE]` runs a pack's golden sets through text-to-SQL, exactly as
  `vardex answer` would, grades each case with code (behaviour and rows) and a model as judge,
  and prints the pass rate by type and language with 95% Wilson intervals, every failure, the
  answers a check sent back, and the cost. It exits with 1 when a suite falls below its
  threshold or a critical case fails.
- `--record` saves every model answer and query result of a run to the pack's
  `evals/recordings/`, and `--replay` plays it back with no API key and no warehouse. A request
  that was not recorded stops the replay.
- `--details` shows each failure's question, SQL and reply; `--case` runs only some cases.
- Packs may keep golden sets in `evals/<suite>.yaml`: cases with reference SQL, or that must be
  asked back or refused, a fixed `today`, a threshold and critical cases. The
  norwegian-companies pack has 63 cases in English and Norwegian.
- The engine prompt `JUDGE`.
- `ModelError.retryable` says whether the same request may work later: true for an overloaded
  or rate-limited provider, a timeout or a lost connection. `Ledger` is a model client that
  keeps every answer, so calls made before a failure can still be logged.
- CI replays the evals on every push and pull request.
- Decision records 0013 to 0016.

### Changed

- The trace accepts a number from the plan's assumptions, which are shown beside every answer;
  reads the Norwegian *billioner* (10¹²); and reads numbers with a scale word in the question
  and SQL. `ANSWER` and `ANSWER_REPAIR` name the assumptions.
- A cache read on Claude Sonnet 5.5 costs $0.10 per million tokens, not $0.20. Prices checked
  13 October 2026.
- An eval case that meets a retryable model error fails instead of stopping the run.
  `vardex answer` and `vardex extract` log the calls already made when a model error stops them.
- norwegian-companies: `fct_accounts` leaves out companies that have left the register, so
  every row has its company.

## [0.5.1] - 2026-10-13

### Added

- Objects in `ontology.yaml` may list `values`: columns whose values are shown to the
  model, as the warehouse spells them, so filters match.
- The pack's metrics and examples are rewritten in the warehouse's SQL dialect, both in the
  prompt and when the semantic layer is checked.
- A query that returns no rows is shown to the model once more, in case a filter is spelt
  differently in the data. This second look does not count as a repair.
- `vardex answer` waits for an answer when the model asks a question back, up to twice;
  without a keyboard, or with `--no-input`, it prints the question and exits with code 2.
- `vardex answer` warns when the answer spells a name unlike the rows, such as Leròy for
  LERØY.

## [0.5.0] - 2026-10-13

### Added

- `vardex answer PACK QUESTION` answers a question about a pack's data, in English or Norwegian:
  a model writes one SQL query against the pack's semantic layer, Vardex checks it and runs it
  read-only, and the model answers from the rows. Every answer shows its assumptions, the SQL,
  the rows and the row and column each number came from. The model may instead ask one
  question back, or refuse.
- Every number in an answer must trace to a query result, the SQL or the question; an answer
  with a number that traces to nothing is repaired once, then rejected.
- `Warehouse`, a provider-neutral connection to a warehouse, with `DuckDBWarehouse` and
  `PostgresWarehouse`. `--warehouse` or `VARDEX_WAREHOUSE_URL` points Vardex at a PostgreSQL
  database instead of the pack's own file.
- `vardex check PACK` checks a pack's semantic layer against a warehouse.
- Packs may declare `warehouse/ontology.yaml`, their objects, keys and links over existing
  tables, and `warehouse/questions.yaml`, a glossary and example questions with their SQL.
- `examples/copy_to_postgres.py` copies a pack's described tables into PostgreSQL.
- The engine prompts `WRITE_SQL`, `SQL_REPAIR`, `ANSWER` and `ANSWER_REPAIR`.
- Decision records 0009 to 0012.

### Changed

- `vardex sql` runs through the same read-only connection, with a time limit, and accepts
  `--warehouse` and `--max-rows`. A DuckDB warehouse can no longer read or write other files.
- Every ingest checks the whole semantic layer: described columns, metrics, keys, links and
  examples. Metric definitions moved from `vardex.warehouse` to `vardex.semantic`.
- CI also runs the tests against PostgreSQL.

## [0.4.1] - 2026-10-12

### Added

- Downloads send the last ETag back; a file the server reports unchanged is not downloaded again.
- The `norwegian-companies` warehouse converts results to NOK at Norges Bank's average rates.
- A source with keys and history lands an unchanged answer as one short row with status 304.
- A keyed fetch stops, and lands nothing, when more than `max_failures` of its keys fail.

## [0.4.0] - 2026-10-09

### Added

- `vardex ingest PACK` fetches the pack's sources that are due, lands every fetch as a file in
  `data/<pack>/landing/`, and rebuilds the pack's DuckDB warehouse from scratch with its dbt
  project. The new warehouse replaces the old one only when every model, test and metric passes;
  a failed fetch falls back on the newest landed file.
- `vardex sql PACK QUERY` runs SQL on a pack's warehouse, read-only.
- Packs declare their sources in `warehouse/ingest.yaml`: a file to download, or a URL fetched once
  per key, with `refresh_days` and `history`. Rate limits, overloads and dropped connections are
  retried.
- Packs define metrics in `warehouse/metrics.yaml`, and every ingest checks that each one runs.
- The `norwegian-companies` warehouse: the Central Coordinating Register for Legal Entities, the
  accounts of limited companies with at least 50 employees, SN2025 industries with a map from
  SN2007, counties, and Statistics Norway's figures by industry, with dbt tests on keys, units,
  the industry map and freshness, and eight metrics.
- `VARDEX_DATA_DIR` sets where fetched files and warehouses are kept (default `data`).
- Decision records in `docs/decisions/`.

### Changed

- Every YAML file in a pack is read by `vardex.packs.read_yaml`, with one line per problem.

## [0.3.1] - 2026-10-01

### Added

- `vardex chat`: type a line, and the model answers it with the whole conversation so far,
  streamed as it is written. An empty line or the end of input ends the chat, and the number of
  turns and the total cost are printed; `--stats` also shows each turn. A failed call prints the
  error and changes nothing, so the next line can carry on; so does an answer without text. The conversation is cached with
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