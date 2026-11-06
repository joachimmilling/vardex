# 0033 OpenTelemetry for traces, Langfuse to view them

Status: Accepted
Date: 2026-10-28

## Context

When an answer is wrong, the call log says a model was called and what it cost. It does not
say which question the call belonged to, which attempt it was, which query ran before it, or
what the model was sent. Answering "why did it say that?" needs every step of one request,
in order, with its inputs, outputs and timings: a trace.

Companies that adopt Vardex already keep traces somewhere — Azure Monitor under Microsoft
Foundry, Datadog, Grafana, a Langfuse of their own — and will not run a second system for
one application. Whatever Vardex writes has to arrive in theirs.

## Options

- Vardex's own format, in its own files, with its own viewer.
- A vendor's SDK (Langfuse's, LangSmith's) called from the engine.
- OpenTelemetry, with the attribute names of its conventions for generative AI and
  databases, sent over OTLP to whatever backend is configured.

## Decision

Vardex writes spans with OpenTelemetry, from one module, `tracing.py`; a test keeps every
other module from importing it. Attribute names follow the GenAI and database conventions
(`gen_ai.usage.input_tokens`, `db.query.text`), written out as strings, because the GenAI
conventions are still in development and their Python constants are deprecated. Where the
conventions have no name — cost, an attempt's number, a check's verdict — Vardex uses
`vardex.*`.

Every span goes to local JSON-lines files, one per day, which Vardex reads back itself.
When OpenTelemetry's standard variables name an endpoint, spans also go there over OTLP. For
a person who wants to look at traces, the recommended viewer is Langfuse: open source (MIT),
self-hostable, with an EU region, and an OTLP endpoint that maps the GenAI attributes.

## Consequences

- Any OTLP backend works with no change to Vardex: Langfuse, Foundry, Jaeger, Grafana.
- The local files make tracing work with no account, offline and in tests, and are what
  the dashboard and the feedback loop read; reading back from another backend is one class.
- Two price tables can disagree: Langfuse prices calls from its own table unless told the
  cost. `vardex.cost_usd`, from `pricing.py`, is the number Vardex stands behind.
- The GenAI conventions will change names before they are stable. Changes are in one module.

## When to revisit

When the GenAI conventions are declared stable (rename what moved); when a backend in use
cannot accept OTLP; or when the local files grow past what a day's file can be read in.
