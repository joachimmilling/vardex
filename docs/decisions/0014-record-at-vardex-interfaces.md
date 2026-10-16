# 0014 Record and replay at Vardex's own interfaces

Status: Accepted
Date: 2026-10-13

## Context

An eval run calls a model and a warehouse. Neither is available in CI: there is no API key in
a pull request from a fork, the pack's warehouse is built from hundreds of megabytes of open
data, and a company's warehouse is behind its firewall. A live run also costs money and gives
a slightly different result each time. Yet a change to the code around the model, such as the
checks, the tracing or the grading, should be tested against real model output on every pull
request.

## Options

- Run evals only by hand, with a key, and never in CI.
- Record HTTP traffic with a cassette library such as vcrpy, and replay it.
- Record at Vardex's own interfaces, `ModelClient` and `Warehouse`: every request's fingerprint
  with the `Answer` it got, and every query with its `Result` or error.

## Decision

`vardex.replay` wraps a model client and a warehouse in recording versions during a live run,
and saves what they saw to `evals/recordings/<suite>.jsonl` in the pack. A replay builds a
client and a warehouse from the file. A request is matched by a fingerprint of the model, the
system prompt, every message, the limits, the effort and the output schema; a query by its SQL
and row limit. Anything not recorded raises `NotRecorded`, and the run stops with the advice to
record again.

## Consequences

- A replay needs no key, no warehouse and no money, and gives the same report every time.
- A recording holds Vardex's own words: no API key, no headers, no provider's wire format. It
  works for any provider and any warehouse, because it sits above them.
- Any change to what a model is sent, from a prompt to a glossary entry to the date, makes the
  recording stale. That is the point: such a change must be measured live and recorded again.
- A replay tests the code around the model, not the model. It cannot say how a new prompt
  would do; only a live run can.
- Recordings are data in the repository and grow with the golden set: about 200 KB for 63
  cases.

## When to revisit

When recordings grow too large for the repository, or when a provider's own features, such as
tool calls run on its servers, must be replayed and are not visible at `ModelClient`.
