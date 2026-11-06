# 0034 Hooks around every model call

Status: Accepted
Date: 2026-10-28

## Context

Several things need to see every model call: the trace, the call log, and whatever is
added next (a rate limiter, a redactor). Until now each was done where a command happened to
have the answer in hand, so the call log missed calls that failed and calls made deep inside
an agent run, and adding one more meant editing every command.

## Options

- Call each concern from every command, as before.
- Put tracing and logging inside each provider's client.
- One more ModelClient wrapper per concern, nested by hand where clients are built.
- `Hooked`: a ModelClient that runs a list of hooks around every call, each a function of
  `(model, request, send)` that must return what `send` returned.

## Decision

`llm.Hooked` with a tuple of hooks, the first outermost. The command-line tool builds every
client through `watched`, which adds the trace hook and the call-log hook. A hook may look,
time and refuse; it may not change the request or the answer. The provider's client is not
touched, so a second provider is traced by the same hook.

`Budget` stays a wrapper of its own. It belongs to one run — one question, one scheduled run —
and callers read what it has spent, while hooks belong to the process. It sits outside the
hooks, so a call the budget refuses never reaches them and makes no span.

## Consequences

- Every call is traced and logged, including those that fail, wherever they are made.
- A new concern is one function and one entry in `watched`.
- Order matters and is visible in one place: tracing outermost, so the log line of a call
  carries the trace it belongs to.
- HTTP retries happen inside the provider's SDK, below the hook. They are traced by the
  provider's client, through the SDK's own middleware, as child spans of the call.

## When to revisit

When a concern needs to change a request (redacting personal data before it is sent):
that is a different contract from watching, and should be named as such.
