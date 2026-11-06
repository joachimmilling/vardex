# 0035 What was asked and answered stays on the machine unless allowed to leave

Status: Accepted
Date: 2026-10-28

## Context

A trace that can explain a wrong answer has to hold what the model was sent and what it
said: the question, the prompt, the rows, the SQL. Those are a company's questions about its
own business, and sometimes personal data. A tracing backend is often another company's
service in another country. Sending content there is a decision about data, not a default
of a debugging tool.

## Options

- Record no content anywhere; trace timings, tokens and costs only.
- Record everything everywhere.
- Mark the attributes that carry content, keep them in the local files, and send them to a
  backend only when the operator says so.

## Decision

`tracing.CONTENT` lists the attributes that hold what people asked and what the model wrote.
`VARDEX_TRACE_CONTENT` decides where they go: `local` (the default) keeps them in the local
files and strips them from what is sent to a backend; `all` sends them too; `none` keeps them
nowhere. A misspelt value means `local`. Stripping happens in an exporter that rebuilds each
span, so the request that made it is never slowed or changed. Local trace files are deleted
after 30 days.

## Consequences

- Langfuse shows the tree, the timings, the tokens and the costs by default, and not the
  questions. Turning content on for a backend is one setting, chosen knowingly.
- The local files hold content and sit beside the pack's data, which is already private.
- An attribute added later that carries content must be added to `CONTENT`, or it leaks.

## When to revisit

When access rules for the data reach the traces themselves: a person allowed to see a trace
should be one allowed to see the rows in it.
