# 0023 Choose the worked examples for each question from a pool

Status: Accepted
Date: 2026-10-16

## Context

The text-to-SQL prompt shows worked examples: a question and the SQL that answers it. A few
fixed examples help with the kinds of question they show. More examples help with more kinds,
and make every prompt longer and dearer.

## Options

- A fixed set of examples in every prompt.
- Every example in every prompt.
- A pool, and for each question the examples whose questions mean the most like it.

## Decision

`questions.yaml` holds a pool of examples, `show_examples` (how many a prompt shows) and
`choose_examples` (`similar` or `first`). `text_to_sql.choose_examples` embeds the question and
every example's question as queries, with the same `Embedder` as retrieval, and shows the
`show_examples` nearest, in the pool's order. Without an embedder, or with `first`, it shows the
first ones. The examples are chosen once per conversation, for the question it began with. No
example may be a golden-set question; `check_suite` refuses one.

## Consequences

- The pool can grow without the prompt growing.
- Every question's prompt differs, so the prompt cache holds less: only the part before the
  examples is shared.
- Recordings of the answer suite now hold the question and example vectors.
- Whether it beats a fixed set is measured, case by case, against a run with `first`.

## When to revisit

When the pool is large enough that choosing by the SQL as well as the question, as DAIL-SQL
does, or a learned selector, would help.
