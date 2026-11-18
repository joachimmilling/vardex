# 0039 Route on each capability's own checks

Status: Accepted
Date: 2026-11-18

## Context

A small model on our own server costs nothing per call and keeps the data in the building, and
it is wrong more often than Claude. Gemma 4 12B passed 52 of the 66 answer cases (79%) where
Claude passes about 95%, and all ten key-figures cases. Most work could go to the small model if
something could tell, before anyone sees the result, which work it got wrong.

## Options

- Send everything to one model, chosen per pack.
- A classifier that reads the input and picks a model before either runs.
- A cascade: the small model first, and the large one whenever the capability's own checks
  reject the small model's result (SQL that does not parse, a number not in the rows, a quote
  not on the page, a missing required field).

## Decision

The cascade, behind `VARDEX_SMALL_MODEL`, for every capability through `capabilities.py`: each
capability's attempt says whether its checks passed, and the router knows nothing else about
it. Routing is on per command, and its eval decides whether it stays on.

## Consequences

- On the answer suite, the routed run passed 55 of 66 (83%), with 10 cases sent on to Claude.
  Three of Gemma's 14 failures were caught and fixed; the other 11 passed every check and went
  out wrong: a count of 0 where the answer is 20, the wrong industry section, an empty answer.
  For answers, routing is not safe to turn on: the checks catch broken work, not wrong work.
- On the key-figures suite, the routed run passed 10 of 10 with one page sent on. An
  extraction's checks (values quoted word for word, digits matching the quote) are close to a
  definition of right, so routing is viable there.
- Seven of the ten sent on were refusals Gemma had already got right: a refusal counts as giving
  up, so the router asks Claude to refuse again. That costs money and fixes nothing.
- Every escalated request costs both models' time. A capability whose checks fail often is
  slower routed than sent straight to the large model.

## When to revisit

When a capability gains a check that catches wrong-but-valid work (a second query that must
agree, a judge on a sample), or when a new small model's eval shows its failures are mostly
ones the checks see.
