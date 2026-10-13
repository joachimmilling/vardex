# 0012 Trace every number in an answer to a query result

Status: Accepted
Date: 2026-10-12

## Context

The model writes the answer in words from the rows a query returned. It can still add a number
of its own: a sum done in its head, an average it estimated, a figure it remembered. Such a
number reads like the others. Asking the model to cite its sources does not help, because the
citations come from the same model.

## Options

- Trust the answer, and show the rows beside it.
- Ask the model to return the answer with a citation for each number.
- Check the answer deterministically: find every number in its text and trace it to a cell of
  the result, exactly or rounded to the digits shown, to the SQL or to the question.

## Decision

`vardex.provenance.trace` reads every number in the answer, in English and Norwegian notation,
with percent signs and scale words, and returns a `Figure` for each with the cells it can be
traced to. An answer with a number that traces to nothing is shown to the model once with the
numbers named, and rejected if it still has one. The CLI prints each figure beside the row and
column it came from.

## Consequences

- An answer never shows a number no query returned; computing in SQL becomes the only way.
- Provenance is a type: a `Cell` names the result, row and column of a value.
- A traced number can still be the wrong cell, such as last year's revenue for this year's. The
  trace proves where a number came from, not that it answers the question.
- Numbers written as words, such as "seven", are not traced.

## When to revisit

When answers combine documents and data, and a figure may come from a passage instead of a row.
