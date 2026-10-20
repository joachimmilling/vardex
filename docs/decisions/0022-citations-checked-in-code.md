# 0022 Cite every claim, and check the citations in code

Status: Accepted
Date: 2026-10-16

## Context

An answer from documents is only useful if a person can check it. Models can be asked to cite
their sources, but a citation written by the model can point at the wrong passage, quote words
that are not there, or sit beside a number the model worked out. Some providers offer citations
that are checked on their side, each in its own form, and not with structured output or scans.

## Options

- Show the passages beside the answer, without citations.
- Use a provider's citations feature.
- Ask for citations in a structured output, and check them in code.

## Decision

`rag.answer` gives the model the best eight passages, numbered, and asks with `CITE` for a
`CitedAnswer`: whether the passages answer the question, the answer with `[n]` after every
sentence that states a fact, and the words copied from each passage cited. `rag.check` passes
it only when every mark names a passage given, every cited passage has words copied from it,
every copy is in its passage whatever the spacing, and every number in the answer is in a copy,
the question or a cited document's title, traced with `provenance.trace_quotes`. A failure is
shown to the model once; a second failure shows no answer. Each source is printed with its
document, page, quote and a link that opens the PDF at the page.

## Consequences

- An answer never shows a quote that is not on its page, or a number that is in no quote.
- It works the same with any provider, with structured output, and with scanned pages.
- A real quote that does not support its sentence is not caught in code; the faithfulness
  judge measures how often that happens.
- The model's quotes cost output tokens.

## When to revisit

When providers' citations become a common, checkable standard that works with structured output
and with text Vardex supplies itself.
