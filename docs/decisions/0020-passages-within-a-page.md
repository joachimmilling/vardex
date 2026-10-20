# 0020 Cut documents into passages within a page, with the document and page above each

Status: Accepted
Date: 2026-10-16

## Context

An index stores passages, not documents: a vector holds one meaning, the model reads only what
it is sent, and a citation must point at a place a person can find. Too large a passage mixes
topics; too small loses its context. Annual reports are long, laid out in pages, and full of
tables whose numbers mean nothing without their column headings.

## Options

- Fixed-size chunks with an overlap, across pages.
- Semantic chunking, cut where the embeddings of consecutive sentences change.
- One passage per page.
- Passages within a page, made of whole lines, up to a size.

## Decision

`documents.read_pdf` reads each page's text layer and cleans it: lines on more than 30% of the
pages are running headers and go; words split at a line end are joined; the lines of a
paragraph become one; a line that looks like a table row keeps its own line.
`documents.passages` packs whole lines into passages of at most 250 words, never across a page;
a long paragraph is cut between sentences, and a long table between rows with its headings
repeated. Each passage is embedded and reranked with its document's title and page above it.

## Consequences

- Every passage cites exactly one page.
- Tables stay readable in their passages, with their headings.
- Passages vary in size, and a short page gives a short passage.
- The heuristics for tables and headers are simple; a layout model would read hard pages
  better, at a cost the search suite can weigh.

## When to revisit

When a pack's documents have no pages, such as web pages, or when the search suite shows that
another way of cutting finds more.
