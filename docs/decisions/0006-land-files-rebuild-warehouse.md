# 0006 Land every fetch as a file, and rebuild the warehouse from the files

Status: Accepted
Date: 2026-10-05

## Context

Sources change and fail. The accounts register's open API returns only the last three years
for each company, so any longer history must be kept by Vardex. Registers correct records after
the fact, APIs go down, and transformation code has bugs that are found after data was loaded.

## Options

- Load each fetch straight into warehouse tables, updating rows in place.
- Keep the warehouse as the only copy, and back it up.
- Land every fetch, unchanged, as a file named by its fetch time, and rebuild the whole
  warehouse from the landed files on every run.

## Decision

`vardex ingest` writes each fetch to `data/<pack>/landing/<source>/<time>.<ext>`, renamed into
place only when complete. Landed files are never edited. A source with `history: true` keeps
every file; others keep the newest. Each run builds a new warehouse from the files in a
separate folder and replaces the old one only when every model, test and metric passes. A
failed fetch falls back on the newest landed file, with a warning.

## Consequences

- The warehouse can be deleted at any time; a fixed model is applied to all history by
  running ingest again.
- History beyond what an API returns exists only in the landing folder, which must be kept and
  backed up. It grows with every fetch of a source with history.
- A rebuild re-reads every file, so build time grows with history; for now it is seconds.

## When to revisit

When rebuilding from all landed files takes too long, load incrementally instead. When landed
files grow too large, keep only fetches that changed something.
