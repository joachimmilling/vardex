# 0021 Read scanned pages with a model, and check every number with OCR

Status: Accepted
Date: 2026-10-16

## Context

Some pages have no text layer: old filings are scanned paper, and the Register of Company
Accounts sends every filed copy as pictures. An OCR program reads characters well and loses the
layout of tables. A vision model reads layout as a person does, and may write a number that is
not on the page.

## Options

- OCR alone.
- A vision model alone.
- A layout model run locally.
- A vision model to read, and OCR to check its numbers.

## Decision

A page with a picture and no text is drawn at 1,568 pixels on its long side.
`documents.read_scan` sends it to the model with `READ_PAGE`, which asks for the page written
out exactly, tables as rows. Tesseract reads the same picture in Norwegian and English, as one
column of lines. Every number in the model's text must be among the numbers Tesseract read,
allowing for the gaps OCR loses between columns. A number that is not is shown to the model
once with `READ_PAGE_REPAIR`; one still there after that is kept, and reported with its document
and page by `vardex index`.

## Consequences

- Scanned tables are indexed as tables, and a misread digit is caught when the two readers
  disagree.
- A disagreement is not proof: OCR misses numbers too, above all handwriting, so a person
  decides.
- Each scanned page costs a model call, about a cent, at indexing time only.
- A `Message` can now carry pictures; a replay matches a picture by its fingerprint.

## When to revisit

When a local model reads scans as well as the vision model, or when OCR's misses make the
warnings too many to read.
