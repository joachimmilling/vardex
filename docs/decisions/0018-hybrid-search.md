# 0018 Search by meaning and by words, fuse by rank, then rerank

Status: Accepted
Date: 2026-10-16

## Context

A search must find the passage that answers a question asked in other words, often in another
language, and also the passage that holds an exact name, code or organisation number. Search by
meaning, with embeddings, finds paraphrases and misses identifiers. Keyword search finds
identifiers and misses paraphrases. Their scores are on different scales.

## Options

- Search by meaning alone.
- Search by words alone.
- Both, combined by a weighted sum of normalised scores.
- Both, combined by reciprocal rank fusion, with an optional reranker on the shortlist.

## Decision

`retrieval.search` asks the index for the 50 nearest passages by meaning and the 50 best by
words, under the same filters, and fuses the two rankings by reciprocal rank fusion with k = 60:
each passage scores the sum of 1 / (60 + rank) over the rankings it is in. With a `Reranker`,
the fused top 20 are scored by a cross-encoder that reads the question with each passage, the
document's title and page above it, and reordered. The best `limit` are returned, each with its
two ranks, its fused score and its rerank score, so a person can see why it was found.

## Consequences

- Questions in Norwegian find English passages, and organisation numbers find their documents.
- Fusion needs no tuning and no score normalisation.
- On the pack's 47 labelled questions, fusion alone found an answering page in the top 10 for
  85%, with a mean reciprocal rank of 0.61; reranking the top 20 raised that to 98% and 0.90.
  Reranking the top 50 found all 47, at three times the time.
- The reranker costs about 0.3 seconds a passage on a laptop's CPU, six seconds a question;
  `--no-rerank` turns it off.

## When to revisit

When the search suite shows that keyword search or the reranker adds nothing on a pack's
documents, or when a learned fusion beats rank fusion on a large labelled set.
