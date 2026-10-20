# 0019 Embed and rerank with open-weight models on the machine

Status: Accepted
Date: 2026-10-16

## Context

Indexing embeds every passage, and every search embeds the question and reranks a shortlist.
The documents may be a company's own and confidential. The provider Vardex uses for answers
has no embedding model. Evals need the same vectors every run.

## Options

- An embedding and reranking API, such as Voyage, OpenAI, Gemini or Cohere.
- Open-weight models run locally, through fastembed and ONNX Runtime, without PyTorch.

## Decision

`vardex.embeddings` defines `Embedder` (`embed_documents`, `embed_query`, `model`,
`dimensions`) and `Reranker` (`score`). `fastembed_models.py`, the only module that imports
fastembed, implements them with `intfloat/multilingual-e5-large` and
`BAAI/bge-reranker-v2-m3`, downloaded once into the data folder. It adds the prefixes the E5
models were trained with. `connect_embedder` and `connect_reranker` in the CLI are the only
places that choose them; `VARDEX_EMBEDDING_MODEL` and `VARDEX_RERANK_MODEL` change them, and
`VARDEX_RERANK_MODEL=none` turns reranking off.

## Consequences

- No text leaves the machine for indexing or search, there is no key and no cost per call, and
  vectors are the same every run.
- Indexing is slow on a CPU: about 2,400 passages in 16 minutes for the large E5 model.
  Reranking 20 passages takes about six seconds.
- The models take about 4.5 GB on disk, and load in a few seconds per command.
- Changing the embedding model means rebuilding the index; the index refuses vectors from
  another model.

## When to revisit

When indexing time matters more than where the text goes, or when a hosted model is clearly
better on the pack's own search suite.
