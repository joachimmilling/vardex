# 0017 Keep passages and their vectors in PostgreSQL with pgvector

Status: Accepted
Date: 2026-10-16

## Context

Answers from documents need an index: every passage of every document, with its vector, its
words for keyword search, its page and its document's metadata, searchable by meaning and by
words, with filters on company, kind, period and language. The index is Vardex's own store,
unlike a warehouse: Vardex writes it. A pack's index holds thousands of passages; a company's
could hold millions.

## Options

- A database built for vectors, such as Qdrant, Weaviate or Pinecone, beside the metadata.
- DuckDB with its vector and full-text extensions, in a file, like the pack's warehouse.
- PostgreSQL with the pgvector extension: passages, vectors, a full-text column and metadata in
  one database, behind Vardex's own `Index` interface.

## Decision

`vardex.retrieval.Index` is what the engine needs from an index: put and remove a document's
passages, list what is indexed, and search by meaning or by words with `Filters`.
`pgvector_index.PostgresIndex` implements it in a schema per pack, with an HNSW index on the
vectors (cosine distance), a stored `tsvector` stemmed in each document's language with a GIN
index, and a table of documents for the metadata. It records the embedding model it was built
with and refuses to open with another. `VARDEX_INDEX_URL` names the database.

## Consequences

- One query filters, searches and joins; everything is SQL a person can inspect.
- A company that runs PostgreSQL needs nothing new; anyone else runs one container. Tests run
  against a real PostgreSQL with pgvector in CI, and against `MemoryIndex` everywhere else.
- Approximate search needs two settings per query, `hnsw.ef_search` and `hnsw.iterative_scan`,
  or a filtered search quietly returns fewer passages.
- PostgreSQL's text ranking is not BM25; the evals measure whether that matters.

## When to revisit

When an index grows past what one PostgreSQL serves comfortably, at hundreds of millions of
vectors, or when a feature only a vector database has, such as built-in multi-vector search, is
needed.

Three things arrive well before that size, and any one of them is reason enough to revisit:

- A search filtered by who may see each passage. Narrow permissions leave an HNSW walk few rows
  it may return, and iterative scan then approaches a sequential scan.
- Sustained concurrent searches. An HNSW search is CPU-bound and uses one core, so a few hundred
  searches a second need read replicas and an instance of their own.
- A deployment that already runs a search platform, with people and procurement behind it.
  Writing an `Index` for it costs less than running a second store beside it.
