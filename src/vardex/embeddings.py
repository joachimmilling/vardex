"""Embeddings: a text as a list of numbers, placed so that texts that mean the same lie close.

Everything in Vardex that embeds or reranks goes through an Embedder or a Reranker. Only the
module that implements one for a library or a provider imports it.
"""

import math
from typing import Protocol

Vector = list[float]


class Embedder(Protocol):
    """What Vardex needs from an embedding model. Any class with these members will do.

    A question and the passages that answer it are embedded differently by some models, so
    there are two methods: one for the texts to search, one for what is searched with.
    """

    model: str  # the model's name; an index records it, and refuses vectors from another
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """One vector per text to be searched, in order."""
        ...

    def embed_query(self, text: str) -> Vector:
        """The vector of a question or a search."""
        ...


class Reranker(Protocol):
    """What Vardex needs from a reranker: a model that reads a question and a passage together
    and scores how well the passage answers it."""

    model: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One score per text, in order: higher means a better answer to the query."""
        ...


def cosine(a: Vector, b: Vector) -> float:
    """How alike two vectors point, from -1 to 1: 1 is the same direction, whatever the length."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def nearest(query: Vector, vectors: list[Vector], n: int) -> list[int]:
    """The positions of the n vectors most like the query, the most alike first. A tie keeps
    the earlier position first, so the same input always gives the same order."""
    scores = [cosine(query, vector) for vector in vectors]
    return sorted(range(len(vectors)), key=lambda i: (-scores[i], i))[:n]
