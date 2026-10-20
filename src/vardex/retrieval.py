"""Retrieval: find the passages in a pack's documents that answer a question.

Two searches run side by side. One compares meanings: the question's vector against every
passage's. The other compares words: the question's words, stemmed, against the passage's.
Each is good where the other is weak: meaning finds a paraphrase in another language, words
find a name, a code or a number exactly. Their two rankings are fused into one, and a reranker,
which reads the question and each passage together, can put the best few on top.
"""

from dataclasses import dataclass, field
from typing import Protocol

from vardex.documents import DocumentSpec
from vardex.embeddings import Embedder, Reranker, Vector

FUSION_K = 60  # reciprocal rank fusion's constant: larger flattens the gap between ranks
RERANK = 20  # the fused passages a reranker reads: each costs about 0.3 s on a laptop's CPU


class SearchError(Exception):
    """Raised when the index cannot be reached, written or searched, or holds vectors from
    another embedding model."""


@dataclass(frozen=True)
class Passage:
    """A piece of one page of a document, as the index keeps it."""

    id: int
    document: DocumentSpec
    page: int
    text: str

    @property
    def link(self) -> str:
        """Where a person can read it: the document's address, opened at the page."""
        return f"{self.document.url}#page={self.page}"


@dataclass(frozen=True)
class Filters:
    """What a passage's document must be. A filter left empty does not filter."""

    about: str | None = None  # the key of the object it is about, such as an organisation number
    kind: str | None = None  # such as annual-report
    period: str | None = None  # such as 2025 or 2026-Q2
    language: str | None = None  # en, nb or nn
    document: str | None = None  # one document, by its id


class Index(Protocol):
    """What Vardex needs from a store of passages and their vectors. Any class with these
    members will do."""

    name: str  # what and where it is, for people; never a password

    def documents(self) -> dict[str, str]:
        """Every indexed document's id, and the fingerprint of the file it was read from."""
        ...

    def put(
        self,
        document: DocumentSpec,
        fingerprint: str,
        pages: list[tuple[int, str]],
        vectors: list[Vector],
    ) -> None:
        """Replace a document's passages with these: (page, text) pairs and their vectors."""
        ...

    def remove(self, document: str) -> None:
        """Remove a document and its passages."""
        ...

    def by_meaning(self, vector: Vector, filters: Filters, limit: int) -> list[Passage]:
        """The passages whose vectors are nearest the vector, the nearest first."""
        ...

    def by_words(self, query: str, filters: Filters, limit: int) -> list[Passage]:
        """The passages that share the most words with the query, the best first."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Hit:
    """A passage a search found, and how it was ranked. Ranks start at 1; None means the
    search did not return it."""

    passage: Passage
    meaning_rank: int | None
    words_rank: int | None
    fused: float  # its reciprocal rank fusion score
    rerank: float | None = None  # the reranker's score, when there was one


@dataclass(frozen=True)
class Search:
    """What one search returned, best first, and what it was asked."""

    query: str
    filters: Filters
    hits: list[Hit] = field(default_factory=list)

    @property
    def passages(self) -> list[Passage]:
        return [hit.passage for hit in self.hits]


def in_context(title: str, page: int, text: str) -> str:
    """A passage as it is embedded and reranked: with its document and page above it, so
    "the group's revenue rose" is known to be Mowi's, in 2025."""
    return f"{title}, page {page}\n\n{text}"


def fuse(rankings: list[list[Passage]], k: int = FUSION_K) -> list[tuple[Passage, float]]:
    """Reciprocal rank fusion: each passage scores 1 / (k + rank) in every ranking it is in,
    and the scores are added. A passage near the top of both beats one at the top of one.
    Only ranks count, never the scores behind them, so rankings on different scales mix."""
    scores: dict[int, float] = {}
    found: dict[int, Passage] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking, start=1):
            scores[passage.id] = scores.get(passage.id, 0.0) + 1 / (k + rank)
            found[passage.id] = passage
    order = sorted(scores, key=lambda pid: (-scores[pid], pid))
    return [(found[pid], scores[pid]) for pid in order]


def search(
    index: Index,
    embedder: Embedder,
    query: str,
    filters: Filters | None = None,
    *,
    limit: int = 10,
    candidates: int = 50,
    reranker: Reranker | None = None,
    rerank: int = RERANK,
) -> Search:
    """Search by meaning and by words, `candidates` passages each, and fuse the two rankings.
    With a reranker, reorder the best `rerank` of them by reading each with the query.
    Returns the best `limit`."""
    filters = filters or Filters()
    by_meaning = index.by_meaning(embedder.embed_query(query), filters, candidates)
    by_words = index.by_words(query, filters, candidates)
    meaning_ranks = {p.id: rank for rank, p in enumerate(by_meaning, start=1)}
    words_ranks = {p.id: rank for rank, p in enumerate(by_words, start=1)}
    hits = [
        Hit(passage, meaning_ranks.get(passage.id), words_ranks.get(passage.id), score)
        for passage, score in fuse([by_meaning, by_words])[:candidates]
    ]
    if reranker and hits:
        top, rest = hits[:rerank], hits[rerank:]
        texts = [in_context(h.passage.document.title, h.passage.page, h.passage.text) for h in top]
        scores = reranker.score(query, texts)
        ranked = sorted(zip(top, scores, strict=True), key=lambda pair: -pair[1])
        hits = [Hit(h.passage, h.meaning_rank, h.words_rank, h.fused, score) for h, score in ranked]
        hits += rest  # below the reranked ones, in their fused order
    return Search(query, filters, hits[:limit])
