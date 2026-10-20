"""A stand-in for a model, so tests never call a real API or spend money."""

from decimal import Decimal

from vardex.llm import Answer
from vardex.pricing import Usage


class FakeModel:
    """A ModelClient that replies with prepared texts, in order, and keeps every request."""

    def __init__(self, *replies: str, stop_reason: str = "end") -> None:
        self.replies = list(replies)
        self.requests = []
        self.model = "claude-sonnet-5-5"
        self.stop_reason = stop_reason

    def send(self, request, on_text=None):
        self.requests.append(request)
        text = self.replies.pop(0)
        if on_text:
            on_text(text)
        return Answer(
            text=text,
            model=self.model,
            usage=Usage(input_tokens=1000, output_tokens=200),
            thinking_tokens=150,
            stop_reason=self.stop_reason,
            seconds=1.5,
            cost_usd=Decimal("0.004"),  # 1,000 × $2 + 200 × $10, per million
            first_text_seconds=0.5 if on_text else None,
        )

    def count_tokens(self, text):
        return 17


class FakeEmbedder:
    """An Embedder that counts words: each text becomes a vector of how often each word of a
    small vocabulary occurs in it. Texts that share words point the same way."""

    model = "fake-embedder"

    def __init__(self, vocabulary=("salmon", "laks", "harvest", "oil", "gas", "dividend", "bank")):
        self.vocabulary = vocabulary
        self.dimensions = len(vocabulary) + 1
        self.calls = []

    def vector(self, text):
        words = text.lower().split()
        counts = [float(sum(w.strip(".,?") == v for w in words)) for v in self.vocabulary]
        return [*counts, 0.1]  # never all zero: every text has some direction

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return self.vector(text)


class FakeReranker:
    """A Reranker that scores a text by how many of the query's words it holds."""

    model = "fake-reranker"

    def score(self, query, texts):
        words = set(query.lower().split())
        return [float(len(words & set(text.lower().split()))) for text in texts]


class MemoryIndex:
    """An Index in memory: meaning by cosine, words by shared words. Keeps what was asked."""

    name = "memory"

    def __init__(self):
        self.rows = []  # (passage, vector, fingerprint)
        self.searches = []
        self.next_id = 1

    def documents(self):
        return {p.document.id: fingerprint for p, _, fingerprint in self.rows}

    def put(self, document, fingerprint, pages, vectors):
        from vardex.retrieval import Passage

        self.remove(document.id)
        for (page, text), vector in zip(pages, vectors, strict=True):
            self.rows.append((Passage(self.next_id, document, page, text), vector, fingerprint))
            self.next_id += 1

    def remove(self, document):
        self.rows = [row for row in self.rows if row[0].document.id != document]

    def matching(self, filters):
        fields = ("about", "kind", "period", "language")
        return [
            (p, v)
            for p, v, _ in self.rows
            if all(getattr(filters, f) in (None, getattr(p.document, f)) for f in fields)
            and filters.document in (None, p.document.id)
        ]

    def by_meaning(self, vector, filters, limit):
        from vardex.embeddings import cosine

        self.searches.append(("meaning", filters, limit))
        found = sorted(self.matching(filters), key=lambda pv: (-cosine(vector, pv[1]), pv[0].id))
        return [p for p, _ in found][:limit]

    def by_words(self, query, filters, limit):
        self.searches.append(("words", filters, limit))
        words = set(query.lower().split())
        scored = [(len(words & set(p.text.lower().split())), p) for p, _ in self.matching(filters)]
        found = sorted((s for s in scored if s[0]), key=lambda sp: (-sp[0], sp[1].id))
        return [p for _, p in found][:limit]

    def close(self):
        pass
