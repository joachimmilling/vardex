"""A stand-in for a model, so tests never call a real API or spend money."""

from dataclasses import dataclass, field
from decimal import Decimal

from vardex.llm import Answer, Reasoning, ToolCall
from vardex.pricing import Usage


@dataclass
class Turn:
    """One prepared turn in which the model asks for tools instead of answering."""

    text: str = ""
    calls: list[tuple[str, dict]] = field(default_factory=list)
    reasoning: str = "Thinking about it."

    def tool_calls(self, step: int) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(id=f"call-{step}-{n}", name=name, arguments=arguments)
            for n, (name, arguments) in enumerate(self.calls, start=1)
        )


class FakeModel:
    """A ModelClient that replies with prepared answers, in order, and keeps every request.

    A reply is a string, which the model answers with, or a Turn, with which it asks for
    tools. Running out of replies raises, so a test that loops for ever fails rather than
    hanging.
    """

    def __init__(self, *replies: str | Turn, stop_reason: str = "end") -> None:
        self.replies = list(replies)
        self.requests = []
        self.model = "claude-sonnet-5-5"
        self.stop_reason = stop_reason
        self.steps = 0

    def send(self, request, on_text=None):
        self.requests.append(request)
        self.steps += 1
        reply = self.replies.pop(0)  # IndexError when a test prepared too few
        turn = reply if isinstance(reply, Turn) else Turn(text=reply)
        if on_text and turn.text:
            on_text(turn.text)
        calls = turn.tool_calls(self.steps)
        return Answer(
            text=turn.text,
            model=self.model,
            usage=Usage(input_tokens=1000, output_tokens=200),
            thinking_tokens=150,
            stop_reason="tool_use" if calls else self.stop_reason,
            seconds=1.5,
            cost_usd=Decimal("0.004"),  # 1,000 × $2 + 200 × $10, per million
            first_text_seconds=0.5 if on_text else None,
            tool_calls=calls,
            reasoning=(Reasoning(turn.reasoning, signature="signed"),) if turn.reasoning else (),
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
