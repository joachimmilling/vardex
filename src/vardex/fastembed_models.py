"""The Embedder and Reranker for open-weight models run on this machine, through fastembed.

This is the only module in Vardex that imports fastembed; a test enforces it. The models are
downloaded once, into the folder given, and run on the CPU: no key, no cost per call, and no
text leaves the machine.
"""

from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from vardex.embeddings import Vector

# What each model was trained to see before a question and before a passage. E5 models were
# trained with these words in front, and score worse without them; fastembed does not add them.
PREFIXES = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}
BATCH = 16  # texts per run of the model: larger is faster, up to the memory it takes


class FastEmbedder:
    """Embeds texts with an open-weight model. The model is loaded on first use, because
    loading takes seconds and some commands never embed anything."""

    def __init__(self, model: str, cache_dir: Path) -> None:
        known = {m["model"]: m["dim"] for m in TextEmbedding.list_supported_models()}
        if model not in known:
            raise ValueError(f"fastembed has no embedding model {model!r}")
        self.model = model
        self.dimensions = known[model]
        self.cache_dir = cache_dir
        self.query_prefix, self.document_prefix = PREFIXES.get(model, ("", ""))
        self._loaded: TextEmbedding | None = None
        self.queries: dict[str, Vector] = {}  # a query asked again, such as an example's, is free

    def loaded(self) -> TextEmbedding:
        if self._loaded is None:
            self._loaded = TextEmbedding(self.model, cache_dir=str(self.cache_dir))
        return self._loaded

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        texts = [self.document_prefix + text for text in texts]
        return [v.tolist() for v in self.loaded().embed(texts, batch_size=BATCH)]

    def embed_query(self, text: str) -> Vector:
        if text not in self.queries:
            vector = next(iter(self.loaded().embed([self.query_prefix + text])))
            self.queries[text] = vector.tolist()
        return self.queries[text]


class FastReranker:
    """Scores passages against a question with an open-weight cross-encoder."""

    def __init__(self, model: str, cache_dir: Path) -> None:
        known = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        if model not in known:
            raise ValueError(f"fastembed has no reranker {model!r}")
        self.model = model
        self.cache_dir = cache_dir
        self._loaded: TextCrossEncoder | None = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._loaded is None:
            self._loaded = TextCrossEncoder(self.model, cache_dir=str(self.cache_dir))
        return [float(s) for s in self._loaded.rerank(query, texts, batch_size=BATCH)]
