"""The open-weight embedder and reranker, with fastembed's models replaced by stand-ins."""

import pytest

from vardex import fastembed_models
from vardex.fastembed_models import FastEmbedder, FastReranker


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding, and keeps every text it was given."""

    loads = 0

    def __init__(self, model, cache_dir):
        FakeTextEmbedding.loads += 1
        self.texts = []

    @staticmethod
    def list_supported_models():
        return [
            {"model": "intfloat/multilingual-e5-large", "dim": 1024},
            {"model": "plain", "dim": 2},
        ]

    def embed(self, texts, batch_size=16):
        self.texts += texts
        return [FakeVector([float(len(t)), 1.0]) for t in texts]


class FakeCrossEncoder:
    def __init__(self, model, cache_dir):
        pass

    @staticmethod
    def list_supported_models():
        return [{"model": "BAAI/bge-reranker-v2-m3"}]

    def rerank(self, query, texts, batch_size=16):
        return [len(t) for t in texts]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(fastembed_models, "TextEmbedding", FakeTextEmbedding)
    monkeypatch.setattr(fastembed_models, "TextCrossEncoder", FakeCrossEncoder)
    FakeTextEmbedding.loads = 0


def test_e5_sees_the_words_it_was_trained_with(tmp_path):
    embedder = FastEmbedder("intfloat/multilingual-e5-large", tmp_path)
    assert embedder.dimensions == 1024
    embedder.embed_documents(["Laks"])
    embedder.embed_query("Laks?")
    assert embedder.loaded().texts == ["passage: Laks", "query: Laks?"]


def test_a_model_without_prefixes_sees_the_text_as_it_is(tmp_path):
    embedder = FastEmbedder("plain", tmp_path)
    assert embedder.embed_query("Laks?") == [5.0, 1.0]
    assert embedder.loaded().texts == ["Laks?"]


def test_the_model_is_loaded_once_and_only_when_used(tmp_path):
    embedder = FastEmbedder("plain", tmp_path)
    assert FakeTextEmbedding.loads == 0
    embedder.embed_query("a")
    embedder.embed_query("b")
    embedder.embed_query("b")
    assert FakeTextEmbedding.loads == 1
    assert embedder.loaded().texts == ["a", "b"]  # a query asked again is not embedded again


def test_an_unknown_model_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no embedding model 'nope'"):
        FastEmbedder("nope", tmp_path)
    with pytest.raises(ValueError, match="no reranker 'nope'"):
        FastReranker("nope", tmp_path)


def test_the_reranker_scores_each_text(tmp_path):
    reranker = FastReranker("BAAI/bge-reranker-v2-m3", tmp_path)
    assert reranker.score("q", ["ab", "abcd"]) == [2.0, 4.0]
