"""Search: two rankings, fused, and reranked."""

import pytest
from fakes import FakeEmbedder, FakeReranker, MemoryIndex

from vardex.documents import DocumentSpec
from vardex.embeddings import cosine, nearest
from vardex.retrieval import Filters, Passage, fuse, in_context, search


def document(id_="demo-q2-2026", about="999999999", language="en"):
    return DocumentSpec(
        id=id_,
        title=f"Report {id_}",
        about=about,
        kind="quarterly-report",
        period="2026-Q2",
        language=language,
        url=f"https://reports.demo.no/{id_}.pdf",
    )


def passage(pid):
    return Passage(pid, document(), pid, f"text {pid}")


def test_cosine_compares_directions_not_lengths():
    assert cosine([1, 0], [5, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_nearest_puts_the_most_alike_first_and_keeps_ties_in_order():
    assert nearest([1, 0], [[0, 1], [1, 0], [2, 0], [1, 1]], 3) == [1, 2, 3]


def test_fusion_adds_one_over_k_plus_rank_from_each_ranking():
    a, b, c = passage(1), passage(2), passage(3)
    fused = fuse([[a, b], [b, c]], k=60)
    assert [p.id for p, _ in fused] == [2, 1, 3]  # second and first beats first alone
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 61)


def test_a_passage_is_embedded_and_reranked_with_its_document_and_page():
    assert in_context("Demo Report", 4, "Revenue rose.") == "Demo Report, page 4\n\nRevenue rose."


@pytest.fixture
def index():
    index = MemoryIndex()
    embedder = FakeEmbedder()
    texts = [
        (1, "Salmon harvest rose to a record"),
        (2, "The board proposes a dividend"),
        (3, "Gas production fell"),
    ]
    index.put(document(), "f1", texts, embedder.embed_documents([t for _, t in texts]))
    other = [(1, "Laks harvest i Norge")]
    index.put(
        document("demo-nb", about="888888888", language="nb"),
        "f2",
        other,
        embedder.embed_documents([t for _, t in other]),
    )
    return index


def test_search_finds_by_meaning_and_by_words(index):
    found = search(index, FakeEmbedder(), "salmon harvest", limit=2)
    first = found.hits[0]
    assert first.passage.text == "Salmon harvest rose to a record"
    assert (first.meaning_rank, first.words_rank) == (1, 1)
    assert [h.passage.document.id for h in found.hits] == ["demo-q2-2026", "demo-nb"]


def test_filters_reach_both_searches(index):
    found = search(index, FakeEmbedder(), "harvest", Filters(language="nb"))
    assert [h.passage.document.id for h in found.hits] == ["demo-nb"]
    assert {f.language for _, f, _ in index.searches} == {"nb"}


def test_the_reranker_reorders_the_candidates(index):
    found = search(index, FakeEmbedder(), "the board proposes a dividend", reranker=FakeReranker())
    assert found.hits[0].passage.text == "The board proposes a dividend"
    assert found.hits[0].rerank == 5.0


def test_candidates_bound_both_searches(index):
    search(index, FakeEmbedder(), "gas", limit=1, candidates=7)
    assert [limit for _, _, limit in index.searches] == [7, 7]


def test_the_reranker_reads_only_the_best_fused_passages(index):
    class Counting(FakeReranker):
        read = []

        def score(self, query, texts):
            self.read.append(len(texts))
            return super().score(query, texts)

    reranker = Counting()
    found = search(
        index, FakeEmbedder(), "salmon harvest dividend gas", reranker=reranker, rerank=2
    )
    assert reranker.read == [2]
    assert [h.rerank is None for h in found.hits] == [False, False, True, True]
