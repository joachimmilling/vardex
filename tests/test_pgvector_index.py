"""PostgresIndex, against a real PostgreSQL with pgvector when one is configured."""

import psycopg
import pytest
from fakes import FakeEmbedder

from vardex.documents import DocumentSpec
from vardex.pgvector_index import PostgresIndex, as_text, conditions
from vardex.retrieval import Filters, SearchError

PACK = "demo-pack"  # its passages go in the schema pack_demo_pack


def report(id_, language="en", about="999999999"):
    return DocumentSpec(
        id=id_,
        title=f"Demo report {id_}",
        about=about,
        kind="annual-report",
        period="2025",
        language=language,
        url=f"https://reports.demo.no/{id_}.pdf",
    )


def test_a_vector_is_written_as_pgvector_reads_it():
    assert as_text([0.5, -1.0, 2]) == "[0.5,-1.0,2.0]"


def test_only_the_filters_that_are_set_become_conditions():
    where, params = conditions(Filters(about="999999999", language="nb"))
    assert params == ["999999999", "nb"]
    assert "d.about = %s and d.language = %s" in where.as_string()
    assert conditions(Filters())[0].as_string() == "true"


@pytest.fixture
def index(postgres_url):
    with psycopg.connect(postgres_url, autocommit=True) as con:
        con.execute("drop schema if exists pack_demo_pack cascade")
    embedder = FakeEmbedder()
    index = PostgresIndex(postgres_url, PACK, embedder.model, embedder.dimensions)
    english = [(3, "Salmon harvest rose to a record"), (9, "The board proposes a dividend")]
    norwegian = [(5, "Bankens utbytte og laks"), (6, "Rapporterte resultater for banken")]
    for document, texts in [(report("en-2025"), english), (report("nb-2025", "nb"), norwegian)]:
        index.put(document, "fp", texts, embedder.embed_documents([t for _, t in texts]))
    yield index
    index.close()


def test_documents_are_listed_with_their_fingerprints(index):
    assert index.documents() == {"en-2025": "fp", "nb-2025": "fp"}


def test_nearest_by_meaning_first(index):
    found = index.by_meaning(FakeEmbedder().embed_query("salmon harvest"), Filters(), 4)
    assert (found[0].document.id, found[0].page) == ("en-2025", 3)
    assert len(found) == 4


def test_words_are_stemmed_in_each_documents_language(index):
    # "rapportert" and "Rapporterte" share a Norwegian stem; "dividends" and "dividend" an
    # English one
    assert [p.page for p in index.by_words("rapportert", Filters(), 5)] == [6]
    assert [p.page for p in index.by_words("dividends paid", Filters(), 5)] == [9]


def test_any_word_of_a_question_can_match(index):
    found = index.by_words("Which company proposes a dividend this year?", Filters(), 5)
    assert [p.page for p in found] == [9]


def test_filters_apply_to_both_searches(index):
    vector = FakeEmbedder().embed_query("laks")
    assert {p.document.id for p in index.by_meaning(vector, Filters(language="nb"), 5)} == {
        "nb-2025"
    }
    assert index.by_words("salmon", Filters(document="nb-2025"), 5) == []


def test_putting_a_document_again_replaces_its_passages(index):
    index.put(report("en-2025"), "fp2", [(1, "Gas")], FakeEmbedder().embed_documents(["Gas"]))
    assert index.documents()["en-2025"] == "fp2"
    assert [p.text for p in index.by_words("gas salmon", Filters(), 5)] == ["Gas"]


def test_a_removed_document_takes_its_passages_with_it(index):
    index.remove("en-2025")
    assert index.by_words("salmon", Filters(), 5) == []


def test_vectors_from_another_model_are_refused(index, postgres_url):
    with pytest.raises(SearchError, match="built with fake-embedder"):
        PostgresIndex(postgres_url, PACK, "another-model", 8)
