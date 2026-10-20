"""Indexing a pack: fetch, read, cut, embed and store; then only what changed."""

import httpx2
from fakes import FakeEmbedder, FakeModel, MemoryIndex
from pdfs import scanned_pdf, text_pdf

from vardex.indexing import index_pack

SOURCES = """
about: Company
documents:
  - id: demo-annual-report-2025
    title: Demo Laks AS Annual Report 2025
    about: "999999999"
    kind: annual-report
    period: "2025"
    language: en
    url: https://reports.demo.no/annual-2025.pdf
  - id: demo-arsregnskap-2000
    title: Demo Laks AS Årsregnskap 2000
    about: "999999999"
    kind: annual-accounts
    period: "2000"
    language: nb
    url: https://register.demo.no/kopi/2000
"""
FILES = {
    "/annual-2025.pdf": text_pdf([["Salmon harvest rose to a record."], ["The board proposes."]]),
    "/kopi/2000": scanned_pdf(),
}


def make_pack(tmp_path, sources=SOURCES):
    pack = tmp_path / "pack"
    (pack / "documents").mkdir(parents=True)
    (pack / "pack.yaml").write_text(
        "name: demo-pack\ndescription: A pack for tests.\nrecipes: [knowledge-assistant]\n"
    )
    (pack / "documents" / "sources.yaml").write_text(sources, encoding="utf-8")
    return pack


def http(requests):
    def handler(request):
        requests.append(request.url.path)
        return httpx2.Response(200, content=FILES[request.url.path])

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def run(pack, data, index, model, rebuild=False):
    requests, lines = [], []
    done = index_pack(
        pack, data, http(requests), index, FakeEmbedder(), model, lambda png: "Side 1",
        rebuild=rebuild, say=lines.append,
    )  # fmt: skip
    return done, requests, lines


def test_every_document_is_fetched_read_and_indexed(tmp_path):
    index = MemoryIndex()
    done, requests, lines = run(make_pack(tmp_path), tmp_path / "data", index, FakeModel("Side 1"))
    assert (done.documents, done.pages, done.passages, done.scanned) == (2, 3, 3, 1)
    assert len(done.answers) == 1
    assert sorted(requests) == ["/annual-2025.pdf", "/kopi/2000"]
    assert (tmp_path / "data/demo-pack/documents/demo-arsregnskap-2000.pdf").is_file()
    assert "demo-arsregnskap-2000: 1 pages, 1 read from a picture, 1 passages" in lines
    texts = {(p.document.id, p.page): p.text for p, _, _ in index.rows}
    assert texts[("demo-arsregnskap-2000", 1)] == "Side 1"


def test_a_passage_is_embedded_with_its_document_and_page(tmp_path):
    embedder = FakeEmbedder()
    index_pack(
        make_pack(tmp_path), tmp_path, http([]), MemoryIndex(), embedder, FakeModel("Side 1"),
        lambda png: "Side 1",
    )  # fmt: skip
    _, texts = embedder.calls[0]
    assert texts[0] == "Demo Laks AS Annual Report 2025, page 1\n\nSalmon harvest rose to a record."


def test_a_second_run_reads_nothing_that_has_not_changed(tmp_path):
    pack, index = make_pack(tmp_path), MemoryIndex()
    run(pack, tmp_path, index, FakeModel("Side 1"))
    done, requests, _ = run(pack, tmp_path, index, FakeModel())
    assert (done.documents, done.unchanged, requests) == (0, 2, [])
    done, _, _ = run(pack, tmp_path, index, FakeModel("Side 1"), rebuild=True)
    assert done.documents == 2


def test_a_document_taken_off_the_list_leaves_the_index(tmp_path):
    index = MemoryIndex()
    run(make_pack(tmp_path), tmp_path, index, FakeModel("Side 1"))
    shorter = SOURCES[: SOURCES.index("  - id: demo-arsregnskap-2000")]
    done, _, lines = run(make_pack(tmp_path / "again", shorter), tmp_path, index, FakeModel())
    assert done.removed == 1
    assert set(index.documents()) == {"demo-annual-report-2025"}
    assert "demo-arsregnskap-2000: removed, no longer in the pack's list" in lines


def test_without_a_model_scanned_pages_are_left_and_named(tmp_path):
    done, _, _ = run(make_pack(tmp_path), tmp_path, MemoryIndex(), None)
    assert done.warnings == [
        "demo-arsregnskap-2000: 1 scanned pages were not read: that needs a model"
    ]


def test_a_number_ocr_did_not_find_is_a_warning(tmp_path):
    model = FakeModel("Utbytte 50.000", "Utbytte 50.000")
    done, _, _ = run(make_pack(tmp_path), tmp_path, MemoryIndex(), model)
    assert done.warnings == ["demo-arsregnskap-2000, page 1: the OCR program did not find 50000"]
