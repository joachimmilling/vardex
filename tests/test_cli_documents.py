"""vardex index, search and cite, on an index in memory with fake models."""

import json
from pathlib import Path

import duckdb
import httpx2
import pytest
from fakes import FakeEmbedder, FakeModel, FakeReranker, MemoryIndex
from pdfs import text_pdf
from test_indexing import FILES, make_pack
from test_rag import PASSAGES, REPORT, reply
from typer.testing import CliRunner

from vardex import cli
from vardex.cli import app
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.llm import MissingAPIKeyError

runner = CliRunner()
PACK = str(Path(__file__).parent.parent / "packs" / "norwegian-companies")


@pytest.fixture
def index(monkeypatch, tmp_path):
    monkeypatch.setenv("VARDEX_DATA_DIR", str(tmp_path / "data"))
    index = MemoryIndex()
    texts = [(p.page, p.text) for p in PASSAGES]
    index.put(REPORT, "f", texts, FakeEmbedder().embed_documents([t for _, t in texts]))
    monkeypatch.setattr(cli, "connect_index", lambda settings, pack, embedder: index)
    return index


def test_index_reads_every_document_and_says_what_it_did(index, monkeypatch, tmp_path):
    transport = httpx2.MockTransport(lambda r: httpx2.Response(200, content=FILES[r.url.path]))
    monkeypatch.setattr(cli, "http_client", lambda: httpx2.Client(transport=transport))
    monkeypatch.setattr(cli, "connect", lambda settings: FakeModel("Side 1"))
    monkeypatch.setattr(cli, "tesseract", lambda picture: "Side 1")
    result = runner.invoke(app, ["index", str(make_pack(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "demo-arsregnskap-2000: 1 pages, 1 read from a picture, 1 passages" in result.output
    assert result.output.splitlines()[-1] == (
        "2 indexed, 0 unchanged, 1 removed: 3 pages (1 read from a picture), 3 passages"
    )


def test_index_without_a_key_indexes_the_text_and_warns_about_scans(index, monkeypatch, tmp_path):
    transport = httpx2.MockTransport(lambda r: httpx2.Response(200, content=FILES[r.url.path]))
    monkeypatch.setattr(cli, "http_client", lambda: httpx2.Client(transport=transport))

    def no_key(settings):
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(cli, "connect", no_key)
    result = runner.invoke(app, ["index", str(make_pack(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "1 scanned pages were not read: that needs a model" in result.output


def test_search_prints_each_passage_with_its_ranks_and_link(index, monkeypatch):
    monkeypatch.setattr(cli, "connect_reranker", lambda settings, rerank=True: FakeReranker())
    result = runner.invoke(app, ["search", PACK, "the board dividend"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(
        " 1  Demo Laks AS Annual Report 2025, page 40   (meaning 1 · words 1"
    )
    assert lines[1] == "    https://reports.demo.no/annual-2025.pdf#page=40"


def test_search_passes_the_filters_on(index):
    result = runner.invoke(
        app, ["search", PACK, "dividend", "--about", "999999999", "--language", "nb"]
    )
    assert result.exit_code == 1
    assert "No passage matches" in result.output
    assert {(f.about, f.language) for _, f, _ in index.searches} == {("999999999", "nb")}


def test_cite_prints_the_answer_and_its_sources(index, monkeypatch):
    answer = reply(True, "The board proposes NOK 4.20 per share [1].", [(1, "NOK 4.20 per share")])
    monkeypatch.setattr(cli, "connect", lambda settings: FakeModel(answer))
    result = runner.invoke(app, ["cite", PACK, "What dividend is proposed?"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "The board proposes NOK 4.20 per share [1].",
        "",
        "Sources",
        "[1] Demo Laks AS Annual Report 2025, page 40",
        "    https://reports.demo.no/annual-2025.pdf#page=40",
        '    "NOK 4.20 per share"',
    ]


def test_cite_says_when_the_documents_do_not_answer(index, monkeypatch):
    answer = reply(False, "The passages do not give the chief executive's pay.")
    monkeypatch.setattr(cli, "connect", lambda settings: FakeModel(answer))
    result = runner.invoke(app, ["cite", PACK, "What does the CEO earn?"])
    assert result.exit_code == 0
    assert "not in the documents" in result.output


def test_cite_that_fails_its_checks_is_red(index, monkeypatch):
    wrong = reply(True, "NOK 9 per share [1].", [(1, "NOK 4.20")])
    monkeypatch.setattr(cli, "connect", lambda settings: FakeModel(wrong, wrong))
    result = runner.invoke(app, ["cite", PACK, "What dividend is proposed?"])
    assert result.exit_code == 1
    assert "no answer passed the checks" in result.output


def test_no_index_url_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("VARDEX_INDEX_URL", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: cli.Settings(None, "m", index_url=None))
    result = runner.invoke(app, ["search", PACK, "x"])
    assert result.exit_code == 1
    assert "VARDEX_INDEX_URL is not set" in result.output


def test_extract_from_a_pdf_gives_each_quoted_value_its_page(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    (pack / "extractions").mkdir(parents=True)
    (pack / "extractions" / "revenue.yaml").write_text(
        "description: Revenue.\ninstructions: Find the revenue.\nfields:\n"
        "  revenue: {type: amount, description: Revenue, quote: true}\n"
    )
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(text_pdf([["Front page"], ["Notes"], ["Revenue 1 480 312 1 234 567"]]))
    answer = '{"problem": null, "revenue": 1480312, "revenue_quote": "1 480 312"}'
    model = FakeModel(answer)
    monkeypatch.setattr(cli, "connect", lambda settings: model)
    result = runner.invoke(app, ["extract", str(pack), "revenue", str(pdf), "--pages", "2-3"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "file": "report.pdf",
        "problem": None,
        "revenue": "1480312",
        "revenue_quote": "1 480 312",
        "revenue_page": 3,
    }
    sent = model.requests[0].messages[0].text
    assert '<page number="2">' in sent and "Front page" not in sent


def test_extract_refuses_pages_it_cannot_read(tmp_path):
    result = runner.invoke(app, ["extract", PACK, "key-figures", __file__, "--pages", "four"])
    assert result.exit_code == 1
    assert "is not a page or a range of pages" in result.output


def test_a_company_by_name_becomes_the_about_filter(index, monkeypatch, tmp_path):
    path = tmp_path / "w.duckdb"
    with duckdb.connect(path) as con:
        con.execute("create table dim_companies (orgnr varchar, name varchar)")
        con.execute("insert into dim_companies values ('964118191', 'MOWI ASA')")
    monkeypatch.setattr(cli, "connect_warehouse", lambda url, pack, s: DuckDBWarehouse(path))
    result = runner.invoke(app, ["search", PACK, "salmon", "--company", "mowi"])
    assert "No passage matches" in result.output  # the index in memory holds no Mowi report
    assert {f.about for _, f, _ in index.searches} == {"964118191"}
    both = runner.invoke(app, ["search", PACK, "salmon", "--company", "Mowi", "--about", "1"])
    assert both.exit_code == 1
    assert "Give --company or --about, not both." in both.output


KEY_FIGURES = json.dumps(
    {
        "problem": None,
        "company_name": "Demo Laks AS",
        "fiscal_year": 2025,
        "accounts": "group",
        "currency": "NOK",
        "unit": "thousands",
        "unit_quote": "NOK thousand",
        "revenue": 1480312,
        "revenue_quote": "1 480 312",
        "operating_profit": 350000,
        "operating_profit_quote": "350 000",
    }
)


def test_extract_from_an_indexed_document_reads_the_pages_a_search_finds(
    index, monkeypatch, tmp_path
):
    report = REPORT.model_copy(update={"id": "mowi-q2-2026"})  # a document in the pack's list
    texts = [(2, "The board proposes a dividend."), (3, "Revenue and operating profit")]
    index.put(report, "f", texts, FakeEmbedder().embed_documents([t for _, t in texts]))
    pdf = tmp_path / "data" / "norwegian-companies" / "documents" / "mowi-q2-2026.pdf"
    pdf.parent.mkdir(parents=True)
    statement = ["Demo Laks AS, group, in NOK thousand", "Revenue 1 480 312", "EBIT 350 000"]
    pages = [["Front page"], ["The board proposes a dividend."], statement]
    pdf.write_bytes(text_pdf(pages))
    model = FakeModel(KEY_FIGURES)
    monkeypatch.setattr(cli, "connect", lambda settings: model)
    result = runner.invoke(app, ["extract", PACK, "key-figures", "--document", "mowi-q2-2026"])
    assert result.exit_code == 0, result.output
    values = json.loads(result.output.splitlines()[-1])
    assert (values["document"], values["revenue_page"], values["unit_page"]) == (
        "mowi-q2-2026",
        3,
        3,
    )
    assert "Front page" not in model.requests[0].messages[0].text  # only the pages found


def test_extract_from_a_document_that_is_not_indexed_says_so(index):
    result = runner.invoke(app, ["extract", PACK, "key-figures", "--document", "orkla-q2-2026"])
    assert result.exit_code == 1
    assert "orkla-q2-2026 is not indexed: run vardex index" in result.output
