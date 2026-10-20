"""Reading a pack's documents: the list, fetching, pages, passages and scanned pages."""

from pathlib import Path

import httpx2
import pytest
from fakes import FakeModel
from pdfs import scanned_pdf, text_pdf

from vardex.documents import (
    DocumentSpec,
    Page,
    clean,
    digit_runs,
    fetch,
    load_documents,
    passages,
    read_document,
    read_pdf,
    read_scan,
    table_row,
    unconfirmed,
)
from vardex.packs import PackError

DOCUMENT = DocumentSpec(
    id="demo-annual-report-2025",
    title="Demo Laks AS Annual Report 2025",  # a made-up company
    about="999999999",
    kind="annual-report",
    period="2025",
    language="en",
    url="https://reports.demo.no/annual-2025.pdf",
)


def write_list(pack: Path, text: str) -> Path:
    (pack / "documents").mkdir(parents=True)
    (pack / "documents" / "sources.yaml").write_text(text, encoding="utf-8")
    return pack


def test_the_list_of_documents_is_read_and_checked(tmp_path):
    pack = write_list(
        tmp_path,
        """
about: Company
documents:
  - id: demo-annual-report-2025
    title: Demo Laks AS Annual Report 2025
    about: "999999999"
    kind: annual-report
    period: "2025"
    language: en
    url: https://reports.demo.no/annual-2025.pdf
""",
    )
    assert load_documents(pack).documents == [DOCUMENT]


def test_two_documents_with_one_id_are_refused(tmp_path):
    entry = DOCUMENT.model_dump_json()
    pack = write_list(tmp_path, f"about: Company\ndocuments: [{entry}, {entry}]\n")
    with pytest.raises(PackError, match="its own id"):
        load_documents(pack)


def serving(body: bytes, status: int = 200) -> httpx2.Client:
    return httpx2.Client(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(status, content=body))
    )


def test_a_document_is_fetched_once(tmp_path):
    path = tmp_path / "doc.pdf"
    assert fetch(serving(text_pdf([["Revenue"]])), DOCUMENT, path) is True
    assert path.read_bytes().startswith(b"%PDF-")
    assert fetch(serving(b"never asked for"), DOCUMENT, path) is False


def test_a_web_page_instead_of_a_pdf_is_refused(tmp_path):
    with pytest.raises(PackError, match="did not return a PDF"):
        fetch(serving(b"<html>Sign in</html>"), DOCUMENT, tmp_path / "doc.pdf")
    assert not list(tmp_path.iterdir())  # no half file is left behind


def test_an_http_error_names_the_document(tmp_path):
    with pytest.raises(PackError, match="demo-annual-report-2025: HTTP 404"):
        fetch(serving(b"", status=404), DOCUMENT, tmp_path / "doc.pdf")


def test_pages_are_read_from_the_pdfs_text(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(text_pdf([["Demo Laks AS harvested salmon."], ["Page two."]]))
    pages = read_pdf(path)
    assert [(p.number, p.text, p.read_by) for p in pages] == [
        (1, "Demo Laks AS harvested salmon.", "text"),
        (2, "Page two.", "text"),
    ]


def test_running_headers_go_and_lines_join_into_paragraphs():
    header = "DEMO LAKS AS ANNUAL REPORT 2025"
    pages = [
        Page(
            n,
            f"{header}\n{name} had {n}0 em-\nployees in {name}by at the\nend of 202{n}, {name} "
            f"says.\nPage {n}",
        )
        for n, name in enumerate(["Laks", "Fisk", "Data", "Bygg"], start=1)
    ]
    # The header, and the footer with its page number, are on every page; the rest is joined
    assert clean(pages)[0].text == "Laks had 10 employees in Laksby at the end of 2021, Laks says."


def test_a_row_of_a_table_keeps_its_own_line():
    page = Page(1, "Income statement\nRevenue 5 123.4 4 987.2\nCosts (3 210.0) (3 001.5)\nNotes")
    assert clean([page])[0].text == page.text


@pytest.mark.parametrize(
    "line, row",
    [
        ("Salgsinntekter 4.040.374 3.540.283", True),
        ("Resultat av finansposter -71.913 -39.979", True),
        ("Margin 12.1% 10.4%", True),
        ("REVENUE AND OTHER INCOME 5 720.2 5 603.8 116.4 100 % 100 %", True),
        ("Net fair value adjustment biomass -14.2 40.6 -54.8 — % 1 %", True),
        ("In 2025 revenue rose by 12% to", False),
        ("Revenue", False),
    ],
)
def test_what_counts_as_a_row_of_a_table(line, row):
    assert table_row(line) is row


def test_a_page_is_cut_into_passages_of_whole_paragraphs():
    page = Page(1, "\n".join(["one two three"] * 4))
    assert passages(page, max_words=7) == ["one two three\none two three"] * 2


def test_a_long_paragraph_is_cut_after_a_sentence():
    page = Page(1, "First sentence ends here. " + "word " * 10)
    assert passages(page, max_words=6)[0] == "First sentence ends here."


def test_a_long_table_is_cut_between_rows_and_keeps_its_headings():
    rows = [f"Line {n} {n}.0 {n}.5" for n in range(1, 7)]
    page = Page(1, "\n".join(["EUR MILLION", "2025 2024", *rows]))
    first, second = passages(page, max_words=15)[:2]
    assert first.splitlines()[:3] == ["EUR MILLION", "2025 2024", "Line 1 1.0 1.5"]
    assert second.splitlines()[:2] == ["EUR MILLION", "2025 2024"]


def test_numbers_are_compared_by_their_digits():
    assert digit_runs("1 480 312 and 4.040.374, (1 234) 197,3 then 2025 2024") == [
        "1480312", "4040374", "1234", "1973", "2025", "2024",
    ]  # fmt: skip
    assert unconfirmed("Salgsinntekter | 4.040.374 | 50.000", "4 040 374\n57.889") == ["50000"]


def test_ocr_that_runs_a_rows_numbers_together_still_confirms_each():
    read = "Avskrivning | 2 | 182 500 | 177 100"
    assert unconfirmed(read, "Avskrivning pa driftsmidler 2 182 500 177 100") == []
    assert unconfirmed("Varer | 1 868 507", "Varekostnad 1 868 570") == ["1868507"]


def ocr_reading(text):
    def ocr(picture):
        assert picture.startswith(b"\x89PNG")
        return text

    return ocr


def test_a_scan_is_read_by_the_model_and_checked_by_ocr():
    model = FakeModel("Salgsinntekter | 4.040.374")
    text, missing, answers = read_scan(
        model, b"\x89PNG...", ocr_reading("Salgsinntekter 4.040.374")
    )
    assert (text, missing, len(answers)) == ("Salgsinntekter | 4.040.374", (), 1)
    assert model.requests[0].messages[0].images[0].data == b"\x89PNG..."


def test_a_number_ocr_did_not_see_is_sent_back_once_then_reported():
    model = FakeModel("Utbytte | 50.000", "Utbytte | 50.000")
    text, missing, answers = read_scan(model, b"\x89PNG...", ocr_reading("Utbytte"))
    assert missing == ("50000",)
    assert len(answers) == 2
    assert "50000" in model.requests[1].messages[-1].text


def test_a_scanned_pdf_is_read_page_by_page(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf(pages=2))
    model = FakeModel("Side 1", "Side 2")
    reading = read_document(path, model, ocr_reading("Side 1 Side 2"))
    assert [(p.number, p.text, p.read_by) for p in reading.pages] == [
        (1, "Side 1", "model"),
        (2, "Side 2", "model"),
    ]
    assert len(reading.answers) == 2


def test_without_a_model_a_scanned_page_is_left_unread(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())
    reading = read_document(path, None, ocr_reading(""))
    assert [(p.text, p.read_by) for p in reading.pages] == [("", "unread")]
