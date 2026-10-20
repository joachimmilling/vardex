"""Documents: a pack's reports, read page by page and cut into passages to search.

A pack lists its documents in documents/sources.yaml: where each one is, what it is about and
what it is. Each is fetched once, and read page by page: from the PDF's own text where a page
has one, and by a model looking at a picture of the page where it has none, as on a scan. A
passage never crosses a page, so every passage can be cited by its page.
"""

import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Literal

import httpx2
import pypdfium2 as pdfium
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vardex.ingest import get
from vardex.llm import Answer, Image, Message, ModelClient, Request
from vardex.packs import Language, PackError, read_yaml
from vardex.prompts import READ_PAGE, READ_PAGE_REPAIR

SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"
LONG_SIDE = 1568  # pixels along a drawn page's longer side: small print stays readable, and
# a model sees the picture whole instead of shrinking it (about 1,600 tokens on an A4 page)
MAX_WORDS = 250  # words per passage: about 400 tokens, inside the 512 an embedding model reads
REPEATED = 0.3  # a line on more than this share of pages is a running header or footer
Ocr = Callable[[bytes], str]  # reads the text of a picture, without a model


class DocumentSpec(BaseModel):
    """One document in sources.yaml: where it is, and what it is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SLUG)
    title: str
    about: str  # the key of the object it is about, such as an organisation number
    kind: str = Field(pattern=SLUG)  # such as annual-report or quarterly-report
    period: str  # the period it covers, such as 2025 or 2026-Q2
    language: Language
    url: str


class DocumentList(BaseModel):
    """The contents of documents/sources.yaml in a pack."""

    model_config = ConfigDict(extra="forbid")

    about: str  # the ontology's object that every document is about, such as Company
    documents: list[DocumentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "DocumentList":
        ids = [document.id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("every document needs its own id")
        return self


@dataclass(frozen=True)
class Page:
    """The text of one page, and how it was read."""

    number: int  # the first page is 1, as a PDF viewer counts
    text: str
    # The PDF's text; a model reading a picture of the page; or a picture nobody read
    read_by: Literal["text", "model", "unread"] = "text"
    unconfirmed: tuple[str, ...] = ()  # numbers the model read that a second reader did not


@dataclass(frozen=True)
class Reading:
    """A whole document, read, and the model calls the scanned pages took."""

    pages: list[Page]
    answers: list[Answer]


def load_documents(pack: Path) -> DocumentList:
    """Read and validate documents/sources.yaml in a pack folder."""
    path = pack / "documents" / "sources.yaml"
    if not path.is_file():
        raise PackError(f"No documents/sources.yaml in {pack}")
    return read_yaml(path, DocumentList, "document list")


def document_path(data_dir: Path, pack_name: str, document: DocumentSpec) -> Path:
    """Where a pack's document is kept once fetched."""
    return data_dir / pack_name / "documents" / f"{document.id}.pdf"


def fetch(client: httpx2.Client, document: DocumentSpec, path: Path) -> bool:
    """Download a document unless it is already there. True when it was downloaded. A
    report does not change once published, so it is fetched once."""
    if path.is_file():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")  # renamed only when complete, so no half file is read
    try:
        response = get(client, document.url, into=part)
    except httpx2.TransportError as err:
        raise PackError(f"{document.id}: could not reach {document.url}: {err}") from err
    if response.status_code != 200:
        part.unlink(missing_ok=True)
        raise PackError(f"{document.id}: HTTP {response.status_code} from {document.url}")
    with part.open("rb") as file:
        if file.read(5) != b"%PDF-":  # every PDF starts so; a web page instead does not
            part.unlink()
            raise PackError(f"{document.id}: {document.url} did not return a PDF")
    part.rename(path)
    return True


def fingerprint(path: Path) -> str:
    """A fingerprint of a file's bytes: a changed file has another."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def draw(page: pdfium.PdfPage) -> bytes:
    """A page as a PNG picture, as it would print, LONG_SIDE pixels along its longer side."""
    width, height = page.get_size()  # in points: 1/72 of an inch
    buffer = BytesIO()
    page.render(scale=LONG_SIDE / max(width, height)).to_pil().save(buffer, format="PNG")
    return buffer.getvalue()


def has_picture(page: pdfium.PdfPage) -> bool:
    """True when a page holds an image, such as the photograph of a scanned page."""
    return any(True for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE], max_depth=2))


def read_pdf(
    path: Path,
    read_picture: Callable[[bytes], Page] | None = None,
    only: set[int] | None = None,
) -> list[Page]:
    """Every page of a PDF, or only the pages numbered in `only`, cleaned. A page with a
    picture and no text at all, as a scan has, is drawn and passed to `read_picture`, when
    there is one. A page of a digital report has some text even when it is mostly a
    photograph: a heading, a caption, a page number."""
    pdf = pdfium.PdfDocument(path)
    pages = []
    try:
        for number, page in enumerate(pdf, start=1):
            if only is not None and number not in only:
                continue
            text = page.get_textpage().get_text_range()
            scanned = not text.strip() and has_picture(page)
            if scanned and read_picture is not None:
                pages.append(replace(read_picture(draw(page)), number=number))
            else:
                pages.append(Page(number, text, "unread" if scanned else "text"))
            page.close()
    finally:
        pdf.close()
    return clean(pages)


def read_document(
    path: Path, client: ModelClient | None, ocr: Ocr, only: set[int] | None = None
) -> Reading:
    """Every page of a document, or only those in `only`. Without a model client, a scanned
    page is left unread."""
    answers: list[Answer] = []

    def read_picture(picture: bytes) -> Page:
        text, missing, used = read_scan(client, picture, ocr)  # type: ignore[arg-type]
        answers.extend(used)
        return Page(0, text, "model", missing)

    return Reading(read_pdf(path, read_picture if client else None, only), answers)


def page_numbers(text: str) -> set[int]:
    """Pages as a person writes them: 4, 4-6 or 2,4-6."""
    numbers: set[int] = set()
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        if not first.isdigit() or (last and not last.isdigit()):
            raise ValueError(f"{text!r} is not a page or a range of pages, such as 4 or 4-6")
        numbers.update(range(int(first), int(last or first) + 1))
    return numbers


def pages_text(pages: list[Page]) -> str:
    """Pages as one text for a model, each marked with its number."""
    return "\n".join(f'<page number="{p.number}">\n{p.text}\n</page>' for p in pages)


def find_page(pages: list[Page], words: str) -> int | None:
    """The first page that holds these words, whatever the spacing, or None."""
    wanted = " ".join(words.split())
    return next((p.number for p in pages if wanted in " ".join(p.text.split())), None)


def clean(pages: list[Page]) -> list[Page]:
    """Pages as paragraphs: running headers and footers taken out, words split at a line end
    joined, and the lines of a paragraph joined into one line. A line that looks like a row of
    a table keeps its own line."""
    counts = Counter(
        running(line) for page in pages for line in {running(x) for x in page.text.splitlines()}
    )
    counts.pop("", None)
    repeated = {key for key, n in counts.items() if n > REPEATED * len(pages) and n > 2}
    return [replace(page, text=paragraphs(page.text, repeated)) for page in pages]


def running(line: str) -> str:
    """A line as running headers and footers are compared: with every number made #, so a
    header that ends in its page number is the same on every page. A row of a table is
    compared as it is: the same labels with other numbers are not a header."""
    line = line.strip()
    return line if table_row(line) else re.sub(r"\d+", "#", line)


NUMERIC = re.compile(r"^\(?[-−–]?[\d.,]*\d[\d.,]*\)?%?$")  # 5.2, (1 234), -71.913, 12%
STANDALONE = {"%", "—", "–", "-", "|"}  # a sign alone, a dash, a cell edge
SENTENCE_END = re.compile(r"[.!?:;]\s*$")


def paragraphs(text: str, repeated: set[str]) -> str:
    """One page's text with each paragraph on one line, and each row of a table."""
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    out: list[str] = []
    for line in lines:
        if not line or running(line) in repeated:
            if out and out[-1]:
                out.append("")  # an empty line ends a paragraph
            continue
        if out and out[-1] and not table_row(line) and not table_row(out[-1]):
            if out[-1].endswith("-") and line[:1].islower():
                out[-1] = out[-1][:-1] + line  # a word split at the end of the line
                continue
            if not SENTENCE_END.search(out[-1]):
                out[-1] += " " + line
                continue
        out.append(line)
    return "\n".join(out).strip()


def table_row(line: str) -> bool:
    """True for a line that looks like a row of a table: it ends in two numbers, not counting
    signs that stand alone, as in `Revenue 5 720.2 5 603.8 100 % 100 %`. A sentence that ends
    in a year, such as `beyond 650,000 GWT by 2029`, does not."""
    tokens = [t for t in line.split() if t not in STANDALONE]
    return len(tokens) >= 2 and all(NUMERIC.match(t) for t in tokens[-2:])


def passages(page: Page, max_words: int = MAX_WORDS) -> list[str]:
    """A page cut into passages of at most `max_words`, each made of whole lines: paragraphs
    and rows of tables. A passage that starts inside a table repeats the table's headings, so
    every part of a table still says what its columns are. A paragraph longer than a passage
    is cut between sentences."""
    lines = [line for line in page.text.split("\n") if line.strip()]
    found: list[list[str]] = []
    heading: list[str] = []  # the lines above the table we are in, if we are in one
    for n, line in enumerate(lines):
        if not table_row(line):
            heading = []
        elif not heading and line[:1].isalpha():  # the first row of data: a label, then numbers
            heading = headings(lines[:n])
        parts = (
            split_sentences(line.split(), max_words) if len(line.split()) > max_words else [line]
        )
        for part in parts:
            if found and words(found[-1]) + len(part.split()) <= max_words:
                found[-1].append(part)
            else:
                found.append([*heading, part] if heading and part not in heading else [part])
    return ["\n".join(passage) for passage in found]


def headings(above: list[str]) -> list[str]:
    """The lines just above a table's first row that name its columns: up to three short ones,
    such as `EUR MILLION` and `2025 2024`."""
    found: list[str] = []
    for line in reversed(above[-3:]):
        if len(line.split()) > 15 or (table_row(line) and line[:1].isalpha()):
            break
        found.insert(0, line)
    return found


def words(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def split_sentences(words: list[str], max_words: int) -> list[str]:
    """A run of words cut after the last sentence that ends in each `max_words`, or at
    `max_words` when none ends in the second half."""
    parts = []
    while len(words) > max_words:
        cut = next(
            (i for i in range(max_words, max_words // 2, -1) if words[i - 1][-1:] in ".!?"),
            max_words,
        )
        parts.append(" ".join(words[:cut]))
        words = words[cut:]
    return [*parts, " ".join(words)] if words else parts


def tesseract(picture: bytes) -> str:
    """The text in a picture, read by the Tesseract OCR program, in Norwegian and English.
    Page segmentation mode 4 reads a page as one column of lines, so a row of a table stays on
    one line instead of being read column by column."""
    try:
        done = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "nor+eng", "--psm", "4"],
            input=picture,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as err:
        raise PackError("Reading a scanned page needs Tesseract: brew install tesseract") from err
    except subprocess.CalledProcessError as err:
        raise PackError(f"Tesseract failed: {err.stderr.decode(errors='replace')}") from err
    return done.stdout.decode("utf-8", errors="replace")


DIGIT_RUN = re.compile(r"\d+")
# What may stand between the digits of one number: a point, a comma or an apostrophe; or a
# space before a group of exactly three, as in 1 480 312, but not between two numbers in a row
JOINED = re.compile(r"(?<=\d)(?:[.,']|[ \u00a0\u202f](?=\d{3}(?!\d)))(?=\d)")


def digit_runs(text: str) -> list[str]:
    """Each number in a text as its digits alone: 1 480 312, 1,480,312 and 1480312 all
    give 1480312, and 197.3 gives 1973."""
    return DIGIT_RUN.findall(JOINED.sub("", text))


SEQUENCE = re.compile(r"\d+(?:[ \u00a0\u202f.,']\d+)*")  # digit groups, one mark apart
SPACE = " \u00a0\u202f"


def possible_numbers(text: str) -> set[str]:
    """Every number the digits in an OCR program's text may stand for. OCR keeps the digits
    but loses the gaps between a table's columns, so `2 182 500 177 100` may be 2, 182 500
    and 177 100, or 2 182 500 and 177 100: every reading is kept."""
    found = set()
    for match in SEQUENCE.finditer(text):
        parts = re.split(f"([{SPACE}.,'])", match.group())
        groups, marks = parts[0::2], parts[1::2]
        for i in range(len(groups)):
            digits = groups[i]
            found.add(digits)
            for j in range(i + 1, len(groups)):
                if marks[j - 1] in SPACE and len(groups[j]) != 3:
                    break  # a space can only stand before a group of three
                digits += groups[j]
                found.add(digits)
    return found


def unconfirmed(read: str, ocr: str) -> list[str]:
    """The numbers a model read that the OCR program did not read anywhere on the page."""
    found = possible_numbers(ocr)
    return list(dict.fromkeys(n for n in digit_runs(read) if n not in found))


def read_scan(
    client: ModelClient, picture: bytes, ocr: Ocr, *, repairs: int = 1
) -> tuple[str, tuple[str, ...], list[Answer]]:
    """Have a model write out a picture of a page, and check every number it wrote against
    what an OCR program reads on the same page. Two readers that fail in different ways seldom
    make the same mistake. A number only the model saw is shown to it, and it reads again; a
    number it still sees is kept, and reported as unconfirmed."""
    request = Request(
        system=READ_PAGE,
        messages=[Message("user", "Write out this page.", images=(Image(picture),))],
        max_tokens=8000,
        effort="low",
    )
    ocr_text = ocr(picture)
    answers: list[Answer] = []
    missing: list[str] = []
    for _ in range(repairs + 1):
        answer = client.send(request)
        answers.append(answer)
        missing = unconfirmed(answer.text, ocr_text)
        if not missing:
            break
        repair = READ_PAGE_REPAIR.format(numbers=", ".join(missing))
        request = replace(
            request,
            messages=[
                *request.messages,
                Message("assistant", answer.text),
                Message("user", repair),
            ],
        )
    return answers[-1].text, tuple(missing), answers
