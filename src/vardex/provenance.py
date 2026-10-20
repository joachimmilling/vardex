"""Provenance: where each number in an answer came from.

A Cell is one value in a query's result: a row and a column of the rows a warehouse returned.
A Quote is words copied from one page of a document. A Figure is one number written in an
answer, with every place it can be traced to: cells or quotes, the SQL that ran, the question
itself, or the assumptions shown beside the answer. A number that traces to none of them came
from the model, not the data, and an answer that holds one is not shown.
"""

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from vardex.sql import Result, is_number

# A number as people write it: 2131417000, 2,131,417,000, 2 131 417 000, 5.2 or 5,2. It may
# not start inside a word or another number, so SN2007 and Q3 hold no number.
NUMBER = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*)(?![\w])"
)
# What may follow a number and change what it means: a percent sign, or a scale word.
SCALES = {
    "%": None, "percent": None, "per cent": None, "prosent": None,
    "thousand": 3, "tusen": 3, "k": 3,
    "million": 6, "millions": 6, "millioner": 6, "mill.": 6, "mill": 6, "mn": 6, "m": 6,
    "billion": 9, "billions": 9, "milliard": 9, "milliarder": 9, "mrd.": 9, "mrd": 9, "bn": 9,
    "trillion": 12, "billioner": 12,  # a Norwegian billion is a million million
}  # fmt: skip
SUFFIX = re.compile(
    r"\s?(" + "|".join(re.escape(s) for s in sorted(SCALES, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "1. " in a numbered list
SPACES = re.compile(r"[ \u00a0\u202f]")
# A run of capitalised words, such as Lerøy Seafood Group ASA: what a name looks like in text
NAME = re.compile(r"[^\W\d_a-zà-ÿ][\w'&-]*(?:[ \u00a0][^\W\d_a-zà-ÿ][\w'&-]*)*")
SIMILAR = 0.85  # how alike a written name and a name in the rows must be to count as a slip


@dataclass(frozen=True)
class Cell:
    """One value in a query's result. Row 1 is the first row."""

    result: Result
    row: int
    column: str

    @property
    def value(self) -> Any:
        return self.result.rows[self.row - 1][self.result.columns.index(self.column)]


@dataclass(frozen=True)
class Quote:
    """Words copied from one page of a document, and where a person can read them."""

    document: str  # the document's id
    title: str
    page: int  # the first page is 1, as a PDF viewer counts
    text: str
    link: str  # the document's address, opened at the page


@dataclass(frozen=True)
class Figure:
    """A number written in an answer, and everywhere it can be traced to."""

    text: str  # as written, such as 5.2%, 2,131 or 2.1 billion
    cells: list[Cell]
    in_sql: bool  # the query itself holds it, such as the year in where fiscal_year = 2025
    in_question: bool
    in_assumptions: bool = False  # such as "at least 50 employees", from the pack's notes
    quotes: list[Quote] = field(default_factory=list)  # quotes from documents that hold it

    @property
    def traced(self) -> bool:
        return (
            bool(self.cells or self.quotes)
            or self.in_sql
            or self.in_question
            or self.in_assumptions
        )


def readings(token: str) -> set[tuple[Decimal, int]]:
    """Every value a written number can have, with the decimals it shows. 2,131 is 2131 in
    English and 2.131 in Norwegian; 5,2 can only be 5.2."""
    token = SPACES.sub("", token)
    found = set()
    for thousands, decimal in ((",", "."), (".", ",")):
        whole, _, fraction = token.partition(decimal)
        groups = whole.split(thousands)
        if decimal in fraction or thousands in fraction:
            continue
        if len(groups) > 1 and (groups[0].startswith("0") or any(len(g) != 3 for g in groups[1:])):
            continue
        try:
            value = Decimal("".join(groups) + ("." + fraction if fraction else ""))
        except InvalidOperation:
            continue
        found.add((value, len(fraction)))
    return found


def numbers_in(text: str) -> list[tuple[str, set[tuple[Decimal, int]], str | None]]:
    """Each number in a text: as written with what follows it, its readings, and its suffix."""
    text = LIST_MARKER.sub("", text)
    found = []
    for match in NUMBER.finditer(text):
        suffix = SUFFIX.match(text, match.end())
        word = suffix.group(1).lower() if suffix else None
        written = match.group() + (suffix.group() if suffix else "")
        found.append((written, readings(match.group()), word))
    return found


def exact_values(text: str) -> set[Decimal]:
    """Every value any number in a text can have, read exactly, ignoring what follows it."""
    return {value for _, values, _ in numbers_in(text) for value, _ in values}


def atoms(value: Any) -> list[tuple[Decimal, bool]]:
    """The numbers in one cell, and whether each may be rounded: a number itself, the numbers
    in a text such as an organisation number, and the year, month and day of a date."""
    if is_number(value):
        try:
            return [(abs(Decimal(str(value))), True)]
        except InvalidOperation:  # inf or nan
            return []
    if isinstance(value, date):
        return [(Decimal(part), False) for part in (value.year, value.month, value.day)]
    if isinstance(value, str):
        return [(v, False) for v in exact_values(value)]
    return []


def targets(
    values: set[tuple[Decimal, int]], word: str | None
) -> list[tuple[Decimal, Decimal, int]]:
    """What a written number may stand for, each with how far a value rounded to it can be:
    2.1 billion is 2,100,000,000 give or take 50,000,000; 5.2% is 0.052 give or take 0.0005,
    or 5.2 give or take 0.05 when the query already gave a percentage. The last value of each
    says how good a reading it is: 0 for the plain one, 1 for the second guess."""
    power = SCALES.get(word) if word else None
    percent = word is not None and power is None
    found = []
    for value, decimals in values:
        scale = Decimal(10) ** (power or 0)
        half = Decimal(5) * Decimal(10) ** (-decimals - 1) * scale  # half the last digit shown
        if percent:
            found.append((value / 100, half / 100, 0))
        found.append((value * scale, half, 1 if percent else 0))
    return found


def amounts(text: str) -> set[Decimal]:
    """Every value the numbers in a text can stand for: as written, and scaled by the word
    after them, so "NOK 50 billion" in a question holds both 50 and 50,000,000,000."""
    found = set()
    for _, values, word in numbers_in(text):
        power = SCALES.get(word) if word else None
        for value, _ in values:
            found.add(value)
            if power:
                found.add(value * Decimal(10) ** power)
    return found


def mentions(
    found: set[Decimal], exact: set[Decimal], scaled: list[tuple[Decimal, Decimal, int]]
) -> bool:
    """True when a text's numbers hold a written number: one of its exact values, or a value
    a scaled number such as 50 billion or 5% can be rounded from."""
    return bool(exact & found) or any(abs(v - t) <= d for v in found for t, d, _ in scaled)


def trace(
    text: str, results: list[Result], question: str, assumptions: list[str] | None = None
) -> list[Figure]:
    """Every number in an answer, with the cells, query, question and assumptions it can be
    traced to. The assumptions are shown to the person beside the answer, so a number taken
    from one, such as a coverage limit from the pack's notes, has a visible source."""
    in_question = amounts(question)
    in_assumptions = amounts("\n".join(assumptions or []))
    in_sql = {v for result in results for v in amounts(result.sql)}
    cells = [
        (Cell(result, row, column), atom, rounded)
        for result in results
        for row, values in enumerate(result.rows, start=1)
        for column, value in zip(result.columns, values, strict=True)
        for atom, rounded in atoms(value)
    ]
    figures = []
    for written, values, word in numbers_in(text):
        exact = {value for value, _ in values} if word is None else set()
        near = targets(values, word)
        scaled = near if word else []  # 50 billion or 5%: compare with what it rounds from
        found: dict[tuple[int, int, str], tuple[tuple[int, Decimal], Cell]] = {}
        for cell, atom, rounded in cells:
            if rounded:  # the best reading, then how far off in units of the rounding allowed
                fits = [(rank, abs(atom - t) / d) for t, d, rank in near if abs(atom - t) <= d]
                if not fits:
                    continue
                miss = min(fits)
            elif atom in exact:
                miss = (0, Decimal(0))
            else:
                continue
            key = (id(cell.result), cell.row, cell.column)
            if key not in found or miss < found[key][0]:
                found[key] = (miss, cell)
        figures.append(
            Figure(
                text=written.strip(),
                cells=[cell for _, cell in sorted(found.values(), key=lambda m: m[0])],
                in_sql=mentions(in_sql, exact, scaled),
                in_question=mentions(in_question, exact, scaled),
                in_assumptions=mentions(in_assumptions, exact, scaled),
            )
        )
    return figures


def trace_quotes(text: str, quotes: list[Quote], question: str) -> list[Figure]:
    """Every number in an answer from documents, with the quotes and question it can be traced
    to. A number in an answer must be in the words it cites, as written or rounded, or in the
    question: 197.3 million may be written 197 million, but never worked out from two others.
    Something that reads as no number at all, such as the date 18.04.18, must be copied as
    written."""
    in_question = amounts(question)
    in_quotes = [(quote, amounts(quote.text)) for quote in quotes]
    figures = []
    for written, values, word in numbers_in(text):
        token = written.strip()
        if values:
            exact = {value for value, _ in values} if word is None else set()
            scaled = targets(values, word) if word else []
            asked = mentions(in_question, exact, scaled)
            cited = [quote for quote, found in in_quotes if mentions(found, exact, scaled)]
        else:  # it reads as no number, such as a date: it must be copied as written
            asked = written_in(token, question)
            cited = [quote for quote, _ in in_quotes if written_in(token, quote.text)]
        figures.append(Figure(token, [], in_sql=False, in_question=asked, quotes=cited))
    return figures


def written_in(token: str, text: str) -> bool:
    """True when a text holds a token as written, not inside a longer word or number."""
    return re.search(rf"(?<![\w.,]){re.escape(token)}(?![\w])", text) is not None


def misspelt(text: str, names: list[str]) -> list[tuple[str, str]]:
    """Names in a text that are close to a name in the rows but not the same: each as written,
    and as the rows spell it. Capitals do not count, and a name may be shortened, so Mowi is
    fine for MOWI ASA; Leròy for LERØY is not. A name that is like none in the rows is not
    reported: it may be a place or a word, not a slip."""
    known = {name.casefold(): name for name in names if name}
    found = []
    for match in NAME.finditer(text):
        written = match.group()
        folded = written.casefold()
        if len(folded) < 4 or any(folded in name or name in folded for name in known):
            continue
        close = difflib.get_close_matches(folded, list(known), n=1, cutoff=SIMILAR)
        if close:
            found.append((written, known[close[0]]))
    return list(dict.fromkeys(found))
