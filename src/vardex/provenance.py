"""Provenance: where each number in an answer came from.

A Cell is one value in a query's result: a row and a column of the rows a warehouse returned.
A Figure is one number written in an answer, with every place it can be traced to: cells, the
SQL that ran, or the question itself. A number that traces to none of them came from the
model, not the data, and an answer that holds one is not shown.
"""

import re
from dataclasses import dataclass
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
}  # fmt: skip
SUFFIX = re.compile(
    r"\s?(" + "|".join(re.escape(s) for s in sorted(SCALES, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)  # "1. " in a numbered list
SPACES = re.compile(r"[ \u00a0\u202f]")


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
class Figure:
    """A number written in an answer, and everywhere it can be traced to."""

    text: str  # as written, such as 5.2%, 2,131 or 2.1 billion
    cells: list[Cell]
    in_sql: bool  # the query itself holds it, such as the year in where fiscal_year = 2025
    in_question: bool

    @property
    def traced(self) -> bool:
        return bool(self.cells) or self.in_sql or self.in_question


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


def trace(text: str, results: list[Result], question: str) -> list[Figure]:
    """Every number in an answer, with the cells, query and question it can be traced to."""
    in_question = exact_values(question)
    in_sql = {v for result in results for v in exact_values(result.sql)}
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
                in_sql=bool(exact & in_sql),
                in_question=bool(exact & in_question),
            )
        )
    return figures
