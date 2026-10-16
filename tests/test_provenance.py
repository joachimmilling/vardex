from datetime import date
from decimal import Decimal

import pytest

from vardex.provenance import misspelt, readings, trace
from vardex.sql import Result

RESULT = Result(
    sql="select name, orgnr, revenue, margin, founded from c where fiscal_year = 2025 limit 10",
    columns=["name", "orgnr", "revenue", "margin", "founded"],
    rows=[
        ("DEMO LAKS AS", "100000008", Decimal("2131417000"), 0.0523, date(2001, 5, 2)),
        ("DEMO HOLDING AS", "100000016", Decimal("441953000"), -0.9848, None),
    ],
    truncated=False,
    warehouse="DuckDB test",
    seconds=0.01,
)


def where(text: str, question: str = "Which companies?") -> list[tuple]:
    """Each number in the text, and the first cell it traces to, or where else it was found."""
    found = []
    for figure in trace(text, [RESULT], question):
        if figure.cells:
            found.append((figure.text, figure.cells[0].row, figure.cells[0].column))
        else:
            found.append((figure.text, figure.in_sql, figure.in_question))
    return found


def test_readings_in_english_and_norwegian():
    assert readings("2,131") == {(Decimal("2131"), 0), (Decimal("2.131"), 3)}
    assert readings("5,2") == {(Decimal("5.2"), 1)}
    assert readings("2 131 417 000") == {(Decimal("2131417000"), 0)}
    assert readings("0.985") == {(Decimal("0.985"), 3)}


@pytest.mark.parametrize(
    ("text", "row", "column"),
    [
        ("5.2%", 1, "margin"),
        ("5,2 prosent", 1, "margin"),
        ("−98.5%", 2, "margin"),  # the sign is in the words, such as "a loss of"
        ("NOK 2.1 billion", 1, "revenue"),
        ("2 131 millioner", 1, "revenue"),
        ("2,131,417,000", 1, "revenue"),
        ("NOK 442 mill.", 2, "revenue"),
        ("100 000 008", 1, "orgnr"),
        ("2001", 1, "founded"),
    ],
)
def test_a_number_traces_to_its_cell_exactly_or_rounded(text, row, column):
    [(_, found_row, found_column)] = where(f"The answer is {text}.")
    assert (found_row, found_column) == (row, column)


@pytest.mark.parametrize("written", ["NOK 1,1 billioner", "NOK 1,106 billioner", "1.1 trillion"])
def test_a_norwegian_billion_is_a_million_million(written):
    equinor = Result("select …", ["revenue_nok"], [(Decimal("1106267934400"),)], False, "", 0.0)
    [figure] = trace(f"Equinor ligger øverst med {written}.", [equinor], "Hvem er størst?")
    assert figure.cells and figure.text == written.removeprefix("NOK ")


def test_a_number_from_the_assumptions_is_traced_to_them():
    assumptions = ["The accounts cover companies with at least 50 employees."]
    text = "Demo Laks AS, among companies with at least 50 employees."
    [figure] = trace(text, [RESULT], "Which company?", assumptions)
    assert (figure.text, figure.cells, figure.in_assumptions) == ("50", [], True)
    assert not trace(text, [RESULT], "Which company?")[0].traced


def test_a_scaled_number_in_the_question_is_traced_to_it():
    question = "Which groups had more than NOK 50 billion in revenue, and a margin above 7%?"
    found = where("Two groups had more than NOK 50 billion and a margin above 7%.", question)
    assert found == [("50 billion", False, True), ("7%", False, True)]


def test_numbers_in_the_query_and_the_question_are_traced_too():
    assert where("In 2025, more than 200 had it.", "Who had more than 200 employees?") == [
        ("2025", True, False),
        ("200", False, True),
    ]


def test_a_number_from_nowhere_is_not_traced():
    [figure] = trace("The average was 7.3%.", [RESULT], "What was the average?")
    assert not figure.traced


def test_rounding_is_allowed_only_to_the_digits_shown():
    [figure] = trace("NOK 2.13 billion", [RESULT], "")
    assert figure.traced
    [figure] = trace("NOK 2.14 billion", [RESULT], "")
    assert not figure.traced


def test_list_markers_codes_and_words_hold_no_numbers():
    assert where("1. Demo Laks AS, SN2007 code, Q3") == []


NAMES = ["LERØY SEAFOOD GROUP ASA", "MOWI ASA", "SJØTROLL HAVBRUK AS"]


def test_a_name_spelt_unlike_the_rows_is_found():
    text = "Leròy Seafood Group ASA was last. Sjøtrol Havbruk AS was third."
    assert misspelt(text, NAMES) == [
        ("Leròy Seafood Group ASA", "LERØY SEAFOOD GROUP ASA"),
        ("Sjøtrol Havbruk AS", "SJØTROLL HAVBRUK AS"),
    ]


def test_capitals_short_names_and_other_words_are_fine():
    text = "Mowi led. Lerøy Seafood Group was last. In Vestland, in NOK, LERØY SEAFOOD GROUP ASA."
    assert misspelt(text, NAMES) == []
