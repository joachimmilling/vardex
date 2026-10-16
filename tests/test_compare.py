"""Comparing the rows a query returned with the reference rows."""

from decimal import Decimal

import pytest

from vardex.evals import compare, normalise
from vardex.sql import Result


def result(columns: list[str], rows: list[tuple]) -> Result:
    return Result("select …", columns, rows, False, "test", 0.0)


def test_more_columns_with_other_names_still_match():
    reference = result(["name"], [("DEMO LAKS AS",), ("DEMO DATA AS",)])
    got = result(
        ["orgnr", "company_name", "margin"],
        [("100000008", "DEMO LAKS AS", 0.236), ("100000016", "DEMO DATA AS", 0.05)],
    )
    assert compare(reference, got, ordered=True) is None


def test_numbers_match_to_six_significant_digits_and_whole_numbers_exactly():
    assert normalise(Decimal("0.15619915392124958")) == normalise(0.156199153921)
    assert normalise(Decimal("1177860")) == "1177860" != normalise(1177861.0)
    assert normalise(" Vestland ") == "Vestland"
    reference = result(["n"], [(Decimal("49019"),)])
    assert compare(reference, result(["count"], [(49019,)]), ordered=False) is None


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([("DEMO LAKS AS",)], "1 rows, expected at least 2"),
        ([("DEMO DATA AS",), ("DEMO LAKS AS",)], "the right rows, in the wrong order"),
        ([("DEMO LAKS AS",), ("DEMO FISK AS",)], "no column holds the values of name"),
    ],
)
def test_wrong_rows_say_what_is_wrong(rows, reason):
    reference = result(["name"], [("DEMO LAKS AS",), ("DEMO DATA AS",)])
    assert compare(reference, result(["name"], rows), ordered=True) == reason


def test_more_rows_are_fine_when_the_reference_rows_come_first():
    reference = result(["name"], [("EQUINOR ASA",)])
    context = result(["name"], [("EQUINOR ASA",), ("Norsk Hydro ASA",), ("YARA ASA",)])
    assert compare(reference, context, ordered=True) is None
    second = result(["name"], [("Norsk Hydro ASA",), ("EQUINOR ASA",)])
    assert compare(reference, second, ordered=True) == "the right rows, in the wrong order"
    assert compare(reference, second, ordered=False) is None


def test_an_empty_reference_leaves_the_words_to_the_judge():
    reference = result(["revenue"], [])
    assert compare(reference, result(["name"], [("DNB EIENDOM AS",)]), ordered=False) is None


def test_the_order_matters_only_when_the_case_says_so():
    reference = result(["name"], [("DEMO LAKS AS",), ("DEMO DATA AS",)])
    reverse = result(["name"], [("DEMO DATA AS",), ("DEMO LAKS AS",)])
    assert compare(reference, reverse, ordered=False) is None


def test_values_must_sit_in_the_same_rows():
    reference = result(["name", "n"], [("A", 1), ("B", 2)])
    swapped = result(["name", "n"], [("A", 2), ("B", 1)])
    assert compare(reference, swapped, ordered=False) == (
        "the right values, but not in the same rows"
    )
