"""Extraction measured with the same runner: the pack's key-figures suite and a fake model."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from fakes import FakeModel
from pydantic import ValidationError
from test_extraction import GOOD
from typer.testing import CliRunner

from vardex.cli import app
from vardex.evals import (
    ExtractionCase,
    ExtractionSuite,
    check_extraction_suite,
    extract_case,
    grade_behaviour,
    grade_fields,
    load_suites,
    run,
    same,
)
from vardex.extraction import load_extraction

PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"
SPEC = load_extraction(PACK, "key-figures")
SUITE = load_suites(PACK)["key-figures"]
GRADERS = [grade_behaviour, grade_fields]


def case(case_id: str) -> ExtractionCase:
    return next(c for c in SUITE.cases if c.id == case_id)


def test_the_pack_has_an_extraction_suite_fit_to_measure_with():
    assert isinstance(SUITE, ExtractionSuite)
    assert len(SUITE.cases) == 10
    assert check_extraction_suite(SUITE, SPEC, PACK) == []


def test_numbers_are_compared_as_exact_decimals():
    assert same(Decimal("842.6"), 842.6)
    assert not same(Decimal("842.6"), 842.7)
    assert same(None, None) and same("NOK", "NOK")


def test_right_values_pass():
    result = extract_case(FakeModel(json.dumps(GOOD)), PACK, SPEC, case("laks-thousands"), GRADERS)
    assert result.passed
    assert result.outcome.values["revenue"] == Decimal("1480312")


def test_each_wrong_field_is_named():
    wrong = GOOD | {"accounts": "group", "unit": "ones", "unit_quote": "Beløp i NOK 1 000"}
    result = extract_case(FakeModel(json.dumps(wrong)), PACK, SPEC, case("laks-thousands"), GRADERS)
    fields = result.grades[1]
    assert (fields.grader, fields.passed) == ("fields", False)
    assert fields.reason == (
        "accounts is 'group', expected 'company'; unit is 'ones', expected 'thousands'"
    )


def test_a_page_without_figures_must_be_refused():
    problem = json.dumps({key: None for key in GOOD} | {"problem": "No figures on this page."})
    passed = extract_case(FakeModel(problem), PACK, SPEC, case("letter-without-figures"), GRADERS)
    assert passed.passed and passed.outcome.kind == "problem"
    good = json.dumps(GOOD)  # Demo Laks AS's figures: not on this page, so twice wrong
    answered = extract_case(
        FakeModel(good, good), PACK, SPEC, case("letter-without-figures"), GRADERS
    )
    assert answered.grades[0].reason.startswith("expected problem, got no answer")


def test_values_that_fail_their_checks_are_no_answer():
    made_up = json.dumps(GOOD | {"revenue_quote": "1 999 999"})
    result = extract_case(FakeModel(made_up, made_up), PACK, SPEC, case("laks-thousands"), GRADERS)
    assert result.outcome.kind == "error"
    assert [g.grader for g in result.grades] == ["behaviour"]


def test_the_suite_runs_and_reports_like_any_other():
    model = FakeModel(json.dumps(GOOD))
    report = run(
        "key-figures",
        SUITE,
        "m",
        lambda c: extract_case(model, PACK, SPEC, c, GRADERS),
        only=["laks-thousands"],
    )
    assert (report.total.cases, report.total.passed) == (1, 1)


def test_a_case_expecting_values_needs_them_and_a_known_field():
    with pytest.raises(ValidationError, match="needs values"):
        ExtractionCase.model_validate({"id": "x", "type": "y", "document": "01.txt"})
    odd = SUITE.model_copy(
        update={"cases": [case("laks-thousands").model_copy(update={"values": {"ebitda": 1}})]}
    )
    assert check_extraction_suite(odd, SPEC, PACK) == [
        "case laks-thousands: the extraction has no field ebitda"
    ]


def test_vardex_eval_runs_an_extraction_suite_without_a_warehouse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VARDEX_DATA_DIR", str(tmp_path))  # no warehouse here
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel(json.dumps(GOOD)))
    result = CliRunner().invoke(app, ["eval", str(PACK), "key-figures", "--case", "laks-thousands"])
    assert result.exit_code == 0, result.output
    assert "key-figures: 1 cases on claude-sonnet-5-5" in result.output
