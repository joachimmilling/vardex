import json
from decimal import Decimal
from pathlib import Path

import pytest
from fakes import FakeModel

from vardex.extraction import (
    ExtractionError,
    build_request,
    extract,
    load_extraction,
    output_model,
)
from vardex.packs import PackError

PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"
SPEC = load_extraction(PACK, "key-figures")
PAGE = """Demo Laks AS
RESULTATREGNSKAP
Beløp i NOK 1 000
                        2025         2024
Sum driftsinntekter     1 480 312    1 321 563
Driftsresultat          353 655      249 355
"""
GOOD = {
    "problem": None,
    "company_name": "Demo Laks AS",
    "fiscal_year": 2025,
    "accounts": "company",
    "currency": "NOK",
    "unit": "thousands",
    "unit_quote": "Beløp i NOK 1 000",
    "revenue": 1480312,
    "revenue_quote": "1 480 312",
    "operating_profit": 353655,
    "operating_profit_quote": "353 655",
}


def reply(**changes):
    return json.dumps(GOOD | changes)


def test_the_pack_describes_key_figures():
    assert {"unit", "revenue", "operating_profit"} <= set(SPEC.fields)
    assert SPEC.fields["revenue"].quote


def test_every_value_may_be_empty_and_quotes_are_added():
    schema = output_model(SPEC).model_json_schema()
    assert schema["required"][0] == "problem"
    assert "revenue_quote" in schema["properties"]
    assert {"type": "null"} in schema["properties"]["fiscal_year"]["anyOf"]


def test_good_values_pass_and_amounts_stay_exact():
    model = FakeModel(reply())

    result = extract(model, SPEC, PAGE)

    assert result.values["revenue"] == Decimal("1480312")
    assert result.values["unit"] == "thousands"
    request = model.requests[0]
    assert request.cache_system
    assert request.effort == "low"
    assert "Beløp i NOK 1 000" in request.system  # the pack's instructions are in the prompt
    assert PAGE in request.messages[0].text


def test_a_problem_reported_by_the_model_is_a_failure():
    with pytest.raises(ExtractionError, match="no figures on this page"):
        extract(FakeModel(reply(problem="There are no figures on this page.")), SPEC, PAGE)


def test_a_missing_required_value_is_a_failure():
    with pytest.raises(ExtractionError, match="operating_profit is missing"):
        extract(FakeModel(reply(operating_profit=None)), SPEC, PAGE, repairs=0)


def test_an_optional_value_may_be_missing():
    result = extract(FakeModel(reply(revenue=None, revenue_quote=None)), SPEC, PAGE)
    assert result.values["revenue"] is None


def test_a_quote_that_is_not_in_the_document_is_a_failure():
    with pytest.raises(ExtractionError, match="quote for unit is not in the document"):
        extract(FakeModel(reply(unit_quote="Amounts in NOK million")), SPEC, PAGE, repairs=0)


def test_an_amount_that_was_scaled_is_a_failure():
    # The classic trap: the page is in thousands, and the model "helpfully" gives kroner.
    with pytest.raises(ExtractionError, match="revenue is 1480312000"):
        extract(FakeModel(reply(revenue=1480312000)), SPEC, PAGE, repairs=0)


def test_an_answer_of_the_wrong_shape_is_a_failure():
    with pytest.raises(ExtractionError, match="unit: must be one of ones, thousands, millions"):
        extract(FakeModel(reply(unit="billions")), SPEC, PAGE, repairs=0)


def test_an_answer_cut_off_by_max_tokens_is_a_failure():
    with pytest.raises(ExtractionError, match="stopped early"):
        extract(FakeModel('{"problem": nu', stop_reason="max_tokens"), SPEC, PAGE)


def test_a_failed_check_is_repaired_once():
    model = FakeModel(reply(revenue=1480312000), reply())

    result = extract(model, SPEC, PAGE)

    assert result.values["revenue"] == Decimal("1480312")
    assert len(result.answers) == 2
    retry = model.requests[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "revenue is 1480312000" in retry[2].text  # the model is told what was wrong


def test_a_failure_after_the_repair_keeps_both_answers_for_the_log():
    model = FakeModel(reply(operating_profit=None), reply(operating_profit=None))
    with pytest.raises(ExtractionError, match="operating_profit is missing") as caught:
        extract(model, SPEC, PAGE)
    assert len(caught.value.answers) == 2
    assert caught.value.answers[0].cost_usd == Decimal("0.004")


def test_a_problem_reported_by_the_model_is_not_repaired():
    model = FakeModel(reply(problem="No figures."), reply())
    with pytest.raises(ExtractionError):
        extract(model, SPEC, PAGE)
    assert len(model.requests) == 1


def test_the_pack_examples_are_in_the_prompt():
    assert SPEC.examples
    system = build_request(SPEC, PAGE).system
    assert "<examples>" in system
    assert SPEC.examples[0].document.strip() in system


def test_an_example_that_fails_its_own_checks_is_refused(tmp_path):
    (tmp_path / "extractions").mkdir()
    (tmp_path / "extractions" / "bad.yaml").write_text(
        "description: d\n"
        "instructions: i\n"
        "fields:\n"
        "  revenue: {type: amount, description: r, quote: true}\n"
        "examples:\n"
        "  - document: 'Revenue 1 200'\n"
        "    values: {revenue: 1200000, revenue_quote: '1 200'}\n"
    )
    with pytest.raises(PackError, match="example 1 fails its own checks"):
        load_extraction(tmp_path, "bad")


def test_an_unknown_extraction_lists_the_known_ones():
    with pytest.raises(PackError, match="Known: key-figures"):
        load_extraction(PACK, "invoices")


def test_an_invalid_extraction_is_explained(tmp_path):
    (tmp_path / "extractions").mkdir()
    (tmp_path / "extractions" / "bad.yaml").write_text(
        "description: d\ninstructions: i\nfields:\n  unit: {type: choice, description: u}\n"
    )
    with pytest.raises(PackError, match="needs choices"):
        load_extraction(tmp_path, "bad")


def test_amounts_must_be_numbers_not_text():
    # With text allowed, a model may answer "1 480 312", which is not a number.
    revenue = output_model(SPEC).model_json_schema()["properties"]["revenue"]
    assert revenue["anyOf"] == [{"type": "number"}, {"type": "null"}]
