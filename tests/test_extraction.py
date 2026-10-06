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
from vardex.llm import Message, ModelError
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


def test_a_failure_keeps_the_answer_so_it_can_be_logged():
    with pytest.raises(ExtractionError) as caught:
        extract(FakeModel(reply(operating_profit=None)), SPEC, PAGE, repairs=0)
    assert [a.cost_usd for a in caught.value.answers] == [Decimal("0.004")]


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


TOTAL_EXAMPLE = (
    "examples:\n  - document: Total 1 200\n    values: {amount: 1200, amount_quote: 1 200}\n"
)


def write_extraction(folder: Path, examples: str) -> Path:
    (folder / "extractions").mkdir(parents=True)
    (folder / "extractions" / "demo.yaml").write_text(
        "description: d\ninstructions: Find the amount.\nfields:\n"
        "  amount: {type: amount, description: a, quote: true}\n"
        "  note: {type: text, description: n, required: false}\n" + examples,
        encoding="utf-8",
    )
    return folder


def test_the_pack_has_a_worked_example_that_passes_its_checks():
    example = SPEC.examples[0]
    assert "TNOK" in example.document
    assert example.values["operating_profit"] == -3145
    assert example.values["operating_profit_quote"] == "(3 145)"


def test_examples_follow_the_instructions_in_the_system_prompt():
    model = FakeModel(reply())
    extract(model, SPEC, PAGE)
    system = model.requests[0].system
    assert system.index("</task>") < system.index("<examples>") < system.index("<example>")
    assert SPEC.examples[0].document.strip() in system
    assert '"operating_profit": -3145, "operating_profit_quote": "(3 145)"' in system


def test_an_extraction_without_examples_shows_none(tmp_path):
    spec = load_extraction(write_extraction(tmp_path, ""), "demo")
    model = FakeModel('{"problem": null, "amount": 5, "amount_quote": "5", "note": null}')
    extract(model, spec, "Total 5")
    assert "<example" not in model.requests[0].system


def test_a_left_out_value_is_shown_as_empty(tmp_path):
    examples = TOTAL_EXAMPLE
    spec = load_extraction(write_extraction(tmp_path, examples), "demo")
    model = FakeModel('{"problem": null, "amount": 5, "amount_quote": "5", "note": null}')
    extract(model, spec, "Total 5")
    answer = '{"problem": null, "amount": 1200, "amount_quote": "1 200", "note": null}'
    assert answer in model.requests[0].system


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ("{amount: 1200, amount_quote: 1 300}", "the quote for amount is not in the document"),
        ("{amount: 120, amount_quote: 1 200}", "amount is 120, but the document says '1 200'"),
        ("{amount_quote: 1 200}", "amount is missing"),
        ("{amount: lots, amount_quote: 1 200}", "amount: "),
        ("{amount: 1200, amount_quote: 1 200, total: 1}", "total is not a field"),
    ],
)
def test_an_example_that_fails_its_checks_is_a_pack_error(tmp_path, values, reason):
    examples = TOTAL_EXAMPLE + f"  - document: Total 1 200\n    values: {values}\n"
    with pytest.raises(PackError, match="example 2 fails its checks") as caught:
        load_extraction(write_extraction(tmp_path, examples), "demo")
    assert reason in str(caught.value)


def test_an_example_with_a_problem_only_needs_the_right_shape(tmp_path):
    examples = "examples:\n  - document: A poem.\n    values: {problem: No amounts here.}\n"
    spec = load_extraction(write_extraction(tmp_path, examples), "demo")
    assert spec.examples[0].values == {"problem": "No amounts here."}

    shape = "examples:\n  - document: A poem.\n    values: {problem: None here., amount: [1]}\n"
    with pytest.raises(PackError, match="example 1 fails its checks: amount: "):
        load_extraction(write_extraction(tmp_path / "shape", shape), "demo")


def test_examples_show_values_in_the_schema_types(tmp_path):
    examples = TOTAL_EXAMPLE.replace("amount: 1200", 'amount: "1200"')
    spec = load_extraction(write_extraction(tmp_path, examples), "demo")
    model = FakeModel('{"problem": null, "amount": 5, "amount_quote": "5", "note": null}')
    extract(model, spec, "Total 5")
    assert '"amount": 1200,' in model.requests[0].system


def test_extract_sends_the_request_build_request_gives():
    model = FakeModel(reply())

    extract(model, SPEC, PAGE, effort="high")

    sent = model.requests[0]
    built = build_request(SPEC, PAGE, effort="high")
    assert (sent.system, sent.messages, sent.effort) == (built.system, built.messages, "high")
    assert sent.output_schema.model_json_schema() == built.output_schema.model_json_schema()


def test_a_failed_answer_is_repaired_once_by_default():
    bad = reply(revenue=1480312000)
    model = FakeModel(bad, reply())

    result = extract(model, SPEC, PAGE)

    assert result.values["revenue"] == Decimal("1480312")
    assert len(result.answers) == 2
    first, second = model.requests
    assert second.system == first.system
    assert second.messages[:2] == [first.messages[0], Message("assistant", bad)]
    assert second.messages[2].role == "user"
    assert "- revenue is 1480312000, but the document says '1 480 312'" in second.messages[2].text
    assert "same JSON format" in second.messages[2].text
    assert second.output_schema is first.output_schema


def test_an_answer_of_the_wrong_shape_is_repaired():
    model = FakeModel(reply(unit="billions"), reply())
    extract(model, SPEC, PAGE)
    assert "unit: must be one of" in model.requests[1].messages[2].text


def test_repairs_stop_at_the_limit_and_keep_every_answer():
    model = FakeModel(*[reply(operating_profit=None)] * 3)

    with pytest.raises(ExtractionError, match="operating_profit is missing") as caught:
        extract(model, SPEC, PAGE, repairs=2)

    assert len(caught.value.answers) == 3
    assert len(model.requests[2].messages) == 5  # document, then two answers and their problems


def test_zero_repairs_sends_one_request():
    model = FakeModel(reply(operating_profit=None), reply())
    with pytest.raises(ExtractionError):
        extract(model, SPEC, PAGE, repairs=0)
    assert len(model.requests) == 1


def test_a_problem_reported_by_the_model_is_not_repaired():
    model = FakeModel(reply(problem="There are no figures on this page."), reply())
    with pytest.raises(ExtractionError, match="no figures") as caught:
        extract(model, SPEC, PAGE)
    assert len(model.requests) == 1
    assert len(caught.value.answers) == 1


def test_an_answer_cut_off_is_not_repaired():
    model = FakeModel('{"problem": nu', reply(), stop_reason="max_tokens")
    with pytest.raises(ExtractionError, match="stopped early"):
        extract(model, SPEC, PAGE)
    assert len(model.requests) == 1


@pytest.mark.parametrize("repairs", [-1, 4])
def test_repairs_must_be_from_0_to_3(repairs):
    with pytest.raises(ValueError, match="from 0 to 3"):
        extract(FakeModel(reply()), SPEC, PAGE, repairs=repairs)


def test_a_failed_repair_call_keeps_the_answers_already_paid_for():
    model = FakeModel(reply(operating_profit=None))  # no second reply: the repair call fails
    send = model.send

    def send_once(request, on_text=None):
        if model.requests:
            raise ModelError("no connection")
        return send(request, on_text)

    model.send = send_once
    with pytest.raises(ExtractionError, match="the repair failed: no connection") as caught:
        extract(model, SPEC, PAGE)
    assert len(caught.value.answers) == 1
