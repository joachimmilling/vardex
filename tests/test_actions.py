"""Actions: the pack's ontology, turned into tools that can change something."""

import json

import duckdb
import pytest

from vardex.actions import FileConnector, action_tools, check_fields
from vardex.approvals import Inbox
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.llm import ToolCall
from vardex.permissions import Permissions
from vardex.semantic import (
    Action,
    ActionField,
    ColumnDoc,
    Knowledge,
    Metric,
    ModelDoc,
    ObjectType,
    Ontology,
    Questions,
)
from vardex.tools import build_toolbox

TABLES = """
create table companies (orgnr varchar, name varchar);
insert into companies values ('100000008', 'DEMO LAKS AS');
"""

ACTION = Action(
    name="adjust_credit_limit",
    description="Propose a new credit limit for a company.",
    object="Company",
    connector="crm",
    confirm="Set {label}'s credit limit to {amount} NOK, because {reason}.",
    fields=[
        ActionField(name="amount", description="The new limit in NOK.", type="amount"),
        ActionField(name="reason", description="Why.", type="text"),
        ActionField(
            name="urgency",
            description="How soon.",
            type="choice",
            choices=["normal", "urgent"],
            required=False,
        ),
    ],
)


def knowledge() -> Knowledge:
    return Knowledge(
        tables={
            "companies": ModelDoc(
                name="companies",
                description="One row per company.",
                columns=[
                    ColumnDoc(name="orgnr", description="Organisation number."),
                    ColumnDoc(name="name", description="Name."),
                ],
            )
        },
        metrics=[
            Metric(
                name="companies",
                description="How many.",
                table="companies",
                sql="count(*)",
                unit="companies",
            )
        ],
        ontology=Ontology(
            objects=[
                ObjectType(
                    name="Company",
                    description="A company.",
                    table="companies",
                    key="orgnr",
                    label="name",
                )
            ],
            actions=[ACTION],
        ),
        questions=Questions(),
    )


def setup(tmp_path, level: str, **kwargs):
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute(TABLES)
    warehouse = DuckDBWarehouse(path)
    connector = FileConnector(tmp_path / "crm")
    inbox = Inbox(tmp_path / "inbox.jsonl")
    tools = action_tools(knowledge(), warehouse, {"crm": connector}, inbox)
    box = build_toolbox(Permissions(tools={"adjust_credit_limit": level}), extra=tools, **kwargs)
    return box, connector, inbox


def call(**arguments) -> ToolCall:
    return ToolCall("c1", "adjust_credit_limit", arguments)


def changed(connector: FileConnector) -> list[dict]:
    """Everything the outside system has been asked to do."""
    if not connector.log_path.is_file():
        return []
    lines = connector.log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_an_action_becomes_a_tool_with_the_right_schema(tmp_path):
    box, _, _ = setup(tmp_path, "draft")
    tool = box.tools["adjust_credit_limit"]
    assert tool.writes
    properties = tool.schema["properties"]
    assert properties["key"]["description"].startswith("The orgnr")
    assert properties["amount"]["type"] == "number"
    assert properties["urgency"]["enum"] == ["normal", "urgent"]
    assert tool.schema["required"] == ["key", "amount", "reason"]  # urgency is optional


def test_at_draft_it_puts_a_proposal_in_the_inbox_and_touches_nothing_else(tmp_path):
    box, connector, inbox = setup(tmp_path, "draft")
    result = box.call(call(key="100000008", amount=500000, reason="three good years"))
    assert not result.is_error and not result.wrote
    assert "Proposed, not done:" in result.content
    assert "Set DEMO LAKS AS's credit limit to 500000.0 NOK" in result.content
    waiting = inbox.items()
    assert [item.state for item in waiting] == ["proposed"]
    assert waiting[0].proposal.label == "DEMO LAKS AS"
    assert changed(connector) == []  # the CRM has not been told anything


def test_a_reason_with_its_own_full_stop_does_not_end_the_sentence_twice(tmp_path):
    box, _, inbox = setup(tmp_path, "draft")
    box.call(call(key="100000008", amount=500000, reason="equity was negative in 2025. "))
    sentence = inbox.items()[0].proposal.sentence
    assert sentence.endswith("because equity was negative in 2025.")


def test_the_same_work_proposed_twice_is_one_proposal(tmp_path):
    box, _, inbox = setup(tmp_path, "draft")
    box.context = {"work": "weekly-watch:brief:100000008:2026-W44"}
    first = box.call(call(key="100000008", amount=500000, reason="three good years"))
    second = box.call(call(key="100000008", amount=500000, reason="three good years"))
    assert len(inbox.items()) == 1
    assert "It was already proposed" not in first.content
    assert "It was already proposed" in second.content


def test_the_same_case_in_a_later_period_may_be_proposed_again(tmp_path):
    box, _, inbox = setup(tmp_path, "draft")
    box.context = {"work": "weekly-watch:brief:100000008:2026-W44"}
    box.call(call(key="100000008", amount=500000, reason="three good years"))
    box.context = {"work": "weekly-watch:brief:100000008:2026-W45"}
    box.call(call(key="100000008", amount=400000, reason="it got worse"))
    assert len(inbox.items()) == 2


def test_at_act_it_carries_the_change_out(tmp_path):
    box, connector, inbox = setup(tmp_path, "act")
    result = box.call(call(key="100000008", amount=500000, reason="three good years"))
    assert result.wrote
    assert result.content.startswith("Done:")
    assert [r["what"] for r in changed(connector)] == ["done"]
    # The inbox is the audit trail even when nobody had to be asked
    assert [item.state for item in inbox.items()] == ["done"]


def test_a_dry_run_writes_nothing_but_says_what_it_would_do(tmp_path):
    box, connector, inbox = setup(tmp_path, "act", dry_run=True)
    result = box.call(call(key="100000008", amount=1, reason="testing"))
    assert "Dry run, nothing written" in result.content
    assert "Set DEMO LAKS AS's credit limit" in result.content
    assert changed(connector) == []
    assert inbox.items() == []


def test_an_action_aimed_at_nothing_fails_before_anything_is_written(tmp_path):
    box, connector, inbox = setup(tmp_path, "act")
    result = box.call(call(key="999999999", amount=1, reason="testing"))
    assert result.is_error
    assert "there is no Company with orgnr 999999999" in result.content
    assert changed(connector) == [] and inbox.items() == []


def test_a_missing_field_is_an_error_the_model_can_fix(tmp_path):
    box, connector, _ = setup(tmp_path, "draft")
    result = box.call(call(key="100000008", amount=1))
    assert result.is_error
    assert "needs reason: Why." in result.content
    assert changed(connector) == []


def test_a_field_the_action_never_declared_is_refused(tmp_path):
    box, _, _ = setup(tmp_path, "act")
    result = box.call(call(key="100000008", amount=1, reason="x", approved_by="itself"))
    assert result.is_error
    assert "has no field approved_by" in result.content


def test_a_choice_outside_the_choices_is_refused(tmp_path):
    box, _, _ = setup(tmp_path, "draft")
    result = box.call(call(key="100000008", amount=1, reason="x", urgency="immediately"))
    assert result.is_error
    assert "must be one of normal, urgent" in result.content


def test_a_number_that_is_not_a_number_is_refused(tmp_path):
    box, _, _ = setup(tmp_path, "draft")
    result = box.call(call(key="100000008", amount="half a million", reason="x"))
    assert result.is_error
    assert "must be a number" in result.content


def test_a_key_with_a_quote_in_it_cannot_reach_the_sql(tmp_path):
    box, _, _ = setup(tmp_path, "act")
    result = box.call(call(key="' or 1=1 --", amount=1, reason="x"))
    assert result.is_error
    assert "there is no Company" in result.content  # quoted as a value, not pasted in


def test_an_action_whose_connector_is_missing_is_a_pack_error(tmp_path):
    from vardex.packs import PackError

    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute(TABLES)
    with pytest.raises(PackError, match="needs the connector crm"):
        action_tools(knowledge(), DuckDBWarehouse(path), {}, Inbox(tmp_path / "inbox.jsonl"))


def test_an_action_on_an_object_the_ontology_lacks_is_refused():
    with pytest.raises(ValueError, match="there is no object Customer"):
        Ontology(actions=[ACTION.model_copy(update={"object": "Customer"})])


def test_fields_are_converted_to_their_declared_types():
    values = check_fields(ACTION, {"amount": "500000", "reason": "x", "urgency": "urgent"})
    assert values == {"amount": 500000.0, "reason": "x", "urgency": "urgent"}


def test_a_change_can_be_undone_and_the_crm_goes_back_to_what_it_was(tmp_path):
    box, connector, inbox = setup(tmp_path, "act")
    box.call(call(key="100000008", amount=500000, reason="three good years"))
    box.context = {"work": "second"}
    box.call(call(key="100000008", amount=200000, reason="it got worse"))
    assert connector.state["adjust_credit_limit:100000008"]["amount"] == 200000.0
    second = next(i for i in inbox.items() if i.proposal.fields["amount"] == 200000.0)
    inbox.undo(second.reference, connector, by="joachim")
    assert connector.state["adjust_credit_limit:100000008"]["amount"] == 500000.0


def test_an_undo_refuses_when_somebody_else_has_been_there_since(tmp_path):
    from vardex.actions import ConnectorError

    box, connector, inbox = setup(tmp_path, "act")
    box.call(call(key="100000008", amount=500000, reason="three good years"))
    change = inbox.items()[0].change
    assert change is not None
    connector.state["adjust_credit_limit:100000008"] = {"amount": 1.0, "reason": "by hand"}
    with pytest.raises(ConnectorError, match="not as this change left it"):
        connector.undo(change)
