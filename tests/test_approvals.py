"""The approval inbox: what waits, who decided, what was changed, and what was put back."""

from datetime import UTC, datetime, timedelta

import pytest

from vardex.actions import Change, ConnectorError, FileConnector, Proposal
from vardex.approvals import ApprovalError, Inbox


def proposal(amount: float = 500000, work: str = "watch:brief:100000008:2026-W44") -> Proposal:
    return Proposal(
        action="adjust_credit_limit",
        key="100000008",
        label="DEMO LAKS AS",
        fields={"amount": amount, "reason": "equity went negative"},
        sentence=f"Set DEMO LAKS AS's credit limit to {amount} NOK.",
        made_at=datetime.now(UTC).isoformat(timespec="seconds"),
        idempotency_key=f"{work}:adjust_credit_limit:100000008",
        why="equity went negative in the 2025 accounts",
    )


def inbox_with(tmp_path, **kwargs) -> tuple[Inbox, FileConnector]:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.add(proposal(**kwargs), by="watch-20261026T063000Z")
    return inbox, FileConnector(tmp_path / "crm", name="the demo CRM")


def test_a_proposal_waits_with_the_sentence_a_person_has_to_agree_to(tmp_path):
    inbox, _ = inbox_with(tmp_path)
    item = inbox.items()[0]
    assert item.state == "proposed" and item.open
    assert "credit limit to 500000" in item.proposal.sentence
    assert item.proposal.why.startswith("equity went negative")
    assert item.history[0].by == "watch-20261026T063000Z"


def test_the_same_proposal_added_twice_stays_one_proposal(tmp_path):
    inbox, _ = inbox_with(tmp_path)
    reference, is_new = inbox.draft(proposal(), by="watch", needed_by="")
    assert not is_new
    assert len(inbox.items()) == 1
    assert reference == inbox.items()[0].reference


def test_approving_does_not_by_itself_change_anything_outside(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    item = inbox.decide(inbox.items()[0].reference, approved=True, by="joachim")
    assert item.state == "approved" and item.decided_by == "joachim"
    assert not connector.log_path.exists()  # the decision and the change are two steps


def test_carrying_it_out_changes_the_outside_and_records_what_it_was(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    reference = inbox.items()[0].reference
    inbox.decide(reference, approved=True, by="joachim")
    done = inbox.carry_out(reference, connector, by="joachim")
    assert done.state == "done"
    assert done.change is not None
    assert done.change.before == {}  # the CRM had no limit for this company before
    assert done.change.after == {"amount": 500000.0, "reason": "equity went negative"}
    assert connector.state["adjust_credit_limit:100000008"]["amount"] == 500000.0


def test_an_unapproved_proposal_cannot_be_carried_out(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    with pytest.raises(ApprovalError, match="is proposed, not approved"):
        inbox.carry_out(inbox.items()[0].reference, connector, by="joachim")


def test_a_decision_needs_a_name_and_is_made_only_once(tmp_path):
    inbox, _ = inbox_with(tmp_path)
    reference = inbox.items()[0].reference
    with pytest.raises(ApprovalError, match="needs a name"):
        inbox.decide(reference, approved=True, by="  ")
    inbox.decide(reference, approved=False, by="joachim", note="the figure is a filing error")
    with pytest.raises(ApprovalError, match="is rejected, not waiting"):
        inbox.decide(reference, approved=True, by="somebody else")


def test_a_change_can_be_undone_and_the_whole_story_is_still_there(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    reference = inbox.items()[0].reference
    inbox.decide(reference, approved=True, by="joachim")
    inbox.carry_out(reference, connector, by="joachim")
    undone = inbox.undo(reference, connector, by="joachim")
    assert undone.state == "undone"
    assert "adjust_credit_limit:100000008" not in connector.state
    assert [entry.state for entry in undone.history] == [
        "proposed",
        "approved",
        "done",
        "undone",
    ]


def test_a_connector_that_cannot_undo_is_refused_rather_than_pretended(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    reference = inbox.items()[0].reference
    inbox.decide(reference, approved=True, by="joachim")
    inbox.carry_out(reference, connector, by="joachim")
    connector.undoable = False
    with pytest.raises(ApprovalError, match="cannot undo a change"):
        inbox.undo(reference, connector, by="joachim")


def test_there_is_nothing_to_undo_before_anything_was_done(tmp_path):
    inbox, connector = inbox_with(tmp_path)
    with pytest.raises(ApprovalError, match="there is nothing to undo"):
        inbox.undo(inbox.items()[0].reference, connector, by="joachim")


def test_a_failure_in_the_other_system_is_recorded_and_can_be_tried_again(tmp_path):
    class Broken(FileConnector):
        def carry_out(self, proposal):
            raise ConnectorError("the CRM returned 503")

    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.add(proposal())
    reference = inbox.items()[0].reference
    inbox.decide(reference, approved=True, by="joachim")
    broken = Broken(tmp_path / "crm")
    with pytest.raises(ApprovalError, match="503"):
        inbox.carry_out(reference, broken, by="joachim")
    assert inbox.get(reference).state == "failed"
    # And the same approval can be carried out once the other system is back
    working = FileConnector(tmp_path / "crm")
    assert inbox.carry_out(reference, working, by="joachim").state == "done"


def test_a_reference_can_be_typed_as_a_prefix_but_never_guessed_at(tmp_path):
    inbox, _ = inbox_with(tmp_path)
    inbox.add(proposal(amount=400000, work="watch:brief:100000008:2026-W45"))
    full = inbox.items()[0].reference
    assert inbox.get(full[:-4]).reference == full  # the digest tells them apart
    with pytest.raises(ApprovalError, match="matches several proposals"):
        inbox.get("adjust_credit_limit-100000008")
    with pytest.raises(ApprovalError, match="There is no proposal"):
        inbox.get("nothing-like-it")


def test_what_is_overdue_is_what_a_run_escalates(tmp_path):
    inbox = Inbox(tmp_path / "inbox.jsonl")
    late = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds")
    soon = (datetime.now(UTC) + timedelta(hours=71)).isoformat(timespec="seconds")
    inbox.add(proposal(), needed_by=late)
    inbox.add(proposal(amount=1, work="other"), needed_by=soon)
    assert [item.proposal.fields["amount"] for item in inbox.overdue()] == [500000.0]


def test_a_proposal_nobody_answered_expires_instead_of_waiting_for_ever(tmp_path):
    inbox = Inbox(tmp_path / "inbox.jsonl")
    old = proposal().as_dict()
    old["made_at"] = (datetime.now(UTC) - timedelta(days=20)).isoformat(timespec="seconds")
    from vardex.actions import proposal_from_dict

    inbox.add(proposal_from_dict(old))
    inbox.add(proposal(amount=1, work="fresh"))
    closed = inbox.expire(after_days=14)
    assert [item.state for item in closed] == ["expired"]
    assert len(inbox.items(state="proposed")) == 1


def test_one_unreadable_line_does_not_hide_the_rest_of_the_inbox(tmp_path):
    inbox, _ = inbox_with(tmp_path)
    with inbox.path.open("a", encoding="utf-8") as file:
        file.write("this is not JSON\n")
    assert len(inbox.items()) == 1


def test_a_change_made_without_waiting_is_in_the_same_log(tmp_path):
    inbox = Inbox(tmp_path / "inbox.jsonl")
    made = proposal()
    inbox.record_done(
        made,
        Change(reference="crm-1", before={}, after=made.fields, at=made.made_at),
        by="joachim",
    )
    item = inbox.items()[0]
    assert item.state == "done"
    assert [entry.state for entry in item.history] == ["proposed", "approved", "done"]
