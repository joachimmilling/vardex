"""Workflows: when a run is due, what the steps do, and what resuming does not do twice."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import duckdb
import httpx2
import pytest
from fakes import FakeModel, Turn

from vardex.actions import FileConnector, action_tools
from vardex.approvals import Inbox
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.llm import ModelError
from vardex.permissions import Approvals, Budgets, Permissions
from vardex.runs import RunStore
from vardex.semantic import Action as OntologyAction
from vardex.semantic import (
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
from vardex.workflows import (
    Case,
    Connections,
    Schedule,
    Watermarks,
    WorkflowSpec,
    check_workflow,
    is_due,
    live_feed,
    run_workflow,
    start,
)

TABLES = """
create table companies (orgnr varchar, name varchar, equity decimal(18, 0), bankrupt boolean);
insert into companies values
    ('100000008', 'DEMO LAKS AS', -4000000, false),
    ('200000006', 'DEMO BYGG AS', -900000, true);
"""

FIND_SQL = """
select orgnr, name, 'Equity is negative.' as reason, equity, bankrupt
from companies
where equity < 0
order by equity
"""

WORKFLOW = {
    "description": "A weekly watch.",
    "schedule": {"every": "week", "weekday": "monday", "at": "06:30", "timezone": "Europe/Oslo"},
    "budget": {"per_run": 2.0},
    "steps": [
        {"id": "risks", "uses": "find", "about": "Company", "sql": FIND_SQL},
        {
            "id": "brief",
            "uses": "brief",
            "question": "{label} ({key}) came up: {reason} Equity {equity}.",
            "propose": "adjust_credit_limit",
        },
        {"id": "digest", "uses": "report", "title": "Weekly company watch"},
    ],
}

ACTION = OntologyAction(
    name="adjust_credit_limit",
    description="Propose a new credit limit.",
    object="Company",
    connector="crm",
    confirm="Set {label}'s credit limit to {amount} NOK, because {reason}.",
    fields=[
        ActionField(name="amount", description="The new limit.", type="amount"),
        ActionField(name="reason", description="Why.", type="text"),
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
                    ColumnDoc(name="equity", description="Equity in NOK."),
                    ColumnDoc(name="bankrupt", description="True when bankrupt."),
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


def permissions(level: str = "approve") -> Permissions:
    return Permissions(
        budgets=Budgets(per_run=Decimal("2.00")),
        approvals=Approvals(approvers=["joachim"], needed_within_hours=72, escalate_to="joachim"),
        tools={
            "describe_tool": "read",
            "list_tables": "read",
            "run_sql": "read",
            "adjust_credit_limit": level,
        },
    )


def spec(**changes) -> WorkflowSpec:
    body = dict(WORKFLOW) | changes
    return WorkflowSpec(**body).model_copy(update={"name": "weekly-watch"})


def wire(tmp_path, client, *, level: str = "approve", feed_pages=None, dry_run=False):
    """Everything a run can reach, with a fake model and no network unless a feed is given."""
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute(TABLES)
    warehouse = DuckDBWarehouse(path)
    rules = permissions(level)
    inbox = Inbox(tmp_path / "inbox.jsonl")
    connector = FileConnector(tmp_path / "crm", name="the demo CRM")
    box = build_toolbox(
        rules,
        warehouse=warehouse,
        knowledge=knowledge(),
        extra=action_tools(knowledge(), warehouse, {"crm": connector}, inbox),
        dry_run=dry_run,
    )

    def http() -> httpx2.Client:
        def handler(request: httpx2.Request) -> httpx2.Response:
            cursor = request.url.params.get("oppdateringsid")
            items = (feed_pages or {}).get(cursor, [])
            body = {"_embedded": {"updates": items}} if items else {}
            return httpx2.Response(200, content=json.dumps(body))

        return httpx2.Client(transport=httpx2.MockTransport(handler))

    feed = live_feed(http)

    return Connections(
        pack="demo",
        folder="packs/demo",
        knowledge=knowledge(),
        permissions=rules,
        warehouse=warehouse,
        client=client,
        inbox=inbox,
        store=RunStore(tmp_path / "runs"),
        marks=Watermarks(tmp_path / "state" / "weekly-watch.json"),
        reports=tmp_path / "reports",
        toolbox=lambda context: replace(box, context=context, calls=[]),
        feed=feed,
    ), connector


def note(text: str = "Equity went negative in 2025. I would lower the limit.") -> str:
    return text


def propose(amount: float = 500000) -> Turn:
    return Turn(
        "",
        [("adjust_credit_limit", {"key": "100000008", "amount": amount, "reason": "negative"})],
    )


# --- the schedule ----------------------------------------------------------------------


def oslo(text: str) -> datetime:
    from zoneinfo import ZoneInfo

    return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("Europe/Oslo"))


def test_the_last_moment_a_weekly_workflow_should_have_run():
    weekly = Schedule(every="week", weekday="monday", at="06:30", timezone="Europe/Oslo")
    # Monday 26 October 2026 at 08:00: this morning's slot has passed
    assert weekly.due_before(oslo("2026-10-26T08:00")) == oslo("2026-10-26T06:30")
    # Monday at 06:00, before the slot: the one it is measured against is last Monday
    assert weekly.due_before(oslo("2026-10-26T06:00")) == oslo("2026-10-19T06:30")
    # Any other day: still last Monday
    assert weekly.due_before(oslo("2026-10-29T23:00")) == oslo("2026-10-26T06:30")


def test_a_daily_workflow_rolls_over_at_its_own_hour():
    daily = Schedule(every="day", at="06:30", timezone="Europe/Oslo")
    assert daily.due_before(oslo("2026-10-26T06:29")) == oslo("2026-10-25T06:30")
    assert daily.due_before(oslo("2026-10-26T06:31")) == oslo("2026-10-26T06:30")
    assert daily.next_after(oslo("2026-10-26T06:31")) == oslo("2026-10-27T06:30")


def test_the_period_names_the_piece_of_work():
    weekly = Schedule(every="week", weekday="monday", at="06:30", timezone="Europe/Oslo")
    assert weekly.period(oslo("2026-10-26T08:00")) == "2026-W44"
    assert weekly.period(oslo("2026-10-30T23:00")) == "2026-W44"  # same week's work
    assert weekly.period(oslo("2026-11-02T08:00")) == "2026-W45"
    daily = Schedule(every="day", at="06:30")
    assert daily.period(oslo("2026-10-26T08:00")) == "2026-10-26"


def test_a_time_zone_that_is_not_a_place_is_refused():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="is not a time zone name"):
        Schedule(timezone="UTC+2")


def test_due_is_about_work_done_not_about_the_clock(tmp_path):
    store = RunStore(tmp_path / "runs")
    workflow = spec()
    assert is_due(workflow, store)  # never run: due
    run = start(workflow, wire(tmp_path, FakeModel("unused"))[0], trigger="manual", dry_run=False)
    run.state = "done"
    run.started_at = datetime.now(UTC).isoformat(timespec="seconds")
    store.save(run)
    assert not is_due(workflow, store)  # this week's run has happened


def test_a_run_that_failed_leaves_the_workflow_still_due(tmp_path):
    store = RunStore(tmp_path / "runs")
    workflow = spec()
    run = start(workflow, wire(tmp_path, FakeModel("unused"))[0], trigger="manual", dry_run=False)
    run.state = "failed"
    store.save(run)
    assert is_due(workflow, store)


def test_a_workflow_with_no_schedule_is_never_due(tmp_path):
    assert not is_due(spec(schedule=None), RunStore(tmp_path / "runs"))


# --- checking the pack's declaration ---------------------------------------------------


def test_a_workflow_is_checked_without_running_it():
    assert check_workflow(spec(), knowledge(), permissions()) == []


def test_sql_that_reads_a_table_the_pack_never_described_is_caught():
    broken = spec(
        steps=[
            {"id": "risks", "uses": "find", "about": "Company", "sql": "select * from secrets"},
            {"id": "digest", "uses": "report", "title": "x"},
        ]
    )
    problems = check_workflow(broken, knowledge(), permissions())
    assert any("secrets" in problem for problem in problems)


def test_an_action_the_pack_only_allows_to_read_could_never_be_proposed():
    problems = check_workflow(spec(), knowledge(), permissions(level="read"))
    assert any("could never propose anything" in problem for problem in problems)


def test_a_workflow_with_no_report_step_leaves_nothing_to_read():
    quiet = spec(steps=[{"id": "risks", "uses": "find", "about": "Company", "sql": FIND_SQL}])
    assert any(
        "no report step" in problem for problem in check_workflow(quiet, knowledge(), permissions())
    )


def test_a_question_that_asks_for_a_figure_the_case_lacks_says_which():
    from vardex.workflows import WorkflowError

    case = Case(key="1", label="DEMO AS", reason="because", facts={"equity": "-1"})
    assert case.filled("{label} has {equity}") == "DEMO AS has -1"
    with pytest.raises(WorkflowError, match="asks for 'margin'"):
        case.filled("{label} has {margin}")


# --- running it ------------------------------------------------------------------------


def test_a_run_finds_its_cases_briefs_each_one_and_writes_the_report(tmp_path):
    client = FakeModel(note(), note("DEMO BYGG AS is bankrupt. Stop the credit."))
    connections, _ = wire(tmp_path, client)
    workflow = spec()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=False), connections
    )
    assert run.state == "done"
    assert [step.state for step in run.steps] == ["done", "done", "done"]
    assert len(run.facts["cases"]) == 2
    assert set(run.facts["notes"]) == {"100000008", "200000006"}
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert "# Weekly company watch" in report
    assert "DEMO LAKS AS" in report and "Equity is negative." in report
    assert "Nothing. No action met the bar this run." in report


def test_a_proposal_lands_in_the_inbox_with_a_deadline_and_nothing_is_changed(tmp_path):
    client = FakeModel(propose(), note(), note())
    connections, connector = wire(tmp_path, client)
    workflow = spec()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    waiting = connections.inbox.items(state="proposed")
    assert len(waiting) == 1
    assert waiting[0].needed_by  # the pack's 72 hours
    assert waiting[0].proposal.why == "Equity is negative."
    assert waiting[0].history[0].by == run.id
    assert not connector.log_path.exists()  # the CRM has not been told anything
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert "Needs a decision (1)" in report
    assert "vardex approve packs/demo" in report


def test_nobody_is_at_the_keyboard_so_a_writing_tool_can_only_draft(tmp_path):
    # The toolbox has no approver, which is what an unattended run means
    client = FakeModel(propose(), note(), note())
    connections, connector = wire(tmp_path, client, level="approve")
    workflow = spec()
    run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert connections.inbox.items(state="proposed")
    assert not connector.state  # nothing was carried out


def test_a_dry_run_changes_nothing_and_says_so_in_the_report(tmp_path):
    client = FakeModel(propose(), note(), note())
    connections, connector = wire(tmp_path, client, dry_run=True)
    workflow = spec()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=True), connections
    )
    assert connections.inbox.items() == []
    assert not connector.state
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert "dry run, nothing written" in report


def test_every_step_is_saved_so_a_run_can_be_read_while_it_is_running(tmp_path):
    client = FakeModel(note(), note())
    connections, _ = wire(tmp_path, client)
    workflow = spec()
    run = start(workflow, connections, trigger="manual", dry_run=False)
    # The file exists before any step has run
    assert connections.store.load(run.id).state == "running"
    run_workflow(workflow, run, connections)
    assert connections.store.load(run.id).state == "done"


# --- stopping and resuming -------------------------------------------------------------


def test_a_run_that_stops_on_its_budget_is_stopped_not_failed(tmp_path):
    from vardex.budgets import Budget

    connections, _ = wire(tmp_path, FakeModel(note(), note()))
    # A cap so small that the first call is refused before it is made
    capped = Budget(connections.client, caps=Budgets(per_run=Decimal("0.0001")))
    connections = replace(connections, client=capped)
    workflow = spec()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert run.state == "stopped"
    assert [step.state for step in run.steps] == ["done", "stopped", "done"]
    assert "budget" in run.note
    assert f"vardex run packs/demo --resume {run.id}" in run.note
    # The report is still written: a run that got half way says what it found
    assert (tmp_path / "reports" / f"{run.id}.md").is_file()


def test_a_run_that_hits_a_bug_is_recorded_as_failed_and_the_bug_is_raised(tmp_path):
    class Broken(FakeModel):
        def send(self, request, on_text=None):
            raise RuntimeError("a driver fell over")

    connections, _ = wire(tmp_path, Broken())
    workflow = spec()
    run = start(workflow, connections, trigger="schedule", dry_run=False)
    with pytest.raises(RuntimeError, match="a driver fell over"):
        run_workflow(workflow, run, connections)
    saved = connections.store.load(run.id)
    assert saved.state == "failed"  # not "running" for ever
    assert "RuntimeError: a driver fell over" in saved.note


def test_a_run_stopped_by_its_budget_is_finished_when_the_budget_is_raised(tmp_path):
    from vardex.budgets import Budget

    connections, _ = wire(tmp_path, FakeModel(note(), note()))
    capped = Budget(connections.client, caps=Budgets(per_run=Decimal("0.0001")))
    workflow = spec()
    stopped = run_workflow(
        workflow,
        start(workflow, replace(connections, client=capped), trigger="schedule", dry_run=False),
        replace(connections, client=capped),
    )
    assert stopped.state == "stopped"
    # The same run, with the cap raised: the step that was cut short is attempted again
    done = run_workflow(workflow, connections.store.load(stopped.id), connections)
    assert done.state == "done"
    assert len(done.facts["notes"]) == 2
    report = (tmp_path / "reports" / f"{done.id}.md").read_text(encoding="utf-8")
    assert "Finished: **done**" in report  # the report from the stopped attempt is replaced


def test_resuming_carries_on_from_the_case_it_stopped_at(tmp_path):
    client = FakeModel(note("first"))  # only one reply: the second case cannot be briefed
    connections, _ = wire(tmp_path, client)
    workflow = spec()
    run = start(workflow, connections, trigger="schedule", dry_run=False)
    with pytest.raises(IndexError):  # the fake model runs out of replies
        run_workflow(workflow, run, connections)
    saved = connections.store.load(run.id)
    assert len(saved.facts["notes"]) == 1  # one case was finished and written down

    carried_on = replace(connections, client=FakeModel(note("second")))
    done = run_workflow(workflow, saved, carried_on)
    assert done.state == "done"
    assert len(done.facts["notes"]) == 2
    assert done.attempts == 2


def test_resuming_never_repeats_a_call_that_wrote(tmp_path):
    """The whole of idempotency: the same proposal, asked for twice, is one proposal."""
    client = FakeModel(propose(), note("first"))
    connections, _ = wire(tmp_path, client)
    workflow = spec(steps=[WORKFLOW["steps"][0], WORKFLOW["steps"][1]])
    run = start(workflow, connections, trigger="schedule", dry_run=False)
    with pytest.raises(IndexError):
        run_workflow(workflow, run, connections)
    first = connections.store.load(run.id)
    assert len(connections.inbox.items()) == 1
    assert len(first.journal) >= 2  # the writing call, and the case that finished

    # The resumed run asks for the same change again, and is given the first answer back
    again = replace(connections, client=FakeModel(propose(), note("second")))
    done = run_workflow(workflow, first, again)
    assert done.state == "done"
    assert len(connections.inbox.items()) == 1  # still one proposal


def test_a_recorded_call_tells_the_model_it_was_done_by_an_earlier_attempt(tmp_path):
    from vardex.llm import ToolCall
    from vardex.workflows import journalled

    connections, _ = wire(tmp_path, FakeModel("unused"))
    workflow = spec()
    run = start(workflow, connections, trigger="manual", dry_run=False)
    box = journalled(connections.toolbox({"work": "w"}), run)
    call = ToolCall("c1", "adjust_credit_limit", {"key": "100000008", "amount": 1, "reason": "x"})
    first = box.call(call)
    assert "Proposed, not done" in first.content
    second = box.call(call)
    assert "Done by an earlier attempt" in second.content
    assert len(connections.inbox.items()) == 1


def test_a_step_that_fails_leaves_the_run_failed_and_says_which_step(tmp_path):
    class Broken(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("the API is down", retryable=False)

    connections, _ = wire(tmp_path, Broken("unused"))
    workflow = spec()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert run.state == "failed"
    assert run.note.startswith("brief:")
    assert "the API is down" in run.note


def test_a_case_the_agent_could_not_finish_is_named_in_the_report(tmp_path):
    # A model that keeps asking for tools runs out of steps instead of answering
    client = FakeModel(Turn("", [("list_tables", {})]), note())
    connections, _ = wire(tmp_path, client)
    workflow = spec(
        steps=[
            WORKFLOW["steps"][0],
            {"id": "brief", "uses": "brief", "question": "{label}", "max_steps": 1},
            WORKFLOW["steps"][2],
        ]
    )
    run = run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=False), connections
    )
    assert run.state == "done"
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert "What went wrong" in report
    assert "steps:" in report  # why it did not finish, in the run's own words


# --- the watch step --------------------------------------------------------------------


def change(number: int, key: str, kind: str = "Endring") -> dict:
    return {
        "oppdateringsid": number,
        "organisasjonsnummer": key,
        "endringstype": kind,
        "dato": f"2026-10-{number:02d}T01:00:00.000Z",
    }


FEED = {
    "url": "https://register.example/api/updates",
    "items": "_embedded.updates",
    "key": "organisasjonsnummer",
    "position": "oppdateringsid",
    "cursor": "oppdateringsid",
    "kind": "endringstype",
    "page_size": 10,
}


def watching(**changes) -> WorkflowSpec:
    return spec(
        steps=[
            {"id": "changes", "uses": "watch", "feed": FEED, **changes},
            dict(WORKFLOW["steps"][0]) | {"changed_only": True},
            WORKFLOW["steps"][1],
            WORKFLOW["steps"][2],
        ]
    )


def test_the_watch_step_keeps_only_what_the_pack_follows_and_saves_the_cursor(tmp_path):
    pages = {None: [change(1, "100000008"), change(2, "999999999")]}
    connections, _ = wire(tmp_path, FakeModel(note()), feed_pages=pages)
    workflow = watching(follow="select orgnr from companies")
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert run.state == "done"
    assert run.facts["changed"] == ["100000008"]  # 999999999 is not a company we follow
    assert run.facts["feed:changes"]["read"] == 2
    assert connections.marks.get("changes") == "3"  # one past the last change seen
    # changed_only means the second company, which did not change, is not looked at
    assert [case["key"] for case in run.facts["cases"]] == ["100000008"]


def test_a_dry_run_does_not_move_the_cursor(tmp_path):
    pages = {None: [change(1, "100000008")]}
    connections, _ = wire(tmp_path, FakeModel(note()), feed_pages=pages, dry_run=True)
    workflow = watching()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=True), connections
    )
    assert run.facts["changed"] == ["100000008"]
    assert connections.marks.get("changes") is None  # a rehearsal consumes nothing


def test_a_second_run_reads_the_feed_from_where_the_first_one_stopped(tmp_path):
    pages = {None: [change(1, "100000008")], "2": []}
    connections, _ = wire(tmp_path, FakeModel(note()), feed_pages=pages)
    workflow = watching()
    run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    second = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert second.facts["changed"] == []  # nothing new since the cursor
    assert second.facts["cases"] == []


# --- cases at the same time ------------------------------------------------------------


def test_cases_are_looked_into_at_the_same_time_and_recorded_in_order(tmp_path):
    import time

    class Slow(FakeModel):
        def send(self, request, on_text=None):
            time.sleep(0.2)
            return super().send(request, on_text)

    connections, _ = wire(tmp_path, Slow(note("first"), note("second")))
    workflow = spec(
        steps=[
            WORKFLOW["steps"][0],
            dict(WORKFLOW["steps"][1]) | {"workers": 2},
            WORKFLOW["steps"][2],
        ]
    )
    started = time.monotonic()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=False), connections
    )
    assert run.state == "done"
    assert time.monotonic() - started < 0.38  # two 0.2 s cases, not 0.4 s of them
    # The cases are written back in the order the find step put them in, whoever finished first
    assert list(run.facts["notes"]) == [case["key"] for case in run.facts["cases"]]


def test_one_cap_holds_however_many_threads_are_asking(tmp_path):
    """Two calls in flight at once must not both be told there is room for the last one."""
    import threading
    import time
    from decimal import Decimal

    from vardex.budgets import Budget, BudgetError
    from vardex.permissions import Budgets

    inside = threading.Event()

    class Slow(FakeModel):
        def send(self, request, on_text=None):
            inside.set()
            time.sleep(0.2)
            return super().send(request, on_text)

    # Room for one call's worst case and not two
    budget = Budget(Slow("one", "two"), caps=Budgets(per_run=Decimal("0.05")))
    refused: list[Exception] = []

    def ask():
        try:
            budget.send(a_request())
        except BudgetError as err:
            refused.append(err)

    first = threading.Thread(target=ask)
    first.start()
    inside.wait(1)
    ask()  # the second asks while the first is still in flight
    first.join()
    assert len(refused) == 1
    assert "could cost up to" in str(refused[0])


def a_request():
    from vardex.llm import Message, Request

    return Request(system="x" * 200, messages=[Message("user", "y" * 200)], max_tokens=4096)


# --- closing what nobody answered ------------------------------------------------------


def tidying(**changes) -> WorkflowSpec:
    return spec(
        steps=[
            {"id": "tidy", "uses": "tidy"},
            *WORKFLOW["steps"],
        ],
        **changes,
    )


def an_old_proposal(connections, days: int, hours_late: int = 0) -> str:
    """A proposal made `days` ago, with a deadline `hours_late` hours in the past."""
    from datetime import timedelta

    from vardex.actions import Proposal

    made = datetime.now(UTC) - timedelta(days=days)
    proposal = Proposal(
        action="adjust_credit_limit",
        key="100000008",
        label="DEMO LAKS AS",
        fields={"amount": 1.0, "reason": "last week"},
        sentence="Set DEMO LAKS AS's credit limit to 1 NOK.",
        made_at=made.isoformat(timespec="seconds"),
        idempotency_key=f"old-{days}-{hours_late}",
    )
    needed = (datetime.now(UTC) - timedelta(hours=hours_late)).isoformat(timespec="seconds")
    connections.inbox.add(proposal, by="an earlier run", needed_by=needed if hours_late else "")
    return proposal.reference


def test_a_proposal_nobody_answered_is_closed_and_said_so(tmp_path):
    connections, _ = wire(tmp_path, FakeModel(note(), note()))
    stale = an_old_proposal(connections, days=20)
    workflow = tidying()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert run.facts["expired"] == [stale]
    assert connections.inbox.get(stale).state == "expired"
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert "Closed, unanswered (1)" in report


def test_what_is_late_goes_to_the_top_with_a_name_on_it(tmp_path):
    connections, _ = wire(tmp_path, FakeModel(note(), note()))
    late = an_old_proposal(connections, days=2, hours_late=5)
    workflow = tidying()
    run = run_workflow(
        workflow, start(workflow, connections, trigger="schedule", dry_run=False), connections
    )
    assert run.facts["overdue"] == [late]
    report = (tmp_path / "reports" / f"{run.id}.md").read_text(encoding="utf-8")
    assert report.index("Late — joachim, these are yours (1)") < report.index("Needs a decision")


def test_a_dry_run_does_not_even_decide_to_give_up(tmp_path):
    connections, _ = wire(tmp_path, FakeModel(note(), note()), dry_run=True)
    stale = an_old_proposal(connections, days=30)
    workflow = tidying()
    run_workflow(
        workflow, start(workflow, connections, trigger="manual", dry_run=True), connections
    )
    assert connections.inbox.get(stale).state == "proposed"


def test_a_recorded_feed_read_replays_without_the_network(tmp_path):
    from datetime import date

    from vardex.replay import NotRecorded, Recording, RecordingFeed, ReplayFeed

    pages = {None: [change(1, "100000008")], "2": []}
    connections, _ = wire(tmp_path, FakeModel(note()), feed_pages=pages)
    recording = Recording("claude-sonnet-5-5", "DuckDB", "duckdb", date.today().isoformat())
    workflow = watching()

    taped = replace(connections, feed=RecordingFeed(connections.feed, recording))
    first = run_workflow(workflow, start(workflow, taped, trigger="schedule", dry_run=False), taped)
    assert first.facts["changed"] == ["100000008"]
    assert len(recording.feeds) == 1

    saved = tmp_path / "weekly-watch.jsonl"
    recording.save(saved)
    played = replace(
        connections,
        feed=ReplayFeed(Recording.load(saved)),
        marks=Watermarks(tmp_path / "state" / "again.json"),  # reading from the start again
        client=FakeModel(note()),
    )
    again = run_workflow(workflow, start(workflow, played, trigger="manual", dry_run=False), played)
    assert again.facts["changed"] == ["100000008"]

    # And a read the recording does not cover is a loud failure, not a quiet empty feed
    moved = replace(played, marks=Watermarks(tmp_path / "state" / "moved.json"))
    moved.marks.set("changes", "99999")
    with pytest.raises(NotRecorded, match="which the recording does not have"):
        run_workflow(watching(), start(watching(), moved, trigger="manual", dry_run=False), moved)


def test_a_run_can_be_started_in_the_period_it_was_recorded_in(tmp_path):
    connections, _ = wire(tmp_path, FakeModel(note()))
    workflow = spec()
    recorded = datetime(2026, 1, 7, 12, tzinfo=UTC)  # a Wednesday in week 2
    run = start(workflow, connections, trigger="manual", dry_run=False, period_of=recorded)
    # Not this week's: the proposals' references are made from the period
    assert run.facts["period"] == workflow.period(recorded) != workflow.period(datetime.now(UTC))
