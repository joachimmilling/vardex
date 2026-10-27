"""Scheduled work through the CLI: running it, reading it, and deciding on what it drafted."""

import json

import duckdb
import pytest
from fakes import FakeModel, Turn
from typer.testing import CliRunner

from vardex import cli

runner = CliRunner()

MODELS = """
models:
  - name: companies
    description: One row per company.
    columns:
      - {name: orgnr, description: Organisation number.}
      - {name: name, description: Registered name.}
      - {name: equity, description: Equity in NOK.}
"""
METRICS = """
metrics:
  - {name: companies, description: How many., table: companies, sql: count(*), unit: companies}
"""
ONTOLOGY = """
objects:
  - name: Company
    description: A company.
    table: companies
    key: orgnr
    label: name
actions:
  - name: adjust_credit_limit
    description: Propose a new credit limit.
    object: Company
    connector: crm
    confirm: Set {label}'s credit limit to {amount} NOK, because {reason}.
    fields:
      - {name: amount, description: The new limit., type: amount}
      - {name: reason, description: Why., type: text}
"""
PERMISSIONS = """
budgets:
  per_run: 0.50
  per_day: 20.00
approvals:
  approvers: [joachim]
  needed_within_hours: 72
tools:
  describe_tool: read
  list_tables: read
  describe_table: read
  run_sql: read
  make_chart: read
  adjust_credit_limit: approve
"""
WORKFLOW = """
description: A weekly watch over the companies we follow.
schedule:
  every: week
  weekday: monday
  at: "06:30"
  timezone: Europe/Oslo
budget:
  per_run: 2.00
steps:
  - id: risks
    uses: find
    about: Company
    sql: |
      select orgnr, name, 'Equity is negative.' as reason, equity
      from companies
      where equity < 0
  - id: brief
    uses: brief
    propose: adjust_credit_limit
    question: "{label} ({key}): {reason} Equity {equity}."
  - id: digest
    uses: report
    title: Weekly company watch
"""
TABLES = """
create table companies (orgnr varchar, name varchar, equity decimal(18, 0));
insert into companies values ('100000008', 'DEMO LAKS AS', -4000000);
"""


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A small pack with a warehouse, an action, permissions and one workflow."""
    folder = tmp_path / "pack"
    files = {
        "pack.yaml": "name: demo\ndescription: A demo pack.\nrecipes: [monitor]\n",
        "warehouse/models/models.yml": MODELS,
        "warehouse/metrics.yaml": METRICS,
        "warehouse/ontology.yaml": ONTOLOGY,
        "permissions.yaml": PERMISSIONS,
        "workflows/weekly-watch.yaml": WORKFLOW,
    }
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    data = tmp_path / "data"
    (data / "demo").mkdir(parents=True)
    with duckdb.connect(data / "demo" / "warehouse.duckdb") as con:
        con.execute(TABLES)
    monkeypatch.setenv("VARDEX_DATA_DIR", str(data))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return folder


def use(monkeypatch, *replies):
    fake = FakeModel(*replies)
    monkeypatch.setattr(cli, "connect", lambda settings: fake)
    return fake


def propose() -> Turn:
    return Turn(
        "",
        [
            (
                "adjust_credit_limit",
                {"key": "100000008", "amount": 500000, "reason": "equity is negative"},
            )
        ],
    )


NOTE = "Equity went negative in 2025. I would lower the limit to 500,000 NOK."


def test_a_run_writes_a_report_and_leaves_a_proposal_waiting(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    result = runner.invoke(cli.app, ["run", str(pack)])
    assert result.exit_code == 0, result.output
    report = result.stdout.strip().splitlines()[-1]
    assert report.endswith(".md")
    text = open(report, encoding="utf-8").read()
    assert "Needs a decision (1)" in text
    assert "Finished: **done**" in text
    assert f"vardex approve {pack}" in text
    assert "DEMO LAKS AS" in text
    waiting = runner.invoke(cli.app, ["inbox", str(pack)])
    assert "proposed" in waiting.stdout
    assert "credit limit to 500000" in waiting.stdout


def test_json_gives_a_scheduler_everything_it_needs(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    result = runner.invoke(cli.app, ["run", str(pack), "--json"])
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["ok"] is True
    run = body["data"]["runs"][0]
    assert run["state"] == "done"
    assert run["workflow"] == "weekly-watch"
    assert [step["id"] for step in run["steps"]] == ["risks", "brief", "digest"]
    assert run["cases"] == 1
    assert len(run["proposals"]) == 1
    assert run["cost_usd"] == "0.008"


def test_if_due_runs_the_first_time_and_not_the_second(pack, monkeypatch):
    use(monkeypatch, NOTE, NOTE)
    first = runner.invoke(cli.app, ["run", str(pack), "--if-due", "--json"])
    assert json.loads(first.stdout)["data"]["runs"][0]["trigger"] == "schedule"
    second = runner.invoke(cli.app, ["run", str(pack), "--if-due", "--json"])
    body = json.loads(second.stdout)
    assert body["data"]["runs"] == []
    assert body["data"]["not_due"] == ["weekly-watch"]
    assert second.exit_code == 0  # nothing to do is not a failure


def test_a_named_workflow_that_does_not_exist_lists_the_ones_that_do(pack, monkeypatch):
    result = runner.invoke(cli.app, ["run", str(pack), "monthly-review"])
    assert result.exit_code == 1
    assert "This pack has: weekly-watch" in result.output


def test_schedule_says_what_is_due_and_when_the_next_one_is(pack, monkeypatch):
    use(monkeypatch, NOTE)
    due = runner.invoke(cli.app, ["schedule", str(pack), "--json"])
    assert json.loads(due.stdout)["data"]["workflows"][0]["due_now"] is True
    runner.invoke(cli.app, ["run", str(pack)])
    after = json.loads(runner.invoke(cli.app, ["schedule", str(pack), "--json"]).stdout)
    row = after["data"]["workflows"][0]
    assert row["due_now"] is False
    assert row["last_done"] and row["next_due"]
    assert "every monday at 06:30" in row["schedule"]


def test_the_runs_list_and_one_run_can_be_read_afterwards(pack, monkeypatch):
    use(monkeypatch, NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    listed = runner.invoke(cli.app, ["runs", str(pack)])
    assert "weekly-watch-" in listed.stdout and "done" in listed.stdout
    run_id = json.loads(runner.invoke(cli.app, ["runs", str(pack), "--json"]).stdout)["data"][
        "runs"
    ][0]["id"]
    one = runner.invoke(cli.app, ["runs", str(pack), run_id, "--json"])
    assert json.loads(one.stdout)["data"]["id"] == run_id


def test_approving_carries_the_change_out_and_it_can_be_undone(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    reference = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]["reference"]

    rehearsal = runner.invoke(
        cli.app, ["approve", str(pack), reference, "--by", "joachim", "--dry-run"]
    )
    assert "Dry run" in rehearsal.stdout
    assert (
        json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"]["items"][
            0
        ]["state"]
        == "proposed"
    )

    done = runner.invoke(cli.app, ["approve", str(pack), reference, "--by", "joachim"])
    assert done.exit_code == 0, done.output
    assert "Done:" in done.stdout and "vardex undo" in done.stdout
    assert (
        json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"]["items"][
            0
        ]["state"]
        == "done"
    )

    undone = runner.invoke(cli.app, ["undo", str(pack), reference, "--by", "joachim"])
    assert undone.exit_code == 0, undone.output
    assert (
        json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"]["items"][
            0
        ]["state"]
        == "undone"
    )


def test_only_somebody_the_pack_names_may_approve(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    reference = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]["reference"]
    result = runner.invoke(cli.app, ["approve", str(pack), reference, "--by", "the-agent"])
    assert result.exit_code == 1
    assert "is not an approver" in result.output


def test_an_approval_with_no_name_is_refused(pack, monkeypatch):
    result = runner.invoke(cli.app, ["approve", str(pack), "anything"])
    assert result.exit_code == 1
    assert "Say who is deciding" in result.output


def test_a_pack_that_allows_only_drafting_cannot_have_its_proposal_carried_out(pack, monkeypatch):
    (pack / "permissions.yaml").write_text(
        PERMISSIONS.replace("adjust_credit_limit: approve", "adjust_credit_limit: draft"),
        encoding="utf-8",
    )
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    reference = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]["reference"]
    result = runner.invoke(cli.app, ["approve", str(pack), reference, "--by", "joachim"])
    assert result.exit_code == 1
    assert "an approval cannot carry it out" in result.output


def test_rejecting_keeps_the_reason_and_changes_nothing(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    reference = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]["reference"]
    result = runner.invoke(
        cli.app,
        ["reject", str(pack), reference, "--by", "joachim", "--note", "a filing error"],
    )
    assert result.exit_code == 0
    item = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]
    assert item["state"] == "rejected" and item["note"] == "a filing error"


def test_a_dry_run_leaves_nothing_in_the_inbox(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    result = runner.invoke(cli.app, ["run", str(pack), "--dry-run", "--json"])
    assert json.loads(result.stdout)["data"]["runs"][0]["dry_run"] is True
    assert (
        json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"]["items"]
        == []
    )


def test_two_runs_of_one_workflow_cannot_overlap(pack, monkeypatch):
    from vardex.runs import Lock, runs_path

    use(monkeypatch, NOTE)
    lock = Lock(runs_path(pack.parent / "data", "demo") / "weekly-watch.lock")
    with lock:
        result = runner.invoke(cli.app, ["run", str(pack), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["message"].startswith("weekly-watch is already")


def test_check_catches_a_workflow_that_could_never_work(pack):
    (pack / "workflows" / "weekly-watch.yaml").write_text(
        WORKFLOW.replace("where equity < 0", "where equity < 0 and name in (select 1 from x)"),
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["check", str(pack)])
    assert result.exit_code == 1
    assert "weekly-watch step risks" in result.output


def test_a_stopped_run_is_not_a_success(pack, monkeypatch):
    (pack / "workflows" / "weekly-watch.yaml").write_text(
        WORKFLOW.replace("per_run: 2.00", "per_run: 0.0001"), encoding="utf-8"
    )
    use(monkeypatch, NOTE)
    result = runner.invoke(cli.app, ["run", str(pack), "--json"])
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["ok"] is False
    assert body["error"]["code"] == "workflow_stopped"
    assert body["data"]["runs"][0]["state"] == "stopped"


def test_a_stopped_run_can_be_resumed_and_asks_for_nothing_twice(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    run_id = json.loads(runner.invoke(cli.app, ["runs", str(pack), "--json"]).stdout)["data"][
        "runs"
    ][0]["id"]
    # Resuming a run that finished does nothing twice: every step is already done
    use(monkeypatch, propose(), NOTE)
    again = runner.invoke(cli.app, ["run", str(pack), "--resume", run_id, "--json"])
    assert again.exit_code == 0
    assert (
        len(
            json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
                "items"
            ]
        )
        == 1
    )


# --- recorded and replayed -------------------------------------------------------------


def test_a_recorded_run_replays_with_no_key_no_network_and_no_warehouse(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    recorded = runner.invoke(cli.app, ["run", str(pack), "--record"])
    assert recorded.exit_code == 0, recorded.output
    recording = pack / "workflows" / "recordings" / "weekly-watch.jsonl"
    assert recording.is_file()

    # Nothing live is reachable: no model, no warehouse file, no feed
    monkeypatch.setattr(
        cli, "connect", lambda settings: pytest.fail("a replay must not ask a model")
    )
    monkeypatch.setattr(
        cli,
        "connect_warehouse",
        lambda *a, **k: pytest.fail("a replay must not open a warehouse"),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    again = runner.invoke(cli.app, ["run", str(pack), "--replay", "--json"])
    assert again.exit_code == 0, again.output
    body = json.loads(again.stdout)["data"]["runs"][0]
    assert body["state"] == "done"
    assert body["cases"] == 1


def test_a_replay_with_no_recording_says_how_to_make_one(pack, monkeypatch):
    result = runner.invoke(cli.app, ["run", str(pack), "--replay"])
    assert result.exit_code == 1
    assert "Make one with --record" in result.output


def test_a_recording_that_does_not_cover_what_the_run_asks_for_fails_loudly(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack), "--record"])
    # The question changes, so the recorded answer no longer matches what is asked
    (pack / "workflows" / "weekly-watch.yaml").write_text(
        WORKFLOW.replace("{reason} Equity {equity}.", "{reason} Equity {equity}. Be brief."),
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["run", str(pack), "--replay"])
    assert result.exit_code == 1
    assert "not recorded" in result.output.lower() or "recording" in result.output.lower()


# --- how is it doing? ------------------------------------------------------------------


def test_stats_say_plainly_when_a_workflow_has_stopped_running(pack, monkeypatch):
    # Nothing has ever run: more than one cadence of silence, and it says so
    silent = runner.invoke(cli.app, ["runs", str(pack), "--stats"])
    assert silent.exit_code == 1
    assert "It is not running." in silent.output

    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    now = runner.invoke(cli.app, ["runs", str(pack), "--stats"])
    assert now.exit_code == 0
    assert "last ran within its cadence" in now.output
    assert "1 drafted, 1 waiting" in now.output


def test_stats_as_json_carry_the_numbers_a_dashboard_would_want(pack, monkeypatch):
    use(monkeypatch, propose(), NOTE)
    runner.invoke(cli.app, ["run", str(pack)])
    reference = json.loads(runner.invoke(cli.app, ["inbox", str(pack), "--json"]).stdout)["data"][
        "items"
    ][0]["reference"]
    runner.invoke(cli.app, ["approve", str(pack), reference, "--by", "joachim"])
    body = json.loads(runner.invoke(cli.app, ["runs", str(pack), "--stats", "--json"]).stdout)
    row = body["data"]["workflows"][0]
    assert row["workflow"] == "weekly-watch"
    assert row["quiet"] is False and row["silent_for"] < 1
    assert row["states"] == {"done": 1}
    assert row["median_cost_usd"] == "0.008"
    assert row["proposals"] == {
        "drafted": 1,
        "waiting": 0,
        "done": 1,
        "rejected": 0,
        "expired": 0,
    }
    assert row["agreed"] == 1.0
