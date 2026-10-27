"""Durable runs: the file a run leaves behind, its journal, and the lock around it."""

import json
import os
import time
from datetime import UTC, datetime

import pytest

from vardex.runs import Lock, Run, RunError, RunStore, Step, call_key, new_run_id


def a_run(store: RunStore, workflow: str = "watch", id: str = "", **kwargs) -> Run:
    run = Run(
        id=id or new_run_id(workflow),
        pack="demo",
        workflow=workflow,
        trigger="manual",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        steps=[Step(id="find", uses="find"), Step(id="brief", uses="brief")],
        **kwargs,
    )
    store.save(run)
    return run


def test_a_saved_run_comes_back_exactly_as_it_was(tmp_path):
    store = RunStore(tmp_path / "runs")
    run = a_run(store)
    run.journal["brief:100000008"] = {"steps": 3}
    run.facts["cases"] = [{"key": "100000008", "label": "DEMO LAKS AS"}]
    run.spend(None)  # an unpriced call adds nothing and is not an error
    store.save(run)
    read = store.load(run.id)
    assert read.workflow == "watch"
    assert read.journal == {"brief:100000008": {"steps": 3}}
    assert read.facts["cases"][0]["label"] == "DEMO LAKS AS"
    assert [step.id for step in read.steps] == ["find", "brief"]


def test_money_survives_the_round_trip_as_a_decimal(tmp_path):
    from decimal import Decimal

    store = RunStore(tmp_path / "runs")
    run = a_run(store)
    run.spend(Decimal("0.0123"))
    run.spend(Decimal("0.0001"))
    store.save(run)
    assert store.load(run.id).spent == Decimal("0.0124")


def test_a_half_written_file_is_never_left_behind(tmp_path):
    store = RunStore(tmp_path / "runs")
    run = a_run(store)
    # Written through a temporary file, so the only names in the folder are finished runs
    assert [p.suffix for p in (tmp_path / "runs").iterdir()] == [".json"]
    assert json.loads(store.path(run.id).read_text())["id"] == run.id


def test_a_missing_run_says_so_rather_than_raising_a_file_error(tmp_path):
    with pytest.raises(RunError, match="There is no run nothing-here"):
        RunStore(tmp_path / "runs").load("nothing-here")


def test_a_file_from_a_newer_version_does_not_hide_the_other_runs(tmp_path):
    store = RunStore(tmp_path / "runs")
    first = a_run(store)
    (tmp_path / "runs" / "watch-20991231T000000Z.json").write_text('{"from": "the future"}')
    assert [run.id for run in store.recent()] == [first.id]


def test_runs_are_listed_newest_first_and_can_be_filtered_by_workflow(tmp_path):
    store = RunStore(tmp_path / "runs")
    a_run(store, id="watch-20260101T000000Z")
    a_run(store, id="watch-20260601T000000Z")
    a_run(store, workflow="digest", id="digest-20260301T000000Z")
    assert [run.id for run in store.recent()][0] == "watch-20260601T000000Z"
    assert [run.id for run in store.recent(workflow="digest")] == ["digest-20260301T000000Z"]


def test_only_a_run_that_finished_counts_as_the_last_one(tmp_path):
    store = RunStore(tmp_path / "runs")
    done = a_run(store, id="watch-20260101T000000Z", state="done")
    store.save(done)
    failed = a_run(store, id="watch-20260601T000000Z", state="failed")
    store.save(failed)
    assert store.last(workflow="watch").id == "watch-20260601T000000Z"
    assert store.last_done(workflow="watch").id == "watch-20260101T000000Z"


def test_a_dry_run_never_counts_as_the_last_one(tmp_path):
    store = RunStore(tmp_path / "runs")
    rehearsal = a_run(store, dry_run=True)
    rehearsal.state = "done"
    store.save(rehearsal)
    assert store.last_done(workflow="watch") is None


def test_the_same_call_with_the_same_arguments_has_the_same_key():
    first = call_key("adjust_credit_limit", {"key": "100000008", "amount": 500000})
    again = call_key("adjust_credit_limit", {"amount": 500000, "key": "100000008"})
    other = call_key("adjust_credit_limit", {"key": "100000008", "amount": 400000})
    assert first == again  # the order the model wrote the arguments in cannot matter
    assert first != other
    assert first.startswith("adjust_credit_limit:")


def test_two_runs_of_one_workflow_cannot_overlap(tmp_path):
    path = tmp_path / "watch.lock"
    with Lock(path):
        with pytest.raises(RunError, match="already running"):
            with Lock(path):
                pass


def test_the_lock_is_let_go_however_the_run_ends(tmp_path):
    path = tmp_path / "watch.lock"
    with pytest.raises(ValueError):
        with Lock(path):
            raise ValueError("the step failed")
    assert not path.exists()
    with Lock(path):  # and the next run can take it
        assert path.exists()


def test_a_lock_left_by_a_machine_that_died_is_taken_over_when_it_is_old_enough(tmp_path):
    path = tmp_path / "watch.lock"
    path.write_text(json.dumps({"pid": 999999, "at": time.time() - 10}))
    with pytest.raises(RunError, match="already running"):
        with Lock(path, stale_seconds=60):
            pass
    with Lock(path, stale_seconds=1):  # old enough that whoever held it is not coming back
        assert json.loads(path.read_text())["pid"] == os.getpid()
