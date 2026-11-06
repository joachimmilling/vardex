"""Watching Vardex through the CLI: one request, one trace; feedback; and the golden set."""

import json
from datetime import UTC, datetime, timedelta

import duckdb
import pytest
import yaml
from fakes import FakeModel, Turn
from typer.testing import CliRunner

from vardex import cli
from vardex.traces import FileTraces, traces

runner = CliRunner()

MODELS = """
models:
  - name: companies
    description: One row per company.
    columns:
      - {name: orgnr, description: Organisation number.}
      - {name: name, description: Registered name.}
      - {name: revenue, description: Revenue in NOK.}
"""
METRICS = """
metrics:
  - {name: revenue, description: Revenue., table: companies, sql: sum(revenue), unit: NOK}
"""
ONTOLOGY = """
objects:
  - {name: Company, description: A company., table: companies, key: orgnr, label: name}
"""
PERMISSIONS = """
budgets:
  per_run: 0.50
tools:
  describe_tool: read
  list_tables: read
  describe_table: read
  run_sql: read
  make_chart: read
"""
SUITE = """
description: Questions about companies.
capability: answer
today: 2026-10-13
threshold: 0.9
cases:
  - id: revenue-of-demo-laks
    type: lookup
    question: "What was Demo Laks's revenue?"
    sql: |
      select revenue from companies where name = 'DEMO LAKS AS'
"""
SQL = "select name, revenue from companies"


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A small pack with a warehouse, a golden set and two fictional companies."""
    folder = tmp_path / "pack"
    files = {
        "pack.yaml": "name: demo\ndescription: A demo pack.\nrecipes: [analyst]\n",
        "warehouse/models/models.yml": MODELS,
        "warehouse/metrics.yaml": METRICS,
        "warehouse/ontology.yaml": ONTOLOGY,
        "permissions.yaml": PERMISSIONS,
        "evals/answer.yaml": SUITE,
    }
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    data = tmp_path / "data"
    (data / "demo").mkdir(parents=True)
    with duckdb.connect(data / "demo" / "warehouse.duckdb") as con:
        con.execute("create table companies (orgnr varchar, name varchar, revenue bigint)")
        con.execute(
            "insert into companies values ('100000008', 'DEMO LAKS AS', 1480312000),"
            " ('100000016', 'DEMO FISK AS', 912000000)"
        )
    monkeypatch.setenv("VARDEX_DATA_DIR", str(data))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return folder


def plan(sql=SQL):
    return json.dumps({"action": "query", "sql": sql, "message": None, "assumptions": []})


def use(monkeypatch, fake):
    monkeypatch.setattr(cli, "connect", lambda settings: fake)


def recorded(tmp_path):
    """Every trace the test's commands wrote."""
    return traces(FileTraces(tmp_path / "traces"), datetime.now(UTC) - timedelta(hours=1))


def test_one_answer_is_one_trace_with_its_attempts_calls_and_queries_inside(
    pack, tmp_path, monkeypatch
):
    # The first plan names a column that is not there; the second passes
    fake = FakeModel(plan("select nme from companies"), plan(), "DEMO LAKS AS: 1,480,312,000.")
    use(monkeypatch, fake)
    result = runner.invoke(cli.app, ["answer", str(pack), "Which revenue is largest?"])
    assert result.exit_code == 0, result.output

    (trace,) = recorded(tmp_path)
    assert trace.name == "vardex answer"
    assert trace.question == "Which revenue is largest?"
    assert trace.outcome == "answer"
    assert trace.sql == SQL
    assert len(trace.calls) == 3
    assert trace.repaired
    assert trace.pack == "demo" and len(trace.pack_version or "") == 12
    tree = [(depth, span.name) for depth, span in trace.walk() if span.name != "duckdb"]
    assert tree == [
        (0, "vardex answer"),
        (1, "write_sql"),
        (2, "chat claude-sonnet-5-5"),
        (1, "write_sql"),
        (2, "chat claude-sonnet-5-5"),
        (1, "answer_from_rows"),
        (2, "chat claude-sonnet-5-5"),
    ]
    first, second = [s for s in trace.spans if s.name == "write_sql"]
    assert first.failed and first.verdicts() == [("sql_checks", False)]
    assert second.verdicts() == [("sql_checks", True)]
    query = next(s for s in trace.children(second) if s.name == "duckdb")
    assert query.get("db.query.text") == SQL and query.get("db.response.returned_rows") == 2
    assert f"trace {trace.trace_id[:12]}" in result.stderr


def test_the_json_envelope_carries_the_trace_id(pack, tmp_path, monkeypatch):
    use(monkeypatch, FakeModel(plan(), "DEMO LAKS AS: 1,480,312,000."))
    result = runner.invoke(cli.app, ["answer", str(pack), "Largest revenue?", "--json"])
    body = json.loads(result.stdout)
    (trace,) = recorded(tmp_path)
    assert body["ok"] and body["trace_id"] == trace.trace_id


def test_tool_calls_run_side_by_side_stay_in_the_runs_trace(pack, tmp_path, monkeypatch):
    fake = FakeModel(
        Turn("Two things at once.", [("list_tables", {}), ("run_sql", {"sql": SQL})]),
        "DEMO LAKS AS had the most revenue.",
    )
    use(monkeypatch, fake)
    result = runner.invoke(cli.app, ["agent", str(pack), "Who had the most revenue?"])
    assert result.exit_code == 0, result.output
    (trace,) = recorded(tmp_path)  # one trace: the calls in threads did not start their own
    agent = next(s for s in trace.spans if s.name == "invoke_agent vardex")
    tools = sorted(s.name for s in trace.children(agent) if s.name.startswith("execute_tool"))
    assert tools == ["execute_tool list_tables", "execute_tool run_sql"]
    assert agent.get("vardex.stop") == "answered"


def test_a_failed_command_is_a_failed_trace_and_a_question_back_is_not(pack, tmp_path, monkeypatch):
    back = json.dumps(
        {"action": "clarify", "sql": None, "message": "Which year?", "assumptions": []}
    )
    use(monkeypatch, FakeModel(back))
    result = runner.invoke(cli.app, ["answer", str(pack), "Revenue?", "--no-input"])
    assert result.exit_code == 2
    use(monkeypatch, FakeModel("not json", "still not json"))
    result = runner.invoke(cli.app, ["answer", str(pack), "Revenue?"])
    assert result.exit_code == 1
    clarified, failed = recorded(tmp_path)
    assert not clarified.failed and clarified.outcome == "clarify"
    assert failed.failed and failed.outcome == "error"


def test_traces_trace_and_dashboard_read_back_what_ran(pack, tmp_path, monkeypatch):
    use(monkeypatch, FakeModel(plan(), "DEMO LAKS AS: 1,480,312,000."))
    runner.invoke(cli.app, ["answer", str(pack), "Largest revenue?"])
    (trace,) = recorded(tmp_path)

    listing = runner.invoke(cli.app, ["traces", str(pack)])
    assert trace.trace_id[:12] in listing.stdout and "Largest revenue?" in listing.stdout

    tree = runner.invoke(cli.app, ["trace", str(pack), trace.trace_id[:8]])
    assert tree.exit_code == 0, tree.output
    assert "Question: Largest revenue?" in tree.stdout
    assert "  write_sql  attempt 1" in tree.stdout and "sql_checks passed" in tree.stdout

    board = runner.invoke(cli.app, ["dashboard", str(pack), "--html"])
    assert board.exit_code == 0, board.output
    assert board.stdout.splitlines()[1].startswith("answer")
    assert (tmp_path / "data" / "demo" / "dashboard.html").is_file()


def test_a_bad_answer_becomes_a_candidate_and_a_case_only_once_a_person_confirms(
    pack, tmp_path, monkeypatch
):
    wrong = "select name, revenue from companies where name = 'DEMO FISK AS'"
    use(monkeypatch, FakeModel(plan(wrong), "DEMO FISK AS: 912,000,000."))
    runner.invoke(cli.app, ["answer", str(pack), "What was Demo Laks's revenue in total?"])
    (trace,) = recorded(tmp_path)

    flagged = runner.invoke(
        cli.app,
        ["feedback", str(pack), trace.trace_id[:12], "--bad", "--note", "wrong company"],
    )
    assert flagged.exit_code == 0, flagged.output
    assert f"Candidate {trace.trace_id[:12]}" in flagged.stdout
    listed = runner.invoke(cli.app, ["candidates", str(pack)])
    assert "wrong company" in listed.stdout

    # Without a keyboard and without --yes, nothing joins the golden set
    refused = runner.invoke(cli.app, ["promote", str(pack), trace.trace_id[:6], "--type", "x"])
    assert refused.exit_code == 1 and "Nothing was added" in refused.output

    right = "select revenue from companies where name = 'DEMO LAKS AS'"
    promoted = runner.invoke(
        cli.app,
        ["promote", str(pack), trace.trace_id[:6], "--type", "lookup", "--sql", right, "--yes"],
    )
    assert promoted.exit_code == 0, promoted.output
    assert "1480312000" in promoted.stdout  # the person saw the rows of their reference
    suite = yaml.safe_load((pack / "evals" / "answer.yaml").read_text())
    case = suite["cases"][-1]
    assert case["question"] == "What was Demo Laks's revenue in total?"
    assert case["sql"].strip() == right

    again = runner.invoke(cli.app, ["promote", str(pack), trace.trace_id[:6], "--type", "x"])
    assert again.exit_code == 1 and "already promoted" in again.output


def test_a_reference_that_does_not_run_is_refused(pack, tmp_path, monkeypatch):
    use(monkeypatch, FakeModel(plan(), "DEMO LAKS AS: 1,480,312,000."))
    runner.invoke(cli.app, ["answer", str(pack), "Largest revenue?"])
    (trace,) = recorded(tmp_path)
    runner.invoke(cli.app, ["feedback", str(pack), trace.trace_id, "--bad"])
    result = runner.invoke(
        cli.app,
        ["promote", str(pack), trace.trace_id[:6], "--type", "x", "--sql", "select nope", "--yes"],
    )
    assert result.exit_code == 1 and "The reference SQL fails" in result.output


def test_a_good_answer_is_counted_and_makes_no_candidate(pack, tmp_path, monkeypatch):
    use(monkeypatch, FakeModel(plan(), "DEMO LAKS AS: 1,480,312,000."))
    runner.invoke(cli.app, ["answer", str(pack), "Largest revenue?"])
    (trace,) = recorded(tmp_path)
    runner.invoke(cli.app, ["feedback", str(pack), trace.trace_id, "--good"])
    assert "No candidates" in runner.invoke(cli.app, ["candidates", str(pack)]).output
    board = runner.invoke(cli.app, ["dashboard", str(pack)])
    assert board.stdout.splitlines()[1].split()[-2:] == ["1", "0"]  # good 1, bad 0


def test_a_sample_and_a_dismissal(pack, tmp_path, monkeypatch):
    for question in ("Largest revenue?", "Smallest revenue?"):
        use(monkeypatch, FakeModel(plan(), "DEMO LAKS AS: 1,480,312,000."))
        runner.invoke(cli.app, ["answer", str(pack), question])
    sampled = runner.invoke(cli.app, ["sample", str(pack), "--size", "5", "--seed", "1"])
    assert "Added 2 candidate(s)" in sampled.stdout
    first = recorded(tmp_path)[0].trace_id[:12]
    dismissed = runner.invoke(
        cli.app, ["dismiss", str(pack), first, "--note", "a duplicate of a case we have"]
    )
    assert dismissed.exit_code == 0, dismissed.output
    listed = runner.invoke(cli.app, ["candidates", str(pack), "--all"])
    assert "[dismissed]" in listed.stdout


def test_the_dashboard_checks_the_packs_limits(pack, tmp_path, monkeypatch):
    (pack / "monitoring.yaml").write_text(
        "min_requests: 1\ncommands:\n  answer:\n    error_rate: 0.0\n"
    )
    use(monkeypatch, FakeModel("not json", "still not json"))
    runner.invoke(cli.app, ["answer", str(pack), "Revenue?"])
    result = runner.invoke(cli.app, ["dashboard", str(pack), "--check"])
    assert result.exit_code == 1
    assert "answer: failed 100% over 1 requests (limit 0%)" in result.stderr
