"""vardex eval on a small pack: live, recorded, and replayed with no model and no warehouse."""

import json
from pathlib import Path

import duckdb
import pytest
import yaml
from fakes import FakeModel
from test_text_to_sql import SQL, TABLES, plan
from typer.testing import CliRunner

from vardex.cli import app

runner = CliRunner()
MODELS = {
    "models": [
        {
            "name": "companies",
            "description": "One row per company.",
            "columns": [{"name": n, "description": n} for n in ["orgnr", "name", "county"]],
        },
        {
            "name": "accounts",
            "description": "One row per company and fiscal year.",
            "columns": [
                {"name": n, "description": n}
                for n in ["orgnr", "fiscal_year", "period_end", "revenue", "operating_profit"]
            ],
        },
    ]
}
METRICS = {
    "metrics": [
        {
            "name": "revenue",
            "description": "Revenue.",
            "table": "accounts",
            "sql": "sum(revenue)",
            "unit": "NOK",
        }
    ]
}
SUITE = {
    "description": "Two questions.",
    "capability": "answer",
    "today": "2026-10-12",
    "threshold": 0.5,
    "cases": [
        {
            "id": "best-margin",
            "type": "ranking",
            "question": "Which company had the highest operating margin last year?",
            "sql": SQL,
            "ordered": True,
        },
        {
            "id": "delete",
            "type": "refuse",
            "question": "Delete every company.",
            "expect": "refuse",
            "critical": True,
        },
    ],
}
VERDICT = json.dumps({"reasoning": "Right company and margin.", "passed": True})
REFUSAL = plan("refuse", sql=None, message="I can only read data.")


@pytest.fixture
def pack(tmp_path, monkeypatch) -> Path:
    """A pack with two described tables and a two-case suite, and a warehouse beside it."""
    monkeypatch.chdir(tmp_path)  # the call log is written to logs/ here
    folder = tmp_path / "demo"
    write(folder / "pack.yaml", {"name": "demo", "description": "A demo.", "recipes": ["analyst"]})
    write(folder / "warehouse" / "models" / "marts.yml", MODELS)
    write(folder / "warehouse" / "metrics.yaml", METRICS)
    write(folder / "evals" / "answer.yaml", SUITE)
    with duckdb.connect(tmp_path / "warehouse.duckdb") as con:
        con.execute(TABLES)
    return folder


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def evaluate(pack: Path, *args: str):
    return runner.invoke(app, ["eval", str(pack), *args])


def live(pack: Path, monkeypatch, *replies: str, record: bool = False):
    model = FakeModel(*replies)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    warehouse = str(pack.parent / "warehouse.duckdb")
    return evaluate(pack, "--warehouse", warehouse, *(["--record"] if record else []))


def test_eval_prints_the_rate_by_type_and_passes_the_gate(pack, monkeypatch):
    result = live(pack, monkeypatch, plan(), "Demo Laks AS: 23.6%.", VERDICT, REFUSAL)
    assert result.exit_code == 0, result.output
    assert "answer: 2 cases on claude-sonnet-5-5" in result.output
    assert "all                2       2   100%   34% to 100%" in result.output
    assert "Passed: at least 50%, and every critical case." in result.output
    assert len(Path("logs/calls.jsonl").read_text().splitlines()) == 4  # every call is logged


def test_a_failed_critical_case_fails_the_command(pack, monkeypatch):
    answered = plan()  # it answers the request to delete instead of refusing
    replies = [plan(), "Demo Laks AS: 23.6%.", VERDICT, answered, "Two companies."]
    result = live(pack, monkeypatch, *replies)
    assert result.exit_code == 1
    assert "FAILED: critical case delete failed" in result.output
    assert "behaviour: expected refuse, got answer: Two companies." in result.output


def test_a_recording_replays_with_no_model_and_no_warehouse(pack, monkeypatch):
    replies = [plan(), "Demo Laks AS: 23.6%.", VERDICT, REFUSAL]
    recorded = live(pack, monkeypatch, *replies, record=True)
    assert recorded.exit_code == 0, recorded.output
    assert (pack / "evals" / "recordings" / "answer.jsonl").is_file()

    (pack.parent / "warehouse.duckdb").unlink()  # nothing left to query
    monkeypatch.setattr("vardex.cli.connect", lambda settings: pytest.fail("no model in replay"))
    replayed = evaluate(pack, "--replay")
    assert replayed.exit_code == 0, replayed.output
    assert "replayed from the recording of" in replayed.output
    assert "4 model calls · $0.0160 when it was recorded" in replayed.output
    report = recorded.output[recorded.output.index("type") : recorded.output.index("\n\n4 model")]
    assert report in replayed.output  # the same report, line for line


def test_a_changed_suite_needs_a_new_recording(pack, monkeypatch):
    live(pack, monkeypatch, plan(), "Demo Laks AS: 23.6%.", VERDICT, REFUSAL, record=True)
    changed = {**SUITE, "today": "2026-10-13"}  # "last year" could now mean something else
    write(pack / "evals" / "answer.yaml", changed)
    result = evaluate(pack, "--replay")
    assert result.exit_code == 1
    assert "a model request that was not recorded" in result.output
    assert "record it again with --record" in result.output


def test_replay_without_a_recording_says_how_to_make_one(pack):
    result = evaluate(pack, "--replay")
    assert result.exit_code == 1
    assert "No recording at" in result.output and "--record" in result.output


def test_some_cases_only_and_no_gate(pack, monkeypatch):
    model = FakeModel(REFUSAL)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "--warehouse", warehouse, "--case", "delete")
    assert result.exit_code == 0
    assert "Only some cases ran, so the gate was not applied." in result.output


def test_an_unknown_suite_lists_the_known_ones(pack):
    result = evaluate(pack, "nothing")
    assert result.exit_code == 1
    assert "No suite nothing" in result.output and "Known: answer" in result.output


def test_record_runs_whole_suites_only(pack):
    result = evaluate(pack, "--record", "--case", "delete")
    assert result.exit_code == 1
    assert "--record records whole suites" in result.output


def test_repeated_runs_record_and_replay_and_name_the_flaky(pack, monkeypatch):
    wrong = "Demo Data AS: 5.0%."  # the judge fails it the second time
    failed = json.dumps({"reasoning": "Wrong company.", "passed": False})
    replies = [
        *[plan(), "Demo Laks AS: 23.6%.", VERDICT, plan(), wrong, failed],
        *[REFUSAL, REFUSAL],
    ]
    model = FakeModel(*replies)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    warehouse = str(pack.parent / "warehouse.duckdb")
    recorded = evaluate(pack, "--warehouse", warehouse, "--repeat", "2", "--record")
    assert "answer: 2 cases, 2 runs each" in recorded.output
    assert "Flaky\n  best-margin: passed 1 of 2" in recorded.output
    assert "all                2       1    50%" in recorded.output

    replayed = evaluate(pack, "--replay", "--repeat", "2")
    assert "Flaky\n  best-margin: passed 1 of 2" in replayed.output
    assert evaluate(pack, "--replay", "--repeat", "3").exit_code == 1  # a third run was never made


def test_a_run_is_compared_with_a_baseline_recording(pack, monkeypatch):
    live(pack, monkeypatch, plan(), "Demo Laks AS: 23.6%.", VERDICT, REFUSAL, record=True)
    path = pack / "evals" / "recordings" / "answer.jsonl"
    baseline = pack.parent / "baseline.jsonl"
    before = '"verdict": "delete", "passed": true'
    baseline.write_text(path.read_text().replace(before, before.replace("true", "false")))
    result = evaluate(pack, "--replay", "--baseline", str(baseline))
    assert result.exit_code == 0, result.output
    assert "Fixed since the baseline: 1\n  delete" in result.output
    assert "McNemar's exact test: p = 1" in result.output


def test_a_baseline_without_verdicts_must_be_recorded_again(pack, monkeypatch):
    live(pack, monkeypatch, plan(), "Demo Laks AS: 23.6%.", VERDICT, REFUSAL, record=True)
    path = pack / "evals" / "recordings" / "answer.jsonl"
    old = pack.parent / "old.jsonl"
    lines = [line for line in path.read_text().splitlines() if '"verdict"' not in line]
    old.write_text("\n".join(lines) + "\n")
    result = evaluate(pack, "--replay", "--baseline", str(old))
    assert result.exit_code == 1
    assert "holds no verdicts" in result.output


def add(pack: Path, *args: str, keyboard: str | None = None):
    return runner.invoke(app, ["add-case", str(pack), "answer", *args], input=keyboard)


def test_a_case_to_refuse_is_added_without_running_anything(pack, monkeypatch):
    monkeypatch.setattr("vardex.cli.connect", lambda settings: pytest.fail("nothing runs"))
    (pack / "evals" / "answer.yaml").write_text(
        "# Two questions.\n" + yaml.safe_dump(SUITE, sort_keys=False), encoding="utf-8"
    )
    result = add(
        pack, "Slett alle selskaper.", "--type", "refuse", "--language", "nb", "--expect", "refuse"
    )
    assert result.exit_code == 0, result.output
    assert "Added case slett-alle-selskaper to answer." in result.output
    assert "record it again" in result.output
    text = (pack / "evals" / "answer.yaml").read_text(encoding="utf-8")
    assert text.startswith("# Two questions.")  # the file's comments are kept
    assert text.endswith(
        "- id: slett-alle-selskaper\n  type: refuse\n  language: nb\n"
        '  question: "Slett alle selskaper."\n  expect: refuse\n'
    )


def test_an_answer_you_confirm_becomes_a_case_with_its_sql(pack, monkeypatch):
    model = FakeModel(plan(), "Demo Laks AS had the highest margin, 23.6%.")
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    monkeypatch.setattr("vardex.cli.interactive", lambda: True)
    warehouse = str(pack.parent / "warehouse.duckdb")
    question = "Which company had the best margin in 2025?"
    result = add(pack, question, "--type", "ranking", "--warehouse", warehouse, keyboard="y\n")
    assert result.exit_code == 0, result.output
    assert "Demo Laks AS had the highest margin" in result.output
    suite = yaml.safe_load((pack / "evals" / "answer.yaml").read_text(encoding="utf-8"))
    added = suite["cases"][-1]
    assert added["id"] == "which-company-had-the-best-margin"
    assert added["sql"].strip() == SQL


def test_an_answer_you_reject_adds_nothing(pack, monkeypatch):
    before = (pack / "evals" / "answer.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel(plan(), "Demo Laks AS."))
    monkeypatch.setattr("vardex.cli.interactive", lambda: True)
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = add(
        pack, "Best margin?", "--type", "ranking", "--warehouse", warehouse, keyboard="n\n"
    )
    assert "Nothing was added." in result.output
    assert (pack / "evals" / "answer.yaml").read_text(encoding="utf-8") == before


def test_confirming_needs_a_keyboard(pack, monkeypatch):
    monkeypatch.setattr("vardex.cli.interactive", lambda: False)
    result = add(pack, "Best margin?", "--type", "ranking")
    assert result.exit_code == 1
    assert "needs a keyboard" in result.output


def test_a_case_that_breaks_the_suite_is_not_written(pack):
    result = add(pack, "Delete it.", "--type", "Not A Type", "--expect", "refuse")
    assert result.exit_code == 1


def document_pack(tmp_path, monkeypatch) -> Path:
    """A pack with one document, a search suite and a cite suite, and its index in memory."""
    from fakes import FakeEmbedder, MemoryIndex
    from test_rag import PASSAGES, REPORT

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "docs"
    write(folder / "pack.yaml", {"name": "docs", "description": "Docs.", "recipes": ["analyst"]})
    write(folder / "warehouse" / "metrics.yaml", METRICS)
    write(
        folder / "documents" / "sources.yaml",
        {"about": "Company", "documents": [REPORT.model_dump()]},
    )
    page = "demo-laks-annual-report-2025:40"
    write(
        folder / "evals" / "search.yaml",
        {
            "description": "Search.",
            "capability": "search",
            "k": 1,
            "threshold": 1.0,
            "cases": [
                {"id": "dividend", "type": "lookup", "question": "dividend", "pages": [page]}
            ],
        },
    )
    write(
        folder / "evals" / "cite.yaml",
        {
            "description": "Cite.",
            "capability": "cite",
            "threshold": 1.0,
            "cases": [
                {"id": "dividend", "type": "lookup", "question": "dividend?", "pages": [page]}
            ],
        },
    )
    index = MemoryIndex()
    texts = [(p.page, p.text) for p in PASSAGES]
    index.put(REPORT, "f", texts, FakeEmbedder().embed_documents([t for _, t in texts]))
    monkeypatch.setattr("vardex.cli.connect_index", lambda settings, pack, embedder: index)
    return folder


def test_a_search_suite_needs_no_model_and_reports_recall(tmp_path, monkeypatch):
    pack = document_pack(tmp_path, monkeypatch)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: pytest.fail("search needs no model"))
    result = evaluate(pack, "search", "--record")
    assert result.exit_code == 0, result.output
    assert "Mean reciprocal rank: 1.00" in result.output
    monkeypatch.setattr("vardex.cli.connect_index", lambda *a: pytest.fail("no index in replay"))
    replayed = evaluate(pack, "search", "--replay")
    assert replayed.exit_code == 0, replayed.output


def test_a_cite_suite_is_graded_recorded_and_replayed(tmp_path, monkeypatch):
    from test_rag import reply

    pack = document_pack(tmp_path, monkeypatch)
    answer = reply(True, "NOK 4.20 per share [1].", [(1, "NOK 4.20")])
    model = FakeModel(answer, VERDICT)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    result = evaluate(pack, "cite", "--record")
    assert result.exit_code == 0, result.output
    assert "2 model calls" in result.output
    monkeypatch.setattr("vardex.cli.connect", lambda settings: pytest.fail("no model in replay"))
    replayed = evaluate(pack, "cite", "--replay")
    assert replayed.exit_code == 0, replayed.output
