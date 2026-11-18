"""A judge named per suite, checked against verdicts a person made by hand."""

import json

import duckdb
import pytest
from fakes import FakeModel
from test_cli_eval import REFUSAL, VERDICT, evaluate, pack, write  # noqa: F401
from test_text_to_sql import SQL, TABLES, plan

from vardex import evals, plugins
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.evals import Grade, JudgeCheck, Label, check_judge

ANSWERED = "Demo Laks AS: 23.6%."
WRONG = json.dumps({"reasoning": "It names the wrong company.", "passed": False})


def test_kappa_is_agreement_beyond_chance():
    # A judge that passes everything agrees 90% with a person who passes 90%, by luck alone
    lenient = JudgeCheck(both_pass=9, both_fail=0, false_pass=1, false_fail=0, disagreements=[])
    assert lenient.agreement == 0.9
    assert lenient.kappa == 0.0
    good = JudgeCheck(both_pass=8, both_fail=1, false_pass=0, false_fail=1, disagreements=[])
    assert good.agreement == 0.9
    assert round(good.kappa, 2) == 0.62


def test_the_judge_is_checked_against_each_label(tmp_path):
    with duckdb.connect(tmp_path / "w.duckdb") as con:
        con.execute(TABLES)
    warehouse = DuckDBWarehouse(tmp_path / "w.duckdb")
    suite = evals.Suite.model_validate(
        {
            "description": "x",
            "capability": "answer",
            "today": "2026-10-12",
            "threshold": 0.5,
            "cases": [{"id": "best", "type": "ranking", "question": "Best margin?", "sql": SQL}],
        }
    )
    labels = [
        Label(case="best", answer="Demo Laks AS: 23.6%.", passed=True, by="joachim"),
        Label(case="best", answer="Demo Data AS: 5%.", passed=False, by="joachim"),
    ]
    judge = evals.Judge(FakeModel(VERDICT, VERDICT))  # it passes both
    found = check_judge(judge, suite, labels, warehouse, "duckdb")
    assert (found.both_pass, found.false_pass) == (1, 1)
    assert found.disagreements == [("best", False, "")]
    assert "Cohen's kappa 0.00" in evals.format_judge_check(found)
    warehouse.close()


def test_labels_are_appended_and_read_back(tmp_path):
    evals.add_label(tmp_path, "answer", Label(case="a", answer="Yes: 5.", passed=True, by="j"))
    evals.add_label(tmp_path, "answer", Label(case="b", answer="No.", passed=False, by="j"))
    assert [label.case for label in evals.load_labels(tmp_path, "answer")] == ["a", "b"]


def test_a_suite_names_its_judge_and_is_warned_when_it_judges_itself(pack, monkeypatch):  # noqa: F811
    models = []

    def connect(settings):
        models.append(settings.model)
        return FakeModel(plan(), ANSWERED, VERDICT, REFUSAL)

    monkeypatch.setattr("vardex.cli.connect", connect)
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "--warehouse", warehouse)
    assert "judged by the model under test" in result.output

    suite = pack / "evals" / "answer.yaml"
    suite.write_text(suite.read_text() + "judge: anthropic/claude-opus-5-5\n")
    models.clear()
    replies = iter([FakeModel(plan(), ANSWERED, REFUSAL), FakeModel(VERDICT)])
    monkeypatch.setattr(
        "vardex.cli.connect", lambda settings: models.append(settings.model) or next(replies)
    )
    result = evaluate(pack, "--warehouse", warehouse)
    assert result.exit_code == 0, result.output
    assert "judged by the model under test" not in result.output
    assert models[1] == "anthropic/claude-opus-5-5"


def test_judge_on_the_command_line_wins(pack, monkeypatch):  # noqa: F811
    models = []
    replies = iter([FakeModel(plan(), ANSWERED, REFUSAL), FakeModel(VERDICT)])
    monkeypatch.setattr(
        "vardex.cli.connect", lambda settings: models.append(settings.model) or next(replies)
    )
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "--warehouse", warehouse, "--judge", "ollama/gemma4:12b")
    assert result.exit_code == 0, result.output
    assert models[1] == "ollama/gemma4:12b"


def test_a_judge_that_is_not_a_model_plugs_in_as_a_grader(pack, monkeypatch):  # noqa: F811
    asked = []

    def decision_model(settings, model, capability):
        def grade(case, outcome, reference):
            asked.append((model, capability, outcome.text))
            return Grade("judge", True)

        return grade

    available, load = plugins.available, plugins.load
    monkeypatch.setattr(
        plugins, "available", lambda kind: {"jev": None} if kind == "judges" else available(kind)
    )
    monkeypatch.setattr(
        plugins, "load", lambda kind, name: decision_model if kind == "judges" else load(kind, name)
    )
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel(plan(), ANSWERED, REFUSAL))
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "--warehouse", warehouse, "--judge", "jev/jev-latest")
    assert result.exit_code == 0, result.output
    assert asked == [("jev-latest", "answer", ANSWERED)]


def test_check_judge_reports_agreement_with_the_labels(pack, monkeypatch):  # noqa: F811
    write(
        pack / "evals" / "labels" / "answer.yaml",
        [
            {"case": "best-margin", "answer": ANSWERED, "passed": True, "by": "joachim"},
            {"case": "best-margin", "answer": "Demo Data AS.", "passed": False, "by": "joachim"},
        ],
    )
    monkeypatch.setattr("vardex.cli.connect", lambda settings: FakeModel(VERDICT, WRONG))
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "answer", "--check-judge", "--warehouse", warehouse)
    assert result.exit_code == 0, result.output
    assert "2 labelled answers: the judge agreed on 100%" in result.output
    assert "Cohen's kappa 1.00" in result.output


def test_check_judge_needs_labels(pack):  # noqa: F811
    result = evaluate(pack, "answer", "--check-judge")
    assert result.exit_code == 1
    assert "No labels for answer" in result.output


@pytest.mark.parametrize("reference", ["anthropic/claude-opus-5-5", "claude-opus-5-5"])
def test_a_judge_named_without_a_provider_is_the_default_providers(reference):
    assert plugins.split_model(reference) == ("anthropic", "claude-opus-5-5")


def test_a_case_in_a_language_the_pack_does_not_read_is_refused():
    suite = evals.Suite.model_validate(
        {
            "description": "x",
            "capability": "answer",
            "today": "2026-10-12",
            "threshold": 0.5,
            "cases": [
                {"id": "sv", "type": "count", "language": "sv", "question": "Hur många?",
                 "expect": "refuse"},
            ],
        }
    )  # fmt: skip
    [problem] = evals.check_languages(suite, ["en", "nb"])
    assert "the pack reads no language 'sv'" in problem
