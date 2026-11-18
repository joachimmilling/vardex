"""The commands for other models: routing, shadow mode, labels, the optimiser and training."""

import json
from pathlib import Path

import yaml
from fakes import FakeModel
from test_cli_eval import REFUSAL, VERDICT, evaluate, pack, runner, write  # noqa: F401
from test_extraction import GOOD, PAGE
from test_text_to_sql import plan

from vardex import evals
from vardex.cli import app
from vardex.shadow import ShadowLog, shadow_path

ANSWERED = "Demo Laks AS: 23.6%."
QUESTION = "Which company had the highest operating margin last year?"


def models(monkeypatch, **by_name):
    """Each model by the name the settings give it; anything else is the default."""
    asked = []

    def connect(settings):
        asked.append(settings.model)
        return by_name.get(settings.model) or by_name["default"]

    monkeypatch.setattr("vardex.cli.connect", connect)
    return asked


def answer(pack: Path, *args: str):  # noqa: F811
    warehouse = str(pack.parent / "warehouse.duckdb")
    return runner.invoke(app, ["answer", str(pack), QUESTION, "--warehouse", warehouse, *args])


def test_answer_tries_the_small_model_first(pack, monkeypatch):  # noqa: F811
    small, large = FakeModel(plan(), ANSWERED), FakeModel()
    models(monkeypatch, **{"ollama/gemma4:12b": small, "default": large})
    result = answer(pack, "--small-model", "ollama/gemma4:12b")
    assert result.exit_code == 0, result.output
    assert ANSWERED in result.output
    assert not large.requests


def test_answer_with_a_shadow_keeps_its_answer_and_shows_only_the_live_one(pack, monkeypatch):  # noqa: F811
    live = FakeModel(plan(), ANSWERED)
    shadow = FakeModel(plan(), "Demo Laks AS has the best margin: 23.6%.")
    models(monkeypatch, **{"ollama/gemma4:12b": shadow, "default": live})
    monkeypatch.setenv("VARDEX_DATA_DIR", str(pack.parent / "data"))
    result = answer(pack, "--shadow-model", "ollama/gemma4:12b")
    assert result.exit_code == 0, result.output
    assert "has the best margin" not in result.output
    [entry] = ShadowLog(shadow_path(pack.parent / "data", "demo")).read()
    assert entry.agree
    assert (entry.capability, entry.input) == ("answer", QUESTION)

    shown = runner.invoke(app, ["shadow", str(pack)])
    assert "1 shadowed (answer), 1 alike: 100%" in shown.output


def test_extract_routes_and_shadows_by_the_extractions_own_checks(tmp_path, monkeypatch):
    made_up = json.dumps(GOOD | {"revenue_quote": "1 480 999"})  # not on the page
    small, large = FakeModel(made_up, made_up), FakeModel(json.dumps(GOOD))
    shadow = FakeModel(json.dumps(GOOD | {"accounts": "group"}))
    models(monkeypatch, **{"ollama/small": small, "ollama/shadow": shadow, "default": large})
    monkeypatch.setenv("VARDEX_DATA_DIR", str(tmp_path / "data"))
    page = tmp_path / "laks.txt"
    page.write_text(PAGE, encoding="utf-8")
    real = Path(__file__).parent.parent / "packs" / "norwegian-companies"
    result = runner.invoke(
        app,
        ["extract", str(real), "key-figures", str(page),
         "--small-model", "ollama/small", "--shadow-model", "ollama/shadow"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["accounts"] == "company"  # the large model's values
    [entry] = ShadowLog(shadow_path(tmp_path / "data", "norwegian-companies")).read()
    assert (entry.capability, entry.input, entry.agree) == ("extract", "laks.txt", False)
    assert entry.why == "accounts: 'company' against 'group'"


def test_a_shadow_that_cannot_connect_never_stops_the_answer(pack, monkeypatch):  # noqa: F811
    from vardex.llm import ModelError

    def connect(settings):
        if settings.model == "ollama/gemma4:12b":
            raise ModelError("Could not reach http://localhost:11434/v1", retryable=True)
        return FakeModel(plan(), ANSWERED)

    monkeypatch.setattr("vardex.cli.connect", connect)
    result = answer(pack, "--shadow-model", "ollama/gemma4:12b")
    assert result.exit_code == 0, result.output
    assert ANSWERED in result.output
    assert "no shadow" in result.output


def test_eval_routes_through_the_small_model_and_counts_what_went_on(pack, monkeypatch):  # noqa: F811
    broken = plan(sql="select nonsense from nowhere")
    small = FakeModel(broken, broken, REFUSAL)
    large = FakeModel(plan(), ANSWERED, VERDICT, REFUSAL)
    models(monkeypatch, **{"ollama/gemma4:12b": small, "default": large})
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = evaluate(pack, "--warehouse", warehouse, "--small-model", "ollama/gemma4:12b")
    assert result.exit_code == 0, result.output
    assert "Routed: 2 of 2 went on to the large model" in result.output


def test_label_records_a_persons_verdicts(pack, monkeypatch):  # noqa: F811
    model = FakeModel(plan(), ANSWERED, VERDICT, REFUSAL)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    warehouse = str(pack.parent / "warehouse.duckdb")
    recorded = evaluate(pack, "--warehouse", warehouse, "--record")
    assert recorded.exit_code == 0, recorded.output
    result = runner.invoke(app, ["label", str(pack), "answer", "--by", "joachim"], input="y\n\n")
    assert result.exit_code == 0, result.output
    assert ANSWERED in result.output
    labels = yaml.safe_load((pack / "evals" / "labels" / "answer.yaml").read_text())
    assert labels == [{"case": "best-margin", "answer": ANSWERED, "passed": True, "by": "joachim"}]
    again = runner.invoke(app, ["label", str(pack), "answer"], input="")
    assert ANSWERED not in again.output  # labelled once is enough


def test_label_asks_with_the_recorded_embedder(pack, monkeypatch):  # noqa: F811
    """A pack that chooses its examples by similarity sends a request that depends on the
    embedder; without the recorded one, the replay asks something never recorded."""
    monkeypatch.setattr(
        "vardex.cli.connect", lambda settings: FakeModel(plan(), ANSWERED, VERDICT, REFUSAL)
    )
    warehouse = str(pack.parent / "warehouse.duckdb")
    assert evaluate(pack, "--warehouse", warehouse, "--record").exit_code == 0
    given = []
    real = evals.answer_case

    def answer_case(*args, **kwargs):
        given.append(kwargs.get("embedder"))
        return real(*args, **kwargs)

    monkeypatch.setattr(evals, "answer_case", answer_case)
    result = runner.invoke(app, ["label", str(pack), "answer"], input="q\n")
    assert result.exit_code == 0, result.output
    assert given and all(embedder is not None for embedder in given)


def test_train_builds_a_dataset_with_no_golden_question(pack, monkeypatch):  # noqa: F811
    questions = {
        "dialect": "duckdb",
        "examples": [
            {
                "question": "How many companies are there in Vestland?",
                "sql": "select count(*) as companies from companies where county = 'Vestland'",
            }
        ],
    }
    write(pack / "warehouse" / "questions.yaml", questions)
    write(
        pack / "warehouse" / "ontology.yaml",
        {
            "objects": [
                {
                    "name": "Company",
                    "description": "A company.",
                    "table": "companies",
                    "key": "orgnr",
                    "label": "name",
                    "values": ["county"],
                }
            ]
        },
    )
    suite = yaml.safe_load((pack / "evals" / "answer.yaml").read_text())
    suite["cases"].append(
        {
            "id": "oslo",
            "type": "count",
            "question": "How many companies are there in Oslo?",
            "sql": "select count(*) as companies from companies where county = 'Oslo'",
        }
    )
    write(pack / "evals" / "answer.yaml", suite)
    monkeypatch.setenv("VARDEX_DATA_DIR", str(pack.parent / "data"))
    warehouse = str(pack.parent / "warehouse.duckdb")
    result = runner.invoke(app, ["train", str(pack), "--dataset-only", "--warehouse", warehouse])
    assert result.exit_code == 0, result.output
    assert "1 examples, 0 generated" in result.output  # Oslo is the only other county
    assert "left out: How many companies are there in Oslo?" in result.output
    [folder] = (pack.parent / "data" / "demo" / "training").iterdir()
    pairs = [json.loads(line) for line in (folder / "pairs.jsonl").read_text().splitlines()]
    assert [pair["input"] for pair in pairs] == ["How many companies are there in Vestland?"]


def test_optimise_needs_a_model_to_optimise_for(pack, monkeypatch):  # noqa: F811
    monkeypatch.delenv("VARDEX_SMALL_MODEL", raising=False)
    result = runner.invoke(app, ["optimise", str(pack)])
    assert result.exit_code == 1
    assert "Name the model to optimise for" in result.output


def test_reuse_answers_what_a_recording_holds_and_calls_only_for_the_rest(pack, monkeypatch):  # noqa: F811
    model = FakeModel(plan(), ANSWERED, VERDICT, REFUSAL)
    monkeypatch.setattr("vardex.cli.connect", lambda settings: model)
    warehouse = str(pack.parent / "warehouse.duckdb")
    assert evaluate(pack, "--warehouse", warehouse, "--record").exit_code == 0
    recorded = pack / "evals" / "recordings" / "answer.jsonl"
    nothing = FakeModel()  # it has no replies: any live call would fail the test
    monkeypatch.setattr("vardex.cli.connect", lambda settings: nothing)
    result = evaluate(pack, "--warehouse", warehouse, "--reuse", str(recorded))
    assert result.exit_code == 0, result.output
    assert not nothing.requests
    assert "3 calls answered from earlier recordings" in result.output
    # Recorded too, the count still finds the reusing client inside the recording one
    again = pack / "evals" / "recordings" / "again.jsonl"
    result = evaluate(
        pack, "--warehouse", warehouse, "--reuse", str(recorded), "--recording", str(again),
        "--record",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "3 calls answered from earlier recordings" in result.output
