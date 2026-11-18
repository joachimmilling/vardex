"""Training data for fine-tuning: SQL pairs on a real DuckDB file, extraction pairs from the
pack's own extraction, and the trainer they are handed to."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime

import duckdb
import pytest
from test_extraction import GOOD, PAGE, SPEC
from test_text_to_sql import KNOWLEDGE, TABLES

from vardex.config import Settings
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.evals import AgentCase, Case
from vardex.lessons import ExtractionLesson, SqlLesson, in_question, variants, written
from vardex.mlx_trainer import MlxTrainer, losses_in
from vardex.semantic import Example, Questions
from vardex.traces import SpanRecord, Trace
from vardex.tracing import EVALUATION, EVALUATION_LABEL, EVALUATION_NAME
from vardex.training import Pair, TrainError, build_dataset, write_dataset

MORE = """
insert into companies values ('100000024', 'NORDLYS KRAFT AS', 'Troms'),
                             ('100000032', 'FJORD FISK AS', 'Vestland');
insert into accounts values ('100000024', 2025, '2025-12-31', 50000000, 1000000),
                            ('100000024', 2024, '2024-12-31', 40000000, 900000);
"""
EXAMPLES = [
    Example(
        question="What was Demo Laks AS's revenue in 2025?",
        sql="select a.revenue from accounts as a join companies as c on c.orgnr = a.orgnr\n"
        "where c.name = 'DEMO LAKS AS' and a.fiscal_year = 2025",
    ),
    Example(
        question="How many companies are there in Vestland?",
        sql="select count(*) as companies from companies where county = 'Vestland'",
    ),
]
KNOWN = replace(
    KNOWLEDGE, questions=Questions(dialect="duckdb", examples=EXAMPLES, show_examples=None)
)
GOLDEN = [
    Case(
        id="oslo",
        type="count",
        question="How many companies are there in Oslo?",
        sql="select count(*) as companies from companies where county = 'Oslo'",
    ),
    AgentCase(id="nordlys", type="lookup", question="What does Nordlys Kraft AS do?"),
]


TODAY = date(2026, 11, 9)


@pytest.fixture
def warehouse(tmp_path):
    with duckdb.connect(tmp_path / "w.duckdb") as con:
        con.execute(TABLES + MORE)
    connection = DuckDBWarehouse(tmp_path / "w.duckdb")
    yield connection
    connection.close()


def test_a_value_is_written_as_the_question_wrote_the_one_it_replaces():
    assert written("AKER BP ASA", "Telenor ASA") == "Aker BP ASA"
    assert written("AKER BP ASA", "TELENOR ASA") == "AKER BP ASA"
    assert written("Møre og Romsdal", "Vestland") == "Møre og Romsdal"
    assert written("company", "group") == "company"
    assert in_question("What was Orkla's revenue?", "ORKLA ASA") == "Orkla"
    assert in_question("Which groups filed?", "group") is None  # a whole word only


def test_variants_swap_what_an_example_names_and_keep_only_queries_that_return_rows(warehouse):
    made = list(variants(EXAMPLES[0], KNOWN, warehouse))
    questions = {pair.input for pair in made}
    assert "What was Nordlys Kraft AS's revenue in 2025?" in questions
    assert "What was Demo Laks AS's revenue in 2024?" not in questions  # no 2024 row: no pair
    assert "What was Fjord Fisk AS's revenue in 2025?" not in questions  # no accounts at all
    assert all(pair.source == "generated" for pair in made)
    counties = {p.input for p in variants(EXAMPLES[1], KNOWN, warehouse)}
    assert counties == {
        "How many companies are there in Oslo?",
        "How many companies are there in Troms?",
    }


def test_no_golden_question_reaches_the_training_data(warehouse):
    """The rule this module exists to keep: the golden set judges the result, so not one of
    its questions, reworded or in another form, and not one of its queries, is trained on."""
    lesson = SqlLesson(KNOWN, warehouse, GOLDEN, TODAY)
    dataset = build_dataset(lesson)
    assert dataset.pairs
    for pair in dataset.pairs:
        assert lesson.why_not(pair) is None, pair.input
    left_out = dict(dataset.left_out)
    # The county swap made the golden question, word for word: it is left out
    assert left_out["How many companies are there in Oslo?"] == "its question is a golden question"
    # So is a question an agent case asks, though it has no query to compare
    assert lesson.why_not(Pair("What does Nordlys Kraft AS do?", "select 1", "approved"))


def test_a_pair_whose_query_is_a_golden_query_is_left_out_whatever_it_asks(warehouse):
    lesson = SqlLesson(KNOWN, warehouse, GOLDEN, TODAY)
    sql = "SELECT count(*) AS companies\nFROM companies WHERE county = 'Oslo'"
    assert lesson.why_not(Pair("Tell me the number of firms in the capital", sql, "approved")) == (
        "its query is a golden reference query"
    )


def test_the_dataset_is_written_as_the_chats_vardex_sends(warehouse, tmp_path):
    lesson = SqlLesson(KNOWN, warehouse, [], TODAY)
    dataset = build_dataset(lesson)
    dataset.pairs.append(
        Pair("How many companies are there?", "select count(*) from companies", "approved")
    )
    train, valid = write_dataset(dataset, lesson, tmp_path)
    assert train + valid == len(dataset.pairs)
    assert valid >= 1
    line = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    system, user, assistant = line["messages"]
    assert system["role"] == "system" and "2026-11-09" in system["content"]
    assert user["role"] == "user"
    plan = json.loads(assistant["content"])
    assert plan["action"] == "query" and plan["sql"]
    assert len((tmp_path / "pairs.jsonl").read_text().splitlines()) == train + valid


def test_variants_of_one_example_stay_on_one_side_of_the_split(warehouse, tmp_path):
    lesson = SqlLesson(KNOWN, warehouse, [], TODAY)
    asked = [f"Question {n}" for n in range(10)]
    dataset = build_dataset(lesson)
    dataset.pairs = [
        Pair(f"{question} {variant}", "select 1", "generated", question)
        for question in asked
        for variant in "abc"
    ] + [Pair(question, "select 1", "example") for question in asked]
    train, valid = write_dataset(dataset, lesson, tmp_path, seed=3)
    assert valid == 4  # one example and its three variants: a tenth of 40
    asked_in = {
        side: {
            json.loads(line)["messages"][1]["content"].split()[1]
            for line in (tmp_path / f"{side}.jsonl").read_text().splitlines()
        }
        for side in ("train", "valid")
    }
    assert not asked_in["train"] & asked_in["valid"]


def extract_trace(document: str, answer: str, passed: bool) -> Trace:
    """A trace of vardex extract as the file exporter writes it: the root, one extract
    attempt with its check, and the model call under it, content kept."""
    at = datetime(2026, 11, 9, tzinfo=UTC)
    verdict = {
        "name": EVALUATION,
        "attributes": {
            EVALUATION_NAME: "extraction_checks",
            EVALUATION_LABEL: "pass" if passed else "fail",
        },
    }
    messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": f"<document>\n{document}\n</document>"}],
        }
    ]
    output = [{"role": "assistant", "parts": [{"type": "text", "content": answer}]}]
    return Trace(
        [
            SpanRecord("t", "root", None, "vardex extract", at, at, "ok",
                       attributes={"vardex.command": "extract"}),
            SpanRecord("t", "attempt", "root", "extract", at, at, "ok", events=[verdict]),
            SpanRecord("t", "call", "attempt", "chat m", at, at, "ok",
                       attributes={"gen_ai.input.messages": json.dumps(messages),
                                   "gen_ai.output.messages": json.dumps(output)}),
        ]
    )  # fmt: skip


def test_an_extraction_trains_on_its_examples_and_approved_documents_never_golden_ones():
    other = (
        "Demo Fisk AS (fictional)\nÅrsregnskap for 2025, selskapet\nBeløp i NOK 1 000\n"
        "Salgsinntekt 1 480 312, sum driftskostnader 1 126 657\nDriftsresultat 353 655\n"
    )
    lesson = ExtractionLesson(SPEC, documents=[PAGE])
    [example] = lesson.worked()  # the extraction's one example, as a complete answer
    assert example.source == "example" and json.loads(example.target)["problem"] is None
    good = json.dumps(GOOD | {"company_name": "Demo Fisk AS"})
    [pair] = lesson.approved(extract_trace(other, good, passed=True))
    assert (pair.input, pair.target, pair.source) == (other, good, "approved")
    assert not list(lesson.approved(extract_trace(other, good, passed=False)))
    # Approved or not, the golden document itself is never trained on
    [golden] = lesson.approved(extract_trace(PAGE, json.dumps(GOOD), passed=True))
    assert lesson.why_not(golden) == "its document is a golden document"
    dataset = build_dataset(lesson, approved_pairs=[pair, golden])
    assert [p.source for p in dataset.pairs] == ["approved", "example"]
    chat = lesson.chat(pair)["messages"]
    assert chat[1]["content"] == f"<document>\n{other}\n</document>"
    assert chat[2]["content"] == good


def test_the_mlx_trainer_trains_lora_on_the_answers_only(monkeypatch, tmp_path):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    trainer = MlxTrainer(Settings(model="m"))
    command = trainer.command("mlx-community/gemma-4-e4b-it-4bit", tmp_path, tmp_path / "a", 300)
    assert command[1:4] == ["-m", "mlx_lm", "lora"]
    assert "--mask-prompt" in command and "--train" in command
    assert command[command.index("--iters") + 1] == "300"


def test_the_mlx_trainer_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(TrainError) as caught:
        MlxTrainer(Settings(model="m"))
    assert "vardex[mlx]" in str(caught.value)


def test_the_mlx_trainer_reads_the_loss_in_either_format():
    assert losses_in("Iter 10: Train loss 0.647, Learning Rate 1.000e-04") == [("Train", 0.647)]
    assert losses_in("Iter 50: Val loss 0.023, Val took 230.4s") == [("Val", 0.023)]
    assert losses_in("    10    0.647 ▼        2      0.6k") == [("Train", 0.647)]
    assert losses_in("\x1b[38;5;244m    50\x1b[0m    \x1b[1;35mval\x1b[0m \x1b[1m0.023\x1b[0m") == [
        ("Val", 0.023)
    ]
    assert losses_in("Trainable parameters: 0.047% (3.473M/7463.013M)") == []
    assert losses_in("  save  0000100_adapters.safetensors") == []


def test_the_mlx_trainer_says_what_to_do_when_the_gpu_runs_out_of_memory(monkeypatch, tmp_path):
    class Process:
        stdout = iter(["     1    val 1.687    214.76s\n", "RuntimeError: Insufficient Memory\n"])

        def wait(self):
            return 1

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: Process())
    with pytest.raises(TrainError) as caught:
        MlxTrainer(Settings(model="m")).train("base", tmp_path, tmp_path / "a", iterations=1)
    assert "iogpu.wired_limit_mb" in str(caught.value)
