"""The prompt optimiser, with a fake teacher and a fake student that runs no model."""

import json

import yaml
from fakes import FakeModel
from test_extraction import PACK
from test_text_to_sql import KNOWLEDGE

from vardex.evals import Case, CaseResult, ExtractionCase, Grade, Outcome
from vardex.optimise import ExtractionFile, QuestionsFile, halves, optimise

CASES = [
    Case(id=f"case-{n}", type="lookup", question=f"Question number {n}?", sql="select 1")
    for n in range(12)
]
TEXT = yaml.safe_dump(KNOWLEDGE.questions.model_dump(), sort_keys=False)
QUESTIONS = QuestionsFile(KNOWLEDGE)
SPEC_TEXT = (PACK / "extractions" / "key-figures.yaml").read_text(encoding="utf-8")


def rewrite(glossary: dict[str, str], examples=None) -> str:
    questions = KNOWLEDGE.questions.model_dump()
    questions["glossary"] = {**questions["glossary"], **glossary}
    if examples is not None:
        questions["examples"] = examples
    return json.dumps(
        {
            "diagnosis": "It does not know what a fiscal year is.",
            "changes": ["Say what a fiscal year is."],
            "rewritten": yaml.safe_dump(questions, sort_keys=False),
        }
    )


def student(passes_with_glossary: set[str]):
    """Runs no model: a case passes when the glossary has the word the case needs."""
    runs = []

    def run(knowledge, cases):
        runs.append([case.id for case in cases])
        known = set(knowledge.questions.glossary)
        return [
            CaseResult(
                case,
                Outcome("answer", "", None, []),
                [Grade("rows", case.id not in passes_with_glossary or "fiscal year" in known)],
            )
            for case in cases
        ]

    return run, runs


def test_the_halves_are_the_same_every_time_and_cover_every_case():
    train, held = halves(CASES)
    shuffled, _ = halves(list(reversed(CASES)))
    assert {c.id for c in train} == {c.id for c in shuffled}  # by id, not by place
    assert sorted(c.id for c in train + held) == sorted(c.id for c in CASES)
    assert train and held


def test_a_rewrite_that_helps_on_both_halves_is_confirmed():
    train, held = halves(CASES)
    needs = {train[0].id, held[0].id}  # one case in each half fails without the new entry
    run, runs = student(needs)
    teacher = FakeModel(rewrite({"fiscal year": "The year the accounts end in."}))
    found = optimise(teacher, QUESTIONS, TEXT, CASES, run, rounds=1)
    assert found.confirmed
    assert found.train == (len(train) - 1, len(train))
    assert found.held == (len(held) - 1, len(held))
    assert found.fixed == [held[0].id]
    assert "fiscal year" in found.text
    # The teacher saw the failures of the half it optimises on, and none from the other
    shown = teacher.requests[0].messages[0].text
    assert '<file name="warehouse/questions.yaml">' in shown
    assert train[0].question in shown
    assert held[0].question not in shown


def test_a_rewrite_that_helps_only_where_it_was_tuned_is_not_confirmed():
    train, _ = halves(CASES)
    run, _ = student({train[0].id})  # nothing to fix on the held-out half
    teacher = FakeModel(rewrite({"fiscal year": "The year the accounts end in."}))
    found = optimise(teacher, QUESTIONS, TEXT, CASES, run, rounds=1)
    assert found.train[1] > found.train[0]
    assert not found.confirmed


def test_a_rewrite_that_copies_a_golden_question_is_refused():
    train, _ = halves(CASES)
    run, runs = student({train[0].id})
    copied = [{"question": train[0].question, "sql": "select 1"}]
    teacher = FakeModel(rewrite({"fiscal year": "x"}, examples=copied))
    found = optimise(teacher, QUESTIONS, TEXT, CASES, run, rounds=1)
    assert found.text == TEXT  # nothing kept
    assert len(runs) == 1  # the copy was never even tried
    assert QUESTIONS.copies(QUESTIONS.load(json.loads(rewrite({}, copied))["rewritten"]), CASES)


def test_a_rewrite_that_names_a_table_that_does_not_exist_is_refused():
    bad = [{"question": "How many?", "sql": "select count(*) from nowhere"}]
    train, _ = halves(CASES)
    run, runs = student({train[0].id})
    found = optimise(FakeModel(rewrite({}, bad)), QUESTIONS, TEXT, CASES, run, rounds=1)
    assert found.text == TEXT
    assert len(runs) == 1


def test_an_extraction_is_optimised_the_same_way_on_its_own_file():
    spec = SPEC_TEXT
    documents = {f"page-{n}.txt": f"Demo Fisk {n} AS. Bruttoinntekt {n} 000." for n in range(8)}
    cases = [
        ExtractionCase(
            id=f"page-{n}", type="plain", document=f"page-{n}.txt", values={"revenue": n * 1000}
        )
        for n in range(8)
    ]
    files = ExtractionFile("key-figures", documents)
    train, held = halves(cases)

    def run(version, chosen):  # a case passes when the instructions name Bruttoinntekt
        knows = "Bruttoinntekt" in version.instructions
        return [
            CaseResult(c, Outcome("values", "{}", None, []), [Grade("fields", knows)])
            for c in chosen
        ]

    better = yaml.safe_load(spec)
    better["instructions"] += "\nRevenue may be called Bruttoinntekt."
    answer = {"diagnosis": "", "changes": ["Name Bruttoinntekt."]}
    teacher = FakeModel(json.dumps(answer | {"rewritten": yaml.safe_dump(better)}))
    found = optimise(teacher, files, spec, cases, run, rounds=1)
    assert found.confirmed and found.held == (0, len(held))
    shown = teacher.requests[0].messages[0].text
    assert '<file name="extractions/key-figures.yaml">' in shown
    assert documents[train[0].document] in shown
    # An example whose document is a golden one shows the model the answer: refused
    copied = yaml.safe_load(spec)
    copied["examples"] = [{"document": documents["page-1.txt"], "values": {"problem": "None."}}]
    assert files.copies(files.load(yaml.safe_dump(copied)), cases)


def test_an_example_already_in_the_file_is_not_a_copy_the_teacher_made():
    """A person's example close to a golden question about another county is the pack's own:
    a rewrite that keeps it is tried, and only an example the teacher adds is refused."""
    vestland = Case(
        id="vestland",
        type="count",
        question="How many limited companies are there in Vestland?",
        sql="select 1",
    )
    cases = [*CASES, vestland]
    near = [{"question": "How many limited companies are there in Innlandet?", "sql": "select 1"}]
    questions = KNOWLEDGE.questions.model_dump() | {"examples": near}
    text = yaml.safe_dump(questions, sort_keys=False)
    assert QUESTIONS.copies(QUESTIONS.load(text), cases)  # close enough to count as one
    train, held = halves(cases)
    run, runs = student({train[0].id, held[0].id})
    teacher = FakeModel(rewrite({"fiscal year": "The year the accounts end in."}, examples=near))
    found = optimise(teacher, QUESTIONS, text, cases, run, rounds=1)
    assert len(runs) > 1  # the rewrite was tried, not refused
    assert found.confirmed
