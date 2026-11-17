"""Prompt optimisation: a stronger model rewrites a pack's instructions for a weaker one.

A small model fails work a large one does, often for reasons a sentence would fix: it does not
know that "company" means a limited company here, or that an invoice's "Å betale" is the total
due. Finding those sentences by hand is slow, and a person tends to fix the case they last
looked at. This does it as a loop, in the way GEPA and DSPy's optimisers do: run the small
model on half the golden set, show its failures to a stronger model (the teacher), which reads
them and rewrites the file that instructs the capability: questions.yaml for answers, an
extraction's own file for extractions. Run the small model again with the rewrite on a batch of
the same half, and keep the rewrite when more of that batch passes. After the last round, the
best rewrite is run once on the other half, which neither model has seen a failure from. Only a
rewrite that does better there is offered.

The result is a file a person reviews, as a pull request, never a change that ships itself:
a rewrite can pass more cases by teaching the model the golden set's quirks rather than the
pack's meaning, and only a person reading the diff can tell those apart.
"""

import difflib
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field, ValidationError

from vardex.evals import AnyCase, CaseResult, ExtractionCase, mcnemar, words
from vardex.extraction import ExtractionSpec
from vardex.llm import Answer, Message, ModelClient, Request
from vardex.prompts import OPTIMISE
from vardex.semantic import Knowledge, Questions, disagreements
from vardex.tracing import span

# How many cases a candidate is tried on in a round: every failure it is meant to fix, and
# passes up to this many, so a rewrite that breaks what worked is seen too
BATCH = 12
ROUNDS = 3
SIMILAR = 0.9  # an example this close to a golden input is a copy of it, and is refused

# Runs the small model on some cases with one version of the instructions
RunCases = Callable[[Any, list[AnyCase]], list[CaseResult]]


class Rewrite(BaseModel):
    """What the teacher returns: why the student failed, and the file rewritten."""

    diagnosis: str = Field(description="What the failures have in common, in a few sentences.")
    changes: list[str] = Field(description="Each change made, in one sentence each.")
    rewritten: str = Field(description="The whole file, rewritten.")


class Instructions(Protocol):
    """The file that instructs one capability, as the optimiser rewrites it."""

    name: str  # its path in the pack, as the teacher is shown it
    rules: str  # what else a rewrite of this file must respect, as lines of the teacher's rules

    def load(self, text: str) -> Any:
        """The version of the instructions the cases run with, from the file's text; a
        ValueError when the text is not a valid file of its kind."""
        ...

    def copies(self, version: Any, cases: list[AnyCase]) -> list[str]:
        """The examples in a version that copy a golden case's input, or nearly."""
        ...

    def failure(self, result: CaseResult) -> str:
        """One failure, as the teacher reads it."""
        ...


def close(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, words(a), words(b)).ratio() >= SIMILAR


@dataclass
class QuestionsFile:
    """questions.yaml, which instructs text-to-SQL: its glossary, style rules and examples."""

    knowledge: Knowledge
    name = "warehouse/questions.yaml"
    rules = (
        "- Write SQL in the file's own dialect, against the tables its examples already use.\n"
        "- Keep the keys dialect, glossary, style, show_examples, choose_examples and examples."
    )

    def load(self, text: str) -> Knowledge:
        """The knowledge with questions.yaml replaced by a rewrite, which must be valid and
        agree with the rest of the semantic layer."""
        try:
            questions = Questions.model_validate(yaml.safe_load(text) or {})
        except (yaml.YAMLError, ValidationError) as err:
            raise ValueError(f"the rewrite is not a valid questions.yaml: {err}") from err
        rewritten = replace(self.knowledge, questions=questions)
        problems = disagreements(rewritten)
        if problems:
            raise ValueError(
                "the rewrite does not agree with the semantic layer: " + "; ".join(problems)
            )
        return rewritten

    def copies(self, version: Knowledge, cases: list[AnyCase]) -> list[str]:
        """An optimiser that sees failures will happily add the failing question, with its
        answer, as an example: that fixes the case and measures nothing."""
        golden = [case.question for case in cases if hasattr(case, "question")]
        return [
            example.question
            for example in version.questions.examples
            if any(close(example.question, question) for question in golden)
        ]

    def failure(self, result: CaseResult) -> str:
        """The question, what the student ran, what the reference ran, and why it failed."""
        case = result.case
        ran = result.outcome.result.sql if result.outcome.result else "(no query ran)"
        expected = getattr(case, "sql", None) or f"(none: {case.expect})"
        return (
            f"<input>{getattr(case, 'question', '')}</input>\n"
            f"<produced>\n{ran}\n</produced>\n<expected>\n{expected}\n</expected>"
        )


@dataclass
class ExtractionFile:
    """An extraction's own file, extractions/<name>.yaml: its instructions, fields and
    examples. Its documents are read from the suite's folder, to show the teacher."""

    extraction: str
    documents: dict[str, str]  # each case's document, by the file name the case gives
    rules = "- Keep every field, its type and its choices: the cases and the code depend on them."

    @property
    def name(self) -> str:
        return f"extractions/{self.extraction}.yaml"

    def load(self, text: str) -> ExtractionSpec:
        """The extraction, rewritten. Its examples must still pass their own checks."""
        try:
            return ExtractionSpec.model_validate(yaml.safe_load(text) or {})
        except (yaml.YAMLError, ValidationError) as err:
            raise ValueError(f"the rewrite is not a valid extraction: {err}") from err

    def copies(self, version: ExtractionSpec, cases: list[AnyCase]) -> list[str]:
        """An example whose document is a golden document: the model is shown the answer."""
        golden = [
            self.documents[case.document] for case in cases if isinstance(case, ExtractionCase)
        ]
        return [
            " ".join(example.document.split())[:60]
            for example in version.examples
            if any(close(example.document, document) for document in golden)
        ]

    def failure(self, result: CaseResult) -> str:
        """The document, the values the student gave, the values expected, and why."""
        case = result.case
        assert isinstance(case, ExtractionCase)
        expected = case.values if case.expect == "values" else f"(none: {case.expect})"
        return (
            f"<input>\n{self.documents[case.document]}\n</input>\n"
            f"<produced>\n{result.outcome.text}\n</produced>\n"
            f"<expected>\n{json.dumps(expected, default=str, ensure_ascii=False)}\n</expected>"
        )


def halves(cases: list[AnyCase]) -> tuple[list[AnyCase], list[AnyCase]]:
    """The cases to optimise on, and the ones held out to confirm on, split by a hash of each
    case's id: the same split every run, and a new case lands in a half by its own id."""
    train, held = [], []
    for case in cases:
        digit = hashlib.sha256(case.id.encode()).digest()[0]
        (train if digit % 2 == 0 else held).append(case)
    return train, held


@dataclass
class Round:
    number: int
    passed: int  # of the batch it was tried on
    out_of: int
    baseline: int  # what the best version so far passed of the same batch
    kept: bool
    changes: list[str]


@dataclass
class Optimised:
    """What a run of the optimiser found."""

    text: str  # the best version of the file: the original when no rewrite was kept
    train: tuple[int, int]  # passed, of the cases optimised on, before and after
    held: tuple[int, int]  # the same on the held-out half
    held_cases: int
    p_value: float  # McNemar's exact test, on the held-out half
    rounds: list[Round] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)  # held-out cases the rewrite broke
    fixed: list[str] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)  # the teacher's calls

    @property
    def confirmed(self) -> bool:
        """True when the rewrite did better on cases it never saw a failure from."""
        return self.held[1] > self.held[0]


def failures_text(instructions: Instructions, results: list[CaseResult]) -> str:
    """The failures as the teacher reads them: each in the instructions' own terms, with why
    the graders failed it."""
    parts = []
    for result in results:
        if result.passed:
            continue
        reasons = "; ".join(g.reason for g in result.grades if not g.passed)
        parts.append(f"<failure>\n{instructions.failure(result)}\n<why>{reasons}</why>\n</failure>")
    return "\n".join(parts)


def propose(
    teacher: ModelClient, instructions: Instructions, text: str, results: list[CaseResult]
) -> tuple[Rewrite, Answer]:
    """Ask the teacher for a rewrite of the instructions that fixes these failures."""
    shown = f'<file name="{instructions.name}">\n{text}\n</file>'
    request = Request(
        system=OPTIMISE.format(rules=instructions.rules),
        messages=[Message("user", f"{shown}\n\n{failures_text(instructions, results)}")],
        max_tokens=16000,
        output_schema=Rewrite,
    )
    answer = teacher.send(request)
    return Rewrite.model_validate_json(answer.text), answer


def batch(results: list[CaseResult], size: int, seed: int) -> list[AnyCase]:
    """Every failure, then passes up to the size, chosen by the seed."""
    failed = [r.case for r in results if not r.passed]
    passed = [r.case for r in results if r.passed]
    room = max(0, size - len(failed))
    return failed + random.Random(seed).sample(passed, min(room, len(passed)))


def optimise(
    teacher: ModelClient,
    instructions: Instructions,
    text: str,
    cases: list[AnyCase],
    run_cases: RunCases,
    *,
    rounds: int = ROUNDS,
    size: int = BATCH,
    say: Callable[[str], None] = lambda line: None,
) -> Optimised:
    """Look for a rewrite of the instructions that the small model does better with.

    `run_cases` runs the small model, and grades it, on some cases with one version of the
    instructions; the teacher only ever sees failures from the first half of `cases`."""
    train, held = halves(cases)
    say(f"Optimising on {len(train)} cases, holding out {len(held)}")
    with span("optimise", {"vardex.rounds": rounds}) as traced:
        original = instructions.load(text)
        already = set(instructions.copies(original, cases))
        best_text, best = text, original
        results = run_cases(original, train)
        start = sum(r.passed for r in results)
        say(f"Before: {start} of {len(train)}")
        log: list[Round] = []
        spent: list[Answer] = []
        for number in range(1, rounds + 1):
            if all(r.passed for r in results):
                break
            rewrite, answer = propose(teacher, instructions, best_text, results)
            spent.append(answer)
            try:
                candidate = instructions.load(rewrite.rewritten)
            except ValueError as err:
                say(f"Round {number}: refused, {err}")
                continue
            # Only what the teacher added: an example a person wrote, already in the file, is
            # the pack's own, however close it comes to a golden question about another county
            copied = [c for c in instructions.copies(candidate, cases) if c not in already]
            if copied:
                say(f"Round {number}: refused, it copies golden questions: {'; '.join(copied)}")
                continue
            tried = batch(results, size, number)
            now = run_cases(candidate, tried)
            before = {r.case.id: r.passed for r in results}
            passed = sum(r.passed for r in now)
            baseline = sum(before[case.id] for case in tried)
            kept = passed > baseline
            log.append(Round(number, passed, len(tried), baseline, kept, rewrite.changes))
            say(f"Round {number}: {passed} of {len(tried)}, against {baseline}; "
                f"{'kept' if kept else 'dropped'}")  # fmt: skip
            if kept:
                # The batch's results replace the old ones; the rest of the half is not run
                # again, which keeps a round to one batch of calls, as GEPA's minibatches do
                best_text, best = rewrite.rewritten, candidate
                again = {r.case.id: r for r in now}
                results = [again.get(r.case.id, r) for r in results]
        after = sum(r.passed for r in results)
        if best_text == text:
            say("No rewrite was kept: the held-out half is not run.")
            return Optimised(text, (start, after), (0, 0), len(held), 1.0, log, answers=spent)
        old = {r.case.id: r.passed for r in run_cases(original, held)}
        new = {r.case.id: r.passed for r in run_cases(best, held)}
        broken = [case for case in new if old[case] and not new[case]]
        fixed = [case for case in new if new[case] and not old[case]]
        held_before, held_after = sum(old.values()), sum(new.values())
        traced.update({"vardex.held.before": held_before, "vardex.held.after": held_after})
        say(f"Held out: {held_before} of {len(held)} before, {held_after} after")
    return Optimised(
        best_text,
        (start, after),
        (held_before, held_after),
        len(held),
        mcnemar(len(broken), len(fixed)),
        log,
        broken,
        fixed,
        spent,
    )
