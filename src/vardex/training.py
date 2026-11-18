"""Training data for supervised fine-tuning: pairs of what a model is given and what a right
answer is, for any capability.

A pair is an input (a question, a document), the request Vardex sends with it, word for word,
and the answer a right one is: the query that answers the question, the values the document
holds. How a capability's pairs are made, checked and written is its lesson's business (see
lessons.py); what every lesson shares is here, with the rule they all keep:

Never the golden set. A model trained on the cases it is judged on has learnt the test, and
its score says nothing; so a pair whose input is close to a golden one, or whose answer is a
golden reference answer, is left out, and the tests prove it.

Pairs come from two places, for every capability:

- work a person approved in production (`vardex feedback … --good`) that passed its checks;
- the pack's own worked examples, each written by a person, and for some capabilities pairs
  made from them by code. No model writes them, so no provider's terms on training with its
  output apply, and no model's mistakes are taught as answers.
"""

import json
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vardex.traces import Trace, TraceStore

VALID_SHARE = 0.1  # of the pairs, kept apart to see whether training is still improving


@dataclass(frozen=True)
class Pair:
    """What a model is given, and what it should give back, with where the pair came from."""

    input: str  # the question, the document
    target: str  # the right answer, as the model writes it: a query, the values as JSON
    source: str  # approved, example or generated
    origin: str = ""  # the trace it was approved in, or the example it was generated from


@dataclass
class Dataset:
    """The pairs, and the ones left out, with why."""

    pairs: list[Pair] = field(default_factory=list)
    left_out: list[tuple[str, str]] = field(default_factory=list)  # input, why

    def count(self, source: str) -> int:
        return sum(pair.source == source for pair in self.pairs)


class Lesson(Protocol):
    """How one capability's pairs are made, checked and written."""

    capability: str  # the command whose approved traces it reads: answer, extract

    def worked(self) -> Iterable[Pair]:
        """The pack's worked examples, and any pairs made from them."""
        ...

    def approved(self, trace: Trace) -> Iterable[Pair]:
        """The pairs in a trace a person marked good: the work that passed its checks."""
        ...

    def why_not(self, pair: Pair) -> str | None:
        """Why a pair may not be trained on, the golden set above all, or None if it may."""
        ...

    def chat(self, pair: Pair) -> dict[str, list[dict[str, str]]]:
        """The pair as a chat to train on: the request Vardex sends, and the target."""
        ...


def as_chat(system: str, messages: list[dict[str, str]], target: str) -> dict[str, Any]:
    """A chat in the shape most trainers read. Train on what the model is shown in
    production, word for word, or it learns another task."""
    return {
        "messages": [
            {"role": "system", "content": system},
            *messages,
            {"role": "assistant", "content": target},
        ]
    }


def approved(lesson: Lesson, traces: TraceStore, verdicts: dict[str, str]) -> Iterator[Pair]:
    """The pairs in every trace of the lesson's command that people marked good."""
    for trace_id, verdict in verdicts.items():
        if verdict != "good":
            continue
        trace = traces.trace(trace_id)
        if trace.command == lesson.capability and not trace.failed:
            yield from lesson.approved(trace)


def build_dataset(
    lesson: Lesson,
    *,
    approved_pairs: Iterable[Pair] = (),
    say: Callable[[str], None] = lambda line: None,
) -> Dataset:
    """Every pair there is: approved work, the pack's examples and the pairs made from them,
    less every pair the lesson rules out."""
    dataset = Dataset()
    for pair in [*approved_pairs, *lesson.worked()]:
        why = lesson.why_not(pair)
        if why:
            dataset.left_out.append((" ".join(pair.input.split())[:100], why))
        else:
            dataset.pairs.append(pair)
    say(
        f"{len(dataset.pairs)} pairs: {dataset.count('approved')} approved, "
        f"{dataset.count('example')} examples, {dataset.count('generated')} generated; "
        f"{len(dataset.left_out)} left out"
    )
    return dataset


def write_dataset(
    dataset: Dataset, lesson: Lesson, folder: Path, *, seed: int = 0
) -> tuple[int, int]:
    """Write train.jsonl and valid.jsonl, chat lines, and pairs.jsonl to read them by. About a
    tenth go to valid.jsonl, chosen by the seed. Pairs generated from one example, and the
    example itself, stay on one side: a variant of a question trained on is not a question the
    model has not seen, and a validation loss over such variants only measures recall.
    Returns how many in each."""
    groups: dict[str, list[Pair]] = {}
    for pair in dataset.pairs:
        groups.setdefault(" ".join((pair.origin or pair.input).split()), []).append(pair)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    want = max(1, round(len(dataset.pairs) * VALID_SHARE)) if len(order) > 1 else 0
    valid: list[Pair] = []
    train: list[Pair] = []
    for group in order:
        (valid if len(valid) < want else train).extend(group)
    random.Random(seed).shuffle(train)
    lines = [json.dumps(lesson.chat(pair), ensure_ascii=False) for pair in valid + train]
    cut = len(valid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "valid.jsonl").write_text("".join(f"{x}\n" for x in lines[:cut]), encoding="utf-8")
    (folder / "train.jsonl").write_text("".join(f"{x}\n" for x in lines[cut:]), encoding="utf-8")
    (folder / "pairs.jsonl").write_text(
        "".join(json.dumps(pair.__dict__, ensure_ascii=False) + "\n" for pair in dataset.pairs),
        encoding="utf-8",
    )
    return len(lines) - cut, cut


class TrainError(Exception):
    """Raised when a training run cannot start or does not finish."""


@dataclass(frozen=True)
class Trained:
    """What a training run left: the adapter, the base it belongs to, and its losses."""

    adapter: Path
    base: str
    iterations: int
    train_loss: float | None
    valid_loss: float | None


class Trainer(Protocol):
    """What Vardex needs from a trainer plug-in, which is made with the settings."""

    name: str

    def train(
        self,
        base: str,
        data: Path,
        adapter: Path,
        *,
        iterations: int,
        say: Callable[[str], None] = ...,
    ) -> Trained:
        """Train an adapter for the base model on data/train.jsonl, into `adapter`."""
        ...


def training_path(data_dir: Path, pack: str) -> Path:
    return data_dir / pack / "training"
