"""What people say about answers, and the cases it turns into.

A thumbs down is the cheapest eval there is: somebody who knew the answer read it and said
it was wrong. It is also the least reliable. Most people never press it, the ones who do
are not a fair sample, and "wrong" may mean the number, the wording or the question. So a
verdict never changes the golden set by itself. It makes a *candidate*: the trace's question,
the SQL that ran and the answer given, waiting for a person to say what the right answer is.
Only then does it become a case.

Candidates also come from sampling. Feedback finds the failures people notice; a random
sample of ordinary traffic finds the ones they do not, and shows how questions are really
asked, which no golden set written in advance does.

Both are kept as JSON lines beside the pack's data, appended to and never rewritten, so what
was said, and what was decided about it, can always be read back.
"""

import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vardex.traces import Trace

Verdict = Literal["good", "bad"]
Reason = Literal["flagged", "sampled"]
State = Literal["open", "promoted", "dismissed"]


class FeedbackError(Exception):
    """Raised when a candidate is unknown, ambiguous or already decided."""


def now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Feedback:
    """One person's verdict on one answer."""

    trace_id: str
    verdict: Verdict
    by: str
    note: str = ""
    at: str = ""


@dataclass
class FeedbackLog:
    """Every verdict given on a pack's answers, oldest first."""

    path: Path

    def add(self, feedback: Feedback) -> Feedback:
        feedback = replace(feedback, at=feedback.at or now_text())
        append(self.path, asdict(feedback))
        return feedback

    def all(self) -> list[Feedback]:
        return [Feedback(**line) for line in read_lines(self.path)]

    def verdicts(self) -> dict[str, str]:
        """Each trace's latest verdict: a person who changes their mind is believed."""
        return {f.trace_id: f.verdict for f in self.all()}


@dataclass(frozen=True)
class Candidate:
    """A request that may belong in the golden set, and what a person decided about it."""

    id: str  # the first twelve characters of the trace id
    trace_id: str
    reason: Reason
    question: str
    asked_on: str  # the day it was asked: "last year" meant the year before this one
    sql: str | None  # what Vardex ran, which is not the reference until a person says so
    answer: str | None
    outcome: str | None  # answer, clarify, refuse or error
    note: str = ""
    state: State = "open"
    case: str | None = None  # the case it became
    decided_by: str | None = None
    at: str = ""


@dataclass
class Candidates:
    """A pack's candidate cases. Each change is a new line; the last line for an id wins."""

    path: Path

    def all(self) -> list[Candidate]:
        latest: dict[str, Candidate] = {}
        for line in read_lines(self.path):
            candidate = Candidate(**line)
            latest[candidate.id] = candidate
        return list(latest.values())

    def open(self) -> list[Candidate]:
        return [c for c in self.all() if c.state == "open"]

    def add(self, candidate: Candidate) -> tuple[Candidate, bool]:
        """Add a candidate, unless its trace is already one. Returns it, and whether it is new:
        flagging the same answer twice makes one candidate, not two."""
        for known in self.all():
            if known.trace_id == candidate.trace_id:
                return known, False
        candidate = replace(candidate, at=candidate.at or now_text())
        append(self.path, asdict(candidate))
        return candidate, True

    def get(self, prefix: str) -> Candidate:
        found = [c for c in self.all() if c.id.startswith(prefix)] if prefix else []
        if not found:
            raise FeedbackError(f"no candidate {prefix}")
        if len(found) > 1:
            raise FeedbackError(f"{prefix} names {len(found)} candidates: give more of the id")
        return found[0]

    def decide(
        self, candidate: Candidate, state: State, by: str, note: str = "", case: str | None = None
    ) -> Candidate:
        """Promote or dismiss a candidate, once."""
        if candidate.state != "open":
            raise FeedbackError(f"candidate {candidate.id} is already {candidate.state}")
        decided = replace(
            candidate,
            state=state,
            decided_by=by,
            note=note or candidate.note,
            case=case,
            at=now_text(),
        )
        append(self.path, asdict(decided))
        return decided


def candidate_from(trace: Trace, reason: Reason, note: str = "") -> Candidate:
    """A candidate case from a traced request, which must have a question to ask again."""
    if not trace.question:
        raise FeedbackError(
            f"trace {trace.trace_id[:12]} has no question in it: it is not a question answered, "
            "or its content was not kept"
        )
    return Candidate(
        id=trace.trace_id[:12],
        trace_id=trace.trace_id,
        reason=reason,
        question=trace.question,
        asked_on=trace.start.date().isoformat(),
        sql=trace.sql,
        answer=trace.answer,
        outcome=trace.outcome,
        note=note,
    )


def sample(traces: list[Trace], size: int, *, seed: int | None = None) -> list[Trace]:
    """A random sample of answered questions, for a person to read. Random, not the newest
    or the slowest: a sample chosen by any rule shows what the rule picks, not the traffic."""
    pool = [t for t in traces if t.question and t.outcome is not None]
    return random.Random(seed).sample(pool, min(size, len(pool)))


def feedback_path(data_dir: Path, pack: str) -> Path:
    return data_dir / pack / "feedback.jsonl"


def candidates_path(data_dir: Path, pack: str) -> Path:
    return data_dir / pack / "candidates.jsonl"


def append(path: Path, line: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    lines = []
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(text))
        except ValueError:
            continue  # a line cut short by a crash; the ones before and after are whole
    return lines
