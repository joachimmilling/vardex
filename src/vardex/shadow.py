"""Shadow mode: a candidate does the same work as the live model, and nobody sees it.

A golden set says how a candidate does on the cases someone thought to write down.
Production brings the others. Shadow mode gives each real piece of work to the candidate as
well, after the live result has been given: the same question, the same invoice. It keeps both
side by side: what each did, what each produced, what each cost and how long each took. The
candidate's work is never shown and never acted on, so a candidate that is worse costs some
compute and nothing else.

Agreement is not correctness. Two models that agree may both be wrong, and the live model is
not the reference. What shadow mode finds is the work on which they differ: that is what a
person reads, and what is worth turning into cases.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from vardex.capabilities import UNREACHABLE, Attempt, Capability
from vardex.evals import median, wilson
from vardex.llm import ModelClient, ModelError
from vardex.tracing import span


@dataclass(frozen=True)
class Side:
    """What one model did with the work."""

    model: str
    kind: str  # the capability's own kinds, or failed, or unreachable
    text: str
    detail: str  # the output in a few words: "12 rows", "7 of 9 fields"
    seconds: float
    cost_usd: str | None  # as text, so JSON keeps it exact; None when a call had no price


@dataclass(frozen=True)
class Shadowed:
    """One piece of work, done by the live model and by the candidate."""

    at: str
    trace_id: str
    capability: str  # answer, extract, …
    input: str  # what the work was: the question, the document's name
    live: Side
    shadow: Side
    agree: bool
    why: str  # how they differ, or empty when they agree


def side(attempt: Attempt) -> Side:
    cost = attempt.cost_usd
    return Side(
        model=attempt.model,
        kind=attempt.kind,
        text=attempt.text,
        detail=attempt.detail,
        seconds=attempt.seconds,
        cost_usd=None if cost is None else str(cost),
    )


def run_shadow(
    candidate: ModelClient, work: Capability, live: Attempt, *, input: str, trace_id: str
) -> Shadowed:
    """Do the work the live model did with the candidate, and compare the two.

    It runs after the live result has been given, in a span of its own, and whatever goes
    wrong with the candidate is recorded, never raised: a failing candidate must not fail the
    request it shadows."""
    with span("shadow", {"vardex.shadow.model": candidate.model}) as shadowed:
        shadowed.set("vardex.capability", work.name)
        try:
            attempt = work.attempt(candidate)
        except ModelError as err:  # a missing key, a refused request: the candidate's, not ours
            attempt = Attempt(candidate.model, UNREACHABLE, str(err), None, "", error=err)
        why = work.differ(live, attempt)
        shadowed.update({"vardex.shadow.agree": not why, "vardex.shadow.why": why})
    return Shadowed(
        at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_id=trace_id,
        capability=work.name,
        input=input,
        live=side(live),
        shadow=side(attempt),
        agree=not why,
        why=why,
    )


def shadow_path(data_dir: Path, pack: str) -> Path:
    return data_dir / pack / "shadow.jsonl"


class ShadowLog:
    """Every shadowed question, one JSON line each, appended and never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def add(self, entry: Shadowed) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def read(
        self,
        since: datetime | None = None,
        model: str | None = None,
        capability: str | None = None,
    ) -> list[Shadowed]:
        if not self.path.is_file():
            return []
        found = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            entry = Shadowed(
                **{**data, "live": Side(**data["live"]), "shadow": Side(**data["shadow"])}
            )
            if since and datetime.fromisoformat(entry.at) < since:
                continue
            if model and entry.shadow.model != model:
                continue
            if capability and entry.capability != capability:
                continue
            found.append(entry)
        return found


def summary(entries: list[Shadowed]) -> list[str]:
    """What the shadowed work shows, as lines a person reads: how often the two agree, with
    the uncertainty a small sample brings; what each did; what each cost and took."""
    if not entries:
        return ["Nothing shadowed yet."]
    agreed = sum(entry.agree for entry in entries)
    low, high = wilson(agreed, len(entries))
    capabilities = ", ".join(sorted({entry.capability for entry in entries}))
    lines = [
        f"{len(entries)} shadowed ({capabilities}), {agreed} alike: "
        f"{agreed / len(entries):.0%} (95% interval {low:.0%} to {high:.0%})",
        "",
    ]
    for name in ("live", "shadow"):
        sides = [getattr(entry, name) for entry in entries]
        kinds = Counter(s.kind for s in sides)
        costs = [Decimal(s.cost_usd) for s in sides if s.cost_usd is not None]
        priced = f"${sum(costs, Decimal(0)):.4f}" if len(costs) == len(sides) else "unpriced"
        did = ", ".join(f"{kind} {count}" for kind, count in kinds.most_common())
        lines.append(
            f"{name:<7}{sides[-1].model[:32]:<34}median {median([s.seconds for s in sides]):.1f} s"
            f", {priced}"
        )
        lines.append(f"{'':<7}{did}")
    differ = [entry for entry in entries if not entry.agree]
    if differ:
        lines += ["", "Where they differ"]
    for entry in differ:
        lines.append(f"  {entry.trace_id[:12]}  {' '.join(entry.input.split())[:70]}")
        lines.append(f"    {entry.why}")
    return lines
