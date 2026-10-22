"""Hard spending caps, enforced where the money is spent.

A budget is not advice. Providers offer a token budget the model can see and pace itself
against, which is useful and is not a cap: the model decides whether to obey it. A cap has to
be enforced by whatever holds the API key, before the call goes out.

So `Budget` is a ModelClient that wraps another one. Everything in Vardex that talks to a
model goes through a ModelClient, so wrapping one puts the cap under every agent, every
channel and every prompt: there is no path to the provider that does not pass through here.

Before each call it prices the worst that call could cost — every input token, and `max_tokens`
of output, which is the most the provider will generate — and refuses if that would take the
total past a cap. Pricing the worst case is what makes the cap hard: priced on the average, a
single long answer could step over it.

The day's spending is kept in a file beside the pack's data, so a cap per day holds across
commands, and across the CLI, the MCP server and whatever comes next.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from vardex.llm import Answer, ModelClient, Request
from vardex.permissions import Budgets
from vardex.pricing import MILLION, UnknownModelError, price_for

RESERVE = Decimal("0.0001")  # a call priced below this is never worth refusing
# How many characters a token is worth, when guessing high on purpose. A token is about four
# characters of English or Norwegian prose, and fewer in SQL, numbers and Nynorsk; two keeps
# the guess above the truth, which is the only safe direction for a cap. The guess is made
# here rather than asked of the provider so that the cap costs nothing, adds no round trip,
# and still holds when the provider's token counter is down. `vardex tokens` counts exactly.
CHARS_PER_TOKEN = 2
IMAGE_TOKENS = 1_600  # a full-page picture, which is the largest Vardex sends


class BudgetError(Exception):
    """Raised when a call would take spending past a cap. The run stops; what it has already
    found is kept, and what it has already spent is still spent."""


@dataclass
class Spend:
    """What a pack has spent on models today, in a file it shares with every other channel."""

    path: Path
    today: date = field(default_factory=date.today)

    def read(self) -> Decimal:
        """Today's total. A missing or unreadable file counts as nothing spent, because
        refusing to answer when the record cannot be read fails in the wrong direction."""
        try:
            days = json.loads(self.path.read_text(encoding="utf-8"))
            return Decimal(str(days.get(self.today.isoformat(), "0")))
        except (OSError, ValueError):
            return Decimal(0)

    def add(self, amount: Decimal) -> Decimal:
        """Add to today's total and return it. Days older than 30 are dropped."""
        try:
            days = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            days = {}
        total = Decimal(str(days.get(self.today.isoformat(), "0"))) + amount
        days[self.today.isoformat()] = str(total)
        keep = sorted(days)[-30:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole, through a temporary file, so a crash cannot leave half a record
        temporary = self.path.with_suffix(".writing")
        temporary.write_text(json.dumps({d: days[d] for d in keep}, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return total


def spend_path(data_dir: Path, pack: str) -> Path:
    """Where a pack's record of what it has spent lives."""
    return data_dir / pack / "spend.json"


def estimate_tokens(request: Request) -> int:
    """A deliberate overestimate of a request's input tokens, made without asking anyone."""
    characters = len(request.system) + len(request.system_tail)
    characters += sum(len(tool.name) + len(tool.description) for tool in request.tools)
    characters += sum(
        len(message.text) + sum(len(reply.content) for reply in message.tool_replies)
        for message in request.messages
    )
    images = sum(len(message.images) for message in request.messages)
    return characters // CHARS_PER_TOKEN + images * IMAGE_TOKENS


def worst_case_usd(model: str, input_tokens: int, max_tokens: int) -> Decimal | None:
    """The most one call could cost: every input token at the input price, and the whole of
    max_tokens at the output price. None when the model has no price."""
    try:
        price = price_for(model)
    except UnknownModelError:
        return None
    return (input_tokens * price.input + max_tokens * price.output) / MILLION


@dataclass
class Budget:
    """A ModelClient that refuses a call it cannot pay for, and records what it did pay."""

    client: ModelClient
    caps: Budgets = field(default_factory=Budgets)
    spend: Spend | None = None  # None keeps no record, so there can be no cap per day
    run_usd: Decimal = Decimal(0)  # spent since this Budget was made: one question, one run
    day_usd: Decimal = Decimal(0)  # today's total, read once and kept up to date
    answers: list[Answer] = field(default_factory=list)
    unpriced: int = 0  # calls whose model has no price, so they could not be counted

    def __post_init__(self) -> None:
        if self.spend is not None:
            self.day_usd = self.spend.read()

    @property
    def model(self) -> str:
        return self.client.model

    def remaining(self) -> Decimal | None:
        """What is left before the first cap is reached, or None when there is no cap."""
        left = [
            cap - spent
            for cap, spent in ((self.caps.per_run, self.run_usd), (self.caps.per_day, self.day_usd))
            if cap is not None
        ]
        return min(left) if left else None

    def check(self, request: Request) -> None:
        """Raise BudgetError if this request could take spending past a cap."""
        left = self.remaining()
        if left is None:
            return
        worst = worst_case_usd(self.model, estimate_tokens(request), request.max_tokens)
        if worst is None or worst <= RESERVE:
            return
        if worst > left:
            which = "this question" if self.caps.per_run == left + self.run_usd else "today"
            raise BudgetError(
                f"The budget for {which} is spent: ${left:.4f} left, and this call could cost "
                f"up to ${worst:.4f}. Raise the cap in permissions.yaml, or ask something "
                "that takes fewer steps."
            )

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        self.check(request)
        answer = self.client.send(request, on_text)
        self.answers.append(answer)
        if answer.cost_usd is None:
            self.unpriced += 1
        else:
            self.run_usd += answer.cost_usd
            self.day_usd = (
                self.spend.add(answer.cost_usd)
                if self.spend is not None
                else self.day_usd + answer.cost_usd
            )
        return answer

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)
