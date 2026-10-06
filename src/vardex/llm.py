"""Talking to a model, without depending on any one provider.

Everything in Vardex that needs a model goes through a ModelClient. Only the module that
implements a client for a provider imports that provider's SDK.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel

from vardex.pricing import Usage
from vardex.prompts import ASK

# How hard the model should think, on Vardex's scale. Each client maps it to its provider's
# setting, or leaves it out where the model has none.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
Role = Literal["user", "assistant"]
# Why a model stopped, in Vardex's own words. Each client maps its provider's reasons onto these.
StopReason = Literal["end", "max_tokens", "tool_use", "refusal", "other"]


class ModelError(Exception):
    """Raised when a model call fails: no key, a refused request or no connection."""


class MissingAPIKeyError(ModelError):
    """Raised when no API key is configured."""


@dataclass(frozen=True)
class Message:
    """One turn of a conversation."""

    role: Role
    text: str


@dataclass(frozen=True)
class Request:
    """Everything a model needs for one call, in a form any provider can handle."""

    system: str
    messages: list[Message]
    max_tokens: int = 4096  # counts thinking as well as the visible answer
    effort: Effort | None = None  # None means the model's default
    output_schema: type[BaseModel] | None = None  # set it to get JSON of this shape back
    cache_system: bool = False  # keep the system prompt in the prompt cache between calls
    cache_conversation: bool = False  # keep everything up to the last message in the cache


@dataclass(frozen=True)
class Answer:
    """A model's answer, and what it took to produce it."""

    text: str
    model: str
    usage: Usage
    thinking_tokens: int | None  # already included in usage.output_tokens
    stop_reason: StopReason
    seconds: float
    cost_usd: Decimal | None  # None when the model is not in the price table
    first_text_seconds: float | None = None  # set when the answer was streamed

    @property
    def truncated(self) -> bool:
        """True when the answer was cut off because it reached max_tokens."""
        return self.stop_reason == "max_tokens"


class ModelClient(Protocol):
    """What Vardex needs from a model provider. Any class with these members will do."""

    model: str

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one request. With `on_text`, pass on each piece of text as it arrives."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count the tokens a text uses when sent as a question."""
        ...


def ask(
    client: ModelClient,
    question: str,
    *,
    effort: Effort | None = None,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Send one question to the model and return its answer with tokens, time and cost."""
    request = Request(system=ASK, messages=[Message("user", question)], effort=effort)
    return client.send(request, on_text=on_text)


class Conversation:
    """A chat with a model: the messages so far, and the answer to each turn.

    Every turn sends the whole conversation, cached so that the next turn reads the earlier
    ones back cheaply. A failed call leaves the conversation as it was, and so does an answer
    without text (a refusal, say), since the API rejects an empty message; its cost still counts.
    """

    def __init__(
        self, client: ModelClient, *, system: str = ASK, effort: Effort | None = None
    ) -> None:
        self.client = client
        self.system = system
        self.effort = effort
        self.messages: list[Message] = []
        self.answers: list[Answer] = []

    def send(self, text: str, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one message with the conversation so far, and keep it and the answer."""
        messages = [*self.messages, Message("user", text)]
        request = Request(
            system=self.system, messages=messages, effort=self.effort, cache_conversation=True
        )
        answer = self.client.send(request, on_text=on_text)
        self.answers.append(answer)
        if answer.text:
            self.messages = [*messages, Message("assistant", answer.text)]
        return answer

    @property
    def cost_usd(self) -> Decimal | None:
        """What every turn cost together, or None when any turn's cost is unknown."""
        costs = [answer.cost_usd for answer in self.answers]
        if None in costs:
            return None
        return sum(costs, Decimal(0))
