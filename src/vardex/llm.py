"""Talking to a model, without depending on any one provider.

Everything in Vardex that needs a model goes through a ModelClient. Only the module that
implements a client for a provider imports that provider's SDK.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol

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
    """Raised when a model call fails: no key, a refused request or no connection.

    `retryable` is True when the same request may well succeed later: the provider was
    overloaded, limited the rate or could not be reached. False means it will fail again.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class MissingAPIKeyError(ModelError):
    """Raised when no API key is configured."""


@dataclass(frozen=True)
class Image:
    """A picture sent with a message, such as a scanned page."""

    data: bytes
    media_type: Literal["image/png", "image/jpeg"] = "image/png"


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model sees it: a name, what it is for, and the shape of its arguments.

    The description is the whole instruction manual the model gets. It never sees the code.
    """

    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema for the arguments, an object with properties


@dataclass(frozen=True)
class ToolCall:
    """A model asking for one tool to be run, with the arguments it chose."""

    id: str  # the provider's id for this call; the result must carry it back
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolReply:
    """What running one tool gave back, on its way to the model."""

    id: str  # the call it answers
    content: str
    is_error: bool = False  # a failure the model can learn from and work around


@dataclass(frozen=True)
class Reasoning:
    """A model's own reasoning from a turn, kept so it can be sent back untouched.

    Providers sign these blocks and reject an edited one, so Vardex stores them opaquely and
    never reads, trims or rewrites them. Dropping them from a turn that also called a tool is
    the classic way to break an agent loop: the provider refuses the next call.
    """

    text: str  # a summary, or empty when the provider returns none
    signature: str = ""
    data: str = ""  # set instead of text when the provider encrypted the block


@dataclass(frozen=True)
class Message:
    """One turn of a conversation: text, and any pictures, tool calls or results with it."""

    role: Role
    text: str
    images: tuple[Image, ...] = ()  # shown to the model before the text
    # An assistant turn carries back the reasoning and the calls it made, in that order; a
    # user turn carries the replies to those calls, all of them in one message.
    reasoning: tuple[Reasoning, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_replies: tuple[ToolReply, ...] = ()


@dataclass(frozen=True)
class Request:
    """Everything a model needs for one call, in a form any provider can handle."""

    system: str
    messages: list[Message]
    max_tokens: int = 4096  # counts thinking as well as the visible answer
    effort: Effort | None = None  # None means the model's default
    output_schema: type[BaseModel] | None = None  # set it to get JSON of this shape back
    cache_system: bool = False  # keep the system prompt in the prompt cache between calls
    # More of the system prompt that changes from call to call, such as examples chosen for
    # this question: it comes after the cached part, so the cached part stays the same
    system_tail: str = ""
    cache_conversation: bool = False  # keep everything up to the last message in the cache
    # The tools the model may call this turn. They are sent before the system prompt, so a
    # list that changes between calls costs the whole prompt cache: build it once per run.
    tools: tuple[ToolSpec, ...] = ()


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
    tool_calls: tuple[ToolCall, ...] = ()  # set when stop_reason is tool_use
    reasoning: tuple[Reasoning, ...] = ()  # to send back with the turn, untouched

    @property
    def truncated(self) -> bool:
        """True when the answer was cut off because it reached max_tokens."""
        return self.stop_reason == "max_tokens"

    def as_message(self) -> Message:
        """This answer as the assistant turn to send back: reasoning, text and calls."""
        return Message("assistant", self.text, reasoning=self.reasoning, tool_calls=self.tool_calls)


class ModelClient(Protocol):
    """What Vardex needs from a model provider. Any class with these members will do."""

    model: str

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one request. With `on_text`, pass on each piece of text as it arrives."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count the tokens a text uses when sent as a question."""
        ...


@dataclass
class Ledger:
    """A ModelClient that passes each request on and keeps every answer that comes back, so the
    calls a run made are still known, and paid for, when a later call fails."""

    client: ModelClient
    answers: list[Answer] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.client.model

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        answer = self.client.send(request, on_text)
        self.answers.append(answer)
        return answer

    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)


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


@dataclass
class Conversation:
    """A conversation with a model. Every turn sends the whole history again, because the
    model remembers nothing between calls; the prompt cache makes the repeated part cheap."""

    client: ModelClient
    system: str = ASK
    messages: list[Message] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)

    def say(self, text: str, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one message and add both it and the reply to the history."""
        request = Request(
            system=self.system,
            messages=[*self.messages, Message("user", text)],
            cache_conversation=True,
        )
        answer = self.client.send(request, on_text=on_text)  # on failure, history is unchanged
        self.messages += [Message("user", text), Message("assistant", answer.text)]
        self.answers.append(answer)
        return answer

    @property
    def cost_usd(self) -> Decimal | None:
        """What the conversation has cost so far, or None if any call had no price."""
        costs = [answer.cost_usd for answer in self.answers]
        return None if None in costs else sum(costs, Decimal(0))
