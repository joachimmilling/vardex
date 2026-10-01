"""The model client: a question in; the answer, its tokens, time and cost out."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from anthropic import Anthropic

from vardex.config import Settings
from vardex.pricing import UnknownModelError, Usage, cost_usd

SYSTEM_PROMPT = (
    "You are Vardex, an assistant for analysts. "
    "Answer briefly and precisely. If you are not sure, say so."
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class MissingAPIKeyError(RuntimeError):
    """Raised when no Anthropic API key is configured."""


@dataclass(frozen=True)
class Answer:
    """A model's answer, and what it took to produce it."""

    text: str
    model: str
    usage: Usage
    thinking_tokens: int | None  # already included in usage.output_tokens
    stop_reason: str | None
    seconds: float
    cost_usd: Decimal | None  # None when the model is not in the price table
    first_text_seconds: float | None = None  # only measured when the answer is streamed

    @property
    def truncated(self) -> bool:
        """True when the answer was cut off because it reached max_tokens."""
        return self.stop_reason == "max_tokens"


def make_client(settings: Settings) -> Anthropic:
    """Create an API client, or fail with a clear message when there is no key."""
    if not settings.anthropic_api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def read_usage(api_usage: Any) -> Usage:
    """Turn the API's usage object into Vardex's own Usage."""
    return Usage(
        input_tokens=api_usage.input_tokens,
        output_tokens=api_usage.output_tokens,
        cache_write_tokens=getattr(api_usage, "cache_creation_input_tokens", None) or 0,
        cache_read_tokens=getattr(api_usage, "cache_read_input_tokens", None) or 0,
    )


def ask(
    question: str,
    settings: Settings,
    client: Anthropic | None = None,
    *,
    effort: Effort | None = None,
    max_tokens: int = 4096,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Send one question to the model and return its answer with tokens, time and cost.

    With on_text, the answer is streamed: on_text gets each piece of text as it is written.
    """
    if client is None:
        client = make_client(settings)

    request: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": max_tokens,  # counts thinking as well as the visible answer
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": question}],
    }
    if effort is not None:
        request["output_config"] = {"effort": effort}

    started = time.perf_counter()
    first_text_seconds = None
    if on_text is None:
        response = client.messages.create(**request)
    else:
        with client.messages.stream(**request) as stream:
            for text in stream.text_stream:
                if first_text_seconds is None:
                    first_text_seconds = time.perf_counter() - started
                on_text(text)
            response = stream.get_final_message()
    seconds = time.perf_counter() - started

    usage = read_usage(response.usage)
    details = getattr(response.usage, "output_tokens_details", None)
    try:
        cost = cost_usd(settings.model, usage)
    except UnknownModelError:
        cost = None

    return Answer(
        text="".join(block.text for block in response.content if block.type == "text"),
        model=response.model,
        usage=usage,
        thinking_tokens=getattr(details, "thinking_tokens", None),
        stop_reason=response.stop_reason,
        seconds=seconds,
        cost_usd=cost,
        first_text_seconds=first_text_seconds,
    )


def count_tokens(text: str, settings: Settings, client: Anthropic | None = None) -> int:
    """Count the tokens a text uses when sent as a question to the configured model."""
    if client is None:
        client = make_client(settings)
    result = client.messages.count_tokens(
        model=settings.model,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens
