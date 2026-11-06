"""The ModelClient for Claude, through Anthropic's Python SDK.

This is the only module in Vardex that imports `anthropic`; a test enforces it. Everything the
rest of Vardex should not have to know about Anthropic's API is handled here.
"""

import base64
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import anthropic

from vardex.config import Settings
from vardex.llm import (
    Answer,
    Message,
    MissingAPIKeyError,
    ModelError,
    Reasoning,
    Request,
    StopReason,
    ToolCall,
)
from vardex.pricing import UnknownModelError, Usage, cost_usd
from vardex.tracing import HTTP_METHOD, HTTP_RESEND, HTTP_STATUS, SERVER, span

MODELS_WITHOUT_EFFORT = ("claude-haiku-4-5",)  # these reject output_config.effort
STOP_REASONS: dict[str, StopReason] = {  # Anthropic's stop_reason → Vardex's StopReason
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "refusal": "refusal",
}

# Statuses that say "not now" rather than "not this request": timeout, conflict, rate limit.
# Every 5xx, including 529 (overloaded), is retryable too.
RETRYABLE_STATUSES = (408, 409, 429)
PROVIDER = "anthropic"  # as OpenTelemetry names it in gen_ai.provider.name


def trace_attempts(request: Any, call_next: Callable[[Any], Any]) -> Any:
    """SDK middleware: a span for each HTTP attempt, inside the span of the model call.

    The SDK retries a rate limit or an overload itself, inside one call to messages.create,
    so from outside a call that took 40 seconds looks like a slow model. The SDK runs its
    middleware once per attempt, inside its retry loop, so each attempt gets a span with its
    status and its number: 0 for the first try, 1 for the first retry. The wait between
    attempts is then the gap between two spans.
    """
    attributes = {
        HTTP_METHOD: request.method,
        SERVER: urlparse(str(request.url)).hostname,
        HTTP_RESEND: request.retries_taken,
    }
    with span(request.method, attributes, kind="client") as attempt:
        response = call_next(request)  # a connection that fails raises, and fails the span
        attempt.set(HTTP_STATUS, response.status_code)
        attempt.set("anthropic.request.id", response.headers.get("request-id"))
        if response.status_code >= 400:
            attempt.fail(f"HTTP {response.status_code}", str(response.status_code))
        return response


class AnthropicClient:
    """Sends Vardex requests to Claude, with timeouts and retries set from Settings."""

    def __init__(self, settings: Settings, sdk: Any = None) -> None:
        if sdk is None:
            if not settings.anthropic_api_key:
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            sdk = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,  # the SDK waits and retries 429, 5xx and drops
                middleware=[trace_attempts],
            )
        self.sdk = sdk
        self.model = settings.model

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one request and return the answer, its tokens, time and cost."""
        params = self.to_params(request)
        first_text_seconds = None
        started = time.perf_counter()
        try:
            if on_text is None:
                response = self.sdk.messages.create(**params)
            else:
                with self.sdk.messages.stream(**params) as stream:
                    for text in stream.text_stream:
                        if first_text_seconds is None:
                            first_text_seconds = time.perf_counter() - started
                        on_text(text)
                    response = stream.get_final_message()
        except anthropic.APIStatusError as err:
            raise ModelError(
                f"The API refused the request ({err.status_code}): {err.message}",
                retryable=err.status_code in RETRYABLE_STATUSES or err.status_code >= 500,
            ) from err
        except anthropic.APIConnectionError as err:  # also covers timeouts
            raise ModelError(f"Could not reach the API: {err}", retryable=True) from err
        seconds = time.perf_counter() - started
        return self.to_answer(response, seconds, first_text_seconds)

    def count_tokens(self, text: str) -> int:
        """Count the tokens a text uses when sent as a question. The API does this for free."""
        try:
            result = self.sdk.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
        except anthropic.APIError as err:
            raise ModelError(f"The API refused the request: {err.message}") from err
        return result.input_tokens

    def to_params(self, request: Request) -> dict[str, Any]:
        """Turn a Vardex request into the keyword arguments of messages.create."""
        system: Any = request.system
        if request.cache_system:
            # A cache breakpoint at the end of the system prompt: calls within five minutes
            # read it back at a tenth of the input price. Below a model's minimum length
            # (512 tokens on Sonnet 5.5) nothing is cached, and no error is raised.
            system = [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ]
        if request.system_tail:
            # After the breakpoint, so a tail that changes leaves the cached part as it was
            blocks = system if isinstance(system, list) else [{"type": "text", "text": system}]
            system = [*blocks, {"type": "text", "text": request.system_tail}]
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": content(m)} for m in request.messages],
        }
        if request.tools:
            # Tools are rendered before the system prompt, so the same list in the same order
            # on every call is what keeps the prompt cache warm through a run.
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in request.tools
            ]
        if request.cache_conversation:
            # Automatic caching: the breakpoint moves to the last message on every turn, so
            # each turn reads the whole earlier conversation from the cache.
            params["cache_control"] = {"type": "ephemeral"}
        output_config: dict[str, Any] = {}
        if request.effort is not None and not self.model.startswith(MODELS_WITHOUT_EFFORT):
            output_config["effort"] = request.effort
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                # transform_schema adapts a Pydantic schema to what structured outputs accept
                "schema": anthropic.transform_schema(request.output_schema),
            }
        if output_config:
            params["output_config"] = output_config
        return params

    def to_answer(self, response: Any, seconds: float, first_text_seconds: float | None) -> Answer:
        """Turn the SDK's response into Vardex's own Answer."""
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", None) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", None) or 0,
        )
        details = getattr(response.usage, "output_tokens_details", None)
        try:
            cost = cost_usd(self.model, usage)
        except UnknownModelError:
            cost = None
        return Answer(
            id=getattr(response, "id", "") or "",
            text="".join(block.text for block in response.content if block.type == "text"),
            model=response.model,
            usage=usage,
            thinking_tokens=getattr(details, "thinking_tokens", None),
            stop_reason=STOP_REASONS.get(response.stop_reason, "other"),
            seconds=seconds,
            cost_usd=cost,
            first_text_seconds=first_text_seconds,
            tool_calls=tuple(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                for block in response.content
                if block.type == "tool_use"
            ),
            reasoning=tuple(
                Reasoning(
                    text=getattr(block, "thinking", "") or "",
                    signature=getattr(block, "signature", "") or "",
                    data=getattr(block, "data", "") or "",
                )
                for block in response.content
                if block.type in ("thinking", "redacted_thinking")
            ),
        )


def content(message: Message) -> str | list[dict[str, Any]]:
    """A message's content as Anthropic wants it: text alone when that is all there is, else
    the blocks in the order the API requires — reasoning, pictures, text, then calls, or the
    replies to the calls of the turn before."""
    if not (message.images or message.reasoning or message.tool_calls or message.tool_replies):
        return message.text
    blocks: list[dict[str, Any]] = []
    for block in message.reasoning:
        # Sent back exactly as it came: the signature covers the text, and an edited block is
        # refused. A block the provider encrypted carries `data` and no readable text.
        if block.data:
            blocks.append({"type": "redacted_thinking", "data": block.data})
        else:
            blocks.append(
                {"type": "thinking", "thinking": block.text, "signature": block.signature}
            )
    blocks += [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": base64.b64encode(image.data).decode("ascii"),
            },
        }
        for image in message.images
    ]
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    blocks += [
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    ]
    # Every reply to a turn's calls goes in this one message. Splitting them over several
    # teaches the model to stop asking for tools in parallel.
    blocks += [
        {
            "type": "tool_result",
            "tool_use_id": reply.id,
            "content": reply.content,
            "is_error": reply.is_error,
        }
        for reply in message.tool_replies
    ]
    return blocks
