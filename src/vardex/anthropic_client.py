"""The ModelClient for Claude, through Anthropic's Python SDK.

This is the only module in Vardex that imports `anthropic`; a test enforces it. Everything the
rest of Vardex should not have to know about Anthropic's API is handled here.
"""

import time
from collections.abc import Callable
from typing import Any

import anthropic

from vardex.config import Settings
from vardex.llm import Answer, MissingAPIKeyError, ModelError, Request, StopReason
from vardex.pricing import UnknownModelError, Usage, cost_usd

MODELS_WITHOUT_EFFORT = ("claude-haiku-4-5",)  # these reject output_config.effort
STOP_REASONS: dict[str, StopReason] = {  # Anthropic's stop_reason → Vardex's StopReason
    "end_turn": "end",
    "stop_sequence": "end",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "refusal": "refusal",
}


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
                f"The API refused the request ({err.status_code}): {err.message}"
            ) from err
        except anthropic.APIConnectionError as err:  # also covers timeouts
            raise ModelError(f"Could not reach the API: {err}") from err
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
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.text} for m in request.messages],
        }
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
            text="".join(block.text for block in response.content if block.type == "text"),
            model=response.model,
            usage=usage,
            thinking_tokens=getattr(details, "thinking_tokens", None),
            stop_reason=STOP_REASONS.get(response.stop_reason, "other"),
            seconds=seconds,
            cost_usd=cost,
            first_text_seconds=first_text_seconds,
        )
