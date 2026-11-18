"""The ModelClient for any server that speaks OpenAI's Chat Completions API.

That is most of them: Ollama and LM Studio on a laptop, vLLM and SGLang on a GPU server, and
hosted providers of open-weight models such as Together, Fireworks and DeepInfra. One client
reaches all of them; only the address, the key and the model's name change.

This is the only module in Vardex that imports `openai`; a test enforces it. "Compatible" is
a promise about the shape of a request, not about every field in it: a server may ignore an
option it does not support, such as the effort, rather than refuse it. Anything one server
offers and the others do not goes through `Request.extra`, untouched.
"""

import base64
import json
import os
import time
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import openai
from pydantic import BaseModel

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

STOP_REASONS: dict[str, StopReason] = {  # Chat Completions' finish_reason → Vardex's StopReason
    "stop": "end",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}
# Chat Completions has three levels of effort; Vardex's two highest ask for the most there is
EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
RETRYABLE_STATUSES = (408, 409, 429)  # and every 5xx, as for any provider
OLLAMA = "http://localhost:11434"
# A model on a laptop writes 15 to 30 tokens a second, so a long answer with its thinking
# takes minutes. At the usual two minutes, the client gives up on an answer still coming and
# the retry starts again from nothing; a server that goes on writing the abandoned answer, as
# mlx-lm's does, makes every request after it wait as well.
LOCAL_TIMEOUT_SECONDS = 600.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def connect(settings: Settings, model: str) -> "OpenAIClient":
    """The `openai` provider: any OpenAI-compatible server, at OPENAI_BASE_URL with
    OPENAI_API_KEY, the variables OpenAI's own SDK reads. Unset, it is OpenAI itself."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Set it, and OPENAI_BASE_URL for a server other than "
            "OpenAI's own; a local server that checks no key takes any value."
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    if is_local(base_url):  # mlx-lm, LM Studio or llama.cpp on this machine, or tunnelled to it
        settings = at_local_pace(settings)
    return OpenAIClient(settings, model, base_url=base_url, api_key=key)


def at_local_pace(settings: Settings) -> Settings:
    return replace(settings, timeout_seconds=max(settings.timeout_seconds, LOCAL_TIMEOUT_SECONDS))


def is_local(url: str | None) -> bool:
    """Whether a server's address is this machine, where answers come at a laptop's pace."""
    return bool(url) and (urlparse(url).hostname or "") in LOCAL_HOSTS


def connect_ollama(settings: Settings, model: str) -> "OpenAIClient":
    """The `ollama` provider: the same client, at Ollama's address on this machine, or at
    OLLAMA_HOST. Ollama checks no key, and a model on your own machine has no price per token."""
    host = os.getenv("OLLAMA_HOST") or OLLAMA
    if "://" not in host:
        host = f"http://{host}"  # Ollama's own variable may be written as host:port alone
    return OpenAIClient(
        at_local_pace(settings),
        model,
        base_url=f"{host.rstrip('/')}/v1",
        api_key="ollama",
        provider="ollama",
        free=True,
    )


class OpenAIClient:
    """Sends Vardex requests to an OpenAI-compatible server."""

    def __init__(
        self,
        settings: Settings,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str = "",
        provider: str = "openai",
        free: bool = False,
        sdk: Any = None,
    ) -> None:
        self.sdk = sdk or openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,  # the SDK waits and retries 429, 5xx and drops
        )
        self.model = model
        self.provider = provider  # as OpenTelemetry names it in gen_ai.provider.name
        self.free = free  # a model on your own machine: no price per token
        self.where = base_url or "OpenAI"

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        """Send one request and return the answer, its tokens, time and cost."""
        params = self.to_params(request)
        started = time.perf_counter()
        try:
            if on_text is None:
                response = self.sdk.chat.completions.create(**params)
                return self.to_answer(response, time.perf_counter() - started, None)
            return self.stream(params, on_text, started)
        except openai.APIStatusError as err:
            raise ModelError(
                f"{self.where} refused the request ({err.status_code}): {err.message}",
                retryable=err.status_code in RETRYABLE_STATUSES or err.status_code >= 500,
            ) from err
        except openai.APIConnectionError as err:  # also covers timeouts
            raise ModelError(f"Could not reach {self.where}: {err}", retryable=True) from err

    def stream(
        self, params: dict[str, Any], on_text: Callable[[str], None], started: float
    ) -> Answer:
        """Stream the answer, passing each piece of text on, and put the pieces together."""
        params = {**params, "stream": True, "stream_options": {"include_usage": True}}
        text, thinking, calls = [], [], {}
        finish, usage, model, first, ident = None, None, self.model, None, ""
        for chunk in self.sdk.chat.completions.create(**params):
            ident, model = chunk.id or ident, chunk.model or model
            usage = chunk.usage or usage  # the last chunk carries it, with no choices
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    if first is None:
                        first = time.perf_counter() - started
                    text.append(delta.content)
                    on_text(delta.content)
                thinking.append(reasoning_of(delta))
                for part in delta.tool_calls or []:
                    # A call arrives in pieces: its id and name first, its arguments in parts
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = part.id or call["id"]
                    if part.function:
                        call["name"] = part.function.name or call["name"]
                        call["arguments"] += part.function.arguments or ""
                finish = choice.finish_reason or finish
        message = {
            "content": "".join(text),
            "reasoning": "".join(thinking),
            "tool_calls": [calls[i] for i in sorted(calls)],
        }
        return self.answer(
            ident, model, message, finish, usage, time.perf_counter() - started, first
        )

    def count_tokens(self, text: str) -> int:
        """Count the tokens a text uses when sent as a question. Chat Completions has no way
        to ask, so this asks for an answer of one token and reads how long the prompt was,
        which includes the few tokens the server's chat template wraps around it."""
        request = Request(system="", messages=[Message("user", text)], max_tokens=1)
        params = {k: v for k, v in self.to_params(request).items() if k != "messages"}
        params["messages"] = [{"role": "user", "content": text}]
        try:
            response = self.sdk.chat.completions.create(**params)
        except openai.APIError as err:
            raise ModelError(f"{self.where} refused the request: {err}") from err
        return response.usage.prompt_tokens

    def to_params(self, request: Request) -> dict[str, Any]:
        """Turn a Vardex request into the keyword arguments of chat.completions.create."""
        system = "\n\n".join(part for part in (request.system, request.system_tail) if part)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for message in request.messages:
            messages += as_messages(message)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # The older name, though OpenAI's SDK calls it deprecated: Ollama reads only
            # max_tokens and quietly ignores max_completion_tokens, and vLLM takes both
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema,
                    },
                }
                for t in request.tools
            ]
        if request.effort is not None:
            params["reasoning_effort"] = EFFORTS[request.effort]
        if request.output_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": strict_schema(request.output_schema),
                    "strict": True,
                },
            }
        # The prompt cache is the server's own business: OpenAI caches long prompts by
        # itself, and a local server keeps the last prompt it read. There is nothing to mark.
        if request.extra:
            params["extra_body"] = request.extra
        return params

    def to_answer(self, response: Any, seconds: float, first_text_seconds: float | None) -> Answer:
        """Turn the SDK's response into Vardex's own Answer."""
        choice = response.choices[0]
        message = {
            "content": choice.message.content or "",
            "reasoning": reasoning_of(choice.message),
            "tool_calls": [
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in choice.message.tool_calls or []
            ],
        }
        return self.answer(
            response.id, response.model, message, choice.finish_reason, response.usage, seconds,
            first_text_seconds,
        )  # fmt: skip

    def answer(
        self,
        ident: str,
        model: str,
        message: dict[str, Any],
        finish: str | None,
        usage: Any,
        seconds: float,
        first_text_seconds: float | None,
    ) -> Answer:
        """An Answer from the parts of a response, whether it came whole or streamed."""
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) or 0
        details = getattr(usage, "completion_tokens_details", None)
        counted = Usage(
            # The prompt's tokens include the cached ones; Vardex counts those apart
            input_tokens=(getattr(usage, "prompt_tokens", 0) or 0) - cached,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
        )
        calls = tuple(
            ToolCall(id=c["id"], name=c["name"], arguments=json.loads(c["arguments"] or "{}"))
            for c in message["tool_calls"]
        )
        return Answer(
            id=ident or "",
            text=message["content"],
            model=model or self.model,
            usage=counted,
            thinking_tokens=getattr(details, "reasoning_tokens", None),
            stop_reason="tool_use" if calls else STOP_REASONS.get(finish or "", "other"),
            seconds=seconds,
            cost_usd=self.cost(counted),
            first_text_seconds=first_text_seconds,
            tool_calls=calls,
            # Read for the record only: Chat Completions takes no reasoning back
            reasoning=(Reasoning(message["reasoning"]),) if message["reasoning"] else (),
        )

    def cost(self, usage: Usage) -> Decimal | None:
        if self.free:
            return Decimal(0)
        try:
            return cost_usd(self.model, usage)
        except UnknownModelError:
            return None


def reasoning_of(message: Any) -> str:
    """The reasoning a server returned beside the answer, under whichever name it uses:
    Ollama calls it reasoning, vLLM and DeepSeek reasoning_content. Neither is in OpenAI's
    own API, so the SDK keeps them as extra fields."""
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def as_messages(message: Message) -> list[dict[str, Any]]:
    """One Vardex message as Chat Completions messages. A turn's tool replies become one
    message each, with the role tool; everything else is one message."""
    if message.tool_replies:
        replies = [
            {
                "role": "tool",
                "tool_call_id": reply.id,
                # Chat Completions has no flag for a failed call, so the text says it
                "content": f"Error: {reply.content}" if reply.is_error else reply.content,
            }
            for reply in message.tool_replies
        ]
        return replies + ([{"role": "user", "content": message.text}] if message.text else [])
    content: Any = message.text
    if message.images:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.media_type};base64,"
                    + base64.b64encode(image.data).decode("ascii")
                },
            }
            for image in message.images
        ]
        if message.text:
            content.append({"type": "text", "text": message.text})
    turn: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return [turn]


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A Pydantic model's JSON Schema, as strict structured outputs want it: every object
    closed to properties it does not list, and every property required. A field with a default
    becomes one the model must fill in; Vardex validates the answer with the model anyway."""

    def close(node: Any) -> Any:
        if isinstance(node, dict):
            node = {key: close(value) for key, value in node.items()}
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            return node
        if isinstance(node, list):
            return [close(item) for item in node]
        return node

    return close(model.model_json_schema())
