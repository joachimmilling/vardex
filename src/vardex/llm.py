"""The smallest useful model client. Chapter 3 turns it into a proper one."""

from anthropic import Anthropic

from vardex.config import Settings

SYSTEM_PROMPT = (
    "You are Vardex, an assistant for analysts. "
    "Answer briefly and precisely. If you are not sure, say so."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when no Anthropic API key is configured."""


def ask(question: str, settings: Settings, client: Anthropic | None = None) -> str:
    """Send one question to the model and return the text of its answer."""
    if client is None:
        if not settings.anthropic_api_key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        client = Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.model,
        max_tokens=4096,  # includes the model's hidden thinking; chapter 2 explains
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
