"""A stand-in for a model, so tests never call a real API or spend money."""

from decimal import Decimal

from vardex.llm import Answer
from vardex.pricing import Usage


class FakeModel:
    """A ModelClient that replies with prepared texts, in order, and keeps every request.

    A reply that is an exception is raised instead, as a failed call.
    """

    def __init__(self, *replies: str, stop_reason: str = "end") -> None:
        self.replies = list(replies)
        self.requests = []
        self.model = "claude-sonnet-5-5"
        self.stop_reason = stop_reason

    def send(self, request, on_text=None):
        self.requests.append(request)
        text = self.replies.pop(0)
        if isinstance(text, Exception):
            raise text
        if on_text:
            on_text(text)
        return Answer(
            text=text,
            model=self.model,
            usage=Usage(input_tokens=1000, output_tokens=200),
            thinking_tokens=150,
            stop_reason=self.stop_reason,
            seconds=1.5,
            cost_usd=Decimal("0.004"),  # 1,000 × $2 + 200 × $10, per million
            first_text_seconds=0.5 if on_text else None,
        )

    def count_tokens(self, text):
        return 17
