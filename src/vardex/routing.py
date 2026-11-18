"""Routing: the small model first, the large one only when the small one's work fails.

Most of the work a pack is given is routine: a lookup in the warehouse, the figures on a
plain invoice. A small model does it as well as a large one, at a fraction of the cost, or at
none on your own machine. The trouble is knowing which work that is before it is done. A
classifier that guesses from the input is one more model to be wrong. Vardex already has a
better signal, after the fact: every capability checks its own output deterministically, and
an answer whose SQL never ran, or values whose quotes are not in the document, are work the
small model got wrong in a way the engine can see. So the small model goes first, and the
large one is asked only when the small one's work fails its checks or it gives up.

What this cannot catch is work that passes every check and is still wrong: the right kind of
query on the wrong rows, a figure quoted from the wrong column. That share is what the evals
measure, on the small model alone, before routing is turned on.
"""

from collections.abc import Callable
from dataclasses import dataclass

from vardex.capabilities import FAILED, UNREACHABLE, Attempt, Capability
from vardex.llm import Answer, ModelClient, ModelError, Request
from vardex.tracing import current, span


@dataclass
class Fallback:
    """A ModelClient that sends a call to a second model when the first cannot be reached.

    Only a failure that says "not now" moves a call: a rate limit, an overload, a server that
    is down, after the provider's own retries. A refused request would be refused by the other
    model too, and a missing key is a setting to fix, so those are raised. Every call that
    fell back is an event on its span, so a dashboard can count them.
    """

    primary: ModelClient
    fallback: ModelClient

    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def provider(self) -> str:
        return getattr(self.primary, "provider", "vardex")

    def send(self, request: Request, on_text: Callable[[str], None] | None = None) -> Answer:
        try:
            return self.primary.send(request, on_text)
        except ModelError as err:
            if not err.retryable:
                raise
            current().event(
                "vardex.fallback", {"vardex.fallback.model": self.fallback.model, "error": str(err)}
            )
        # Text streamed before the first model failed has been passed on already; the
        # fallback's answer starts again from nothing
        return self.fallback.send(request, on_text)

    def count_tokens(self, text: str) -> int:
        return self.primary.count_tokens(text)


@dataclass(frozen=True)
class Route:
    """The two models a piece of work may go to, in the order it goes to them."""

    small: ModelClient
    large: ModelClient


@dataclass(frozen=True)
class Routed:
    """The attempt that stands, every call both models made, and why the large model was
    asked, or empty when it was not."""

    attempt: Attempt
    answers: list[Answer]
    escalated: str = ""


def why_escalate(attempt: Attempt, gives_up: tuple[str, ...]) -> str:
    """Why the small model's attempt is not good enough to stand, or empty when it is."""
    if attempt.kind == FAILED:
        return f"it failed the checks: {attempt.text}"
    if attempt.kind == UNREACHABLE:
        return f"it could not be reached: {attempt.text}"
    if attempt.kind in gives_up:
        return f"it gave up: {attempt.text}"
    return ""


def run(route: Route, capability: Capability) -> Routed:
    """Do the work with the small model, and with the large one when the small one's attempt
    failed its checks, gave up, or could not be had."""
    with span("route", {"vardex.route.small": route.small.model}) as routed:
        routed.set("vardex.capability", capability.name)
        small = capability.attempt(route.small)
        why = why_escalate(small, capability.gives_up)
        if not why:
            routed.set("vardex.route.model", route.small.model)
            return Routed(small, small.answers)
        routed.update({"vardex.route.model": route.large.model, "vardex.route.reason": why})
        large = capability.attempt(route.large)
        return Routed(large, small.answers + large.answers, why)


def attempt(work: Capability, client: ModelClient, small: ModelClient | None = None) -> Routed:
    """Do the work with the client alone, or routed when there is a small model: the one
    place that decides, so a command and its eval route alike."""
    if small is None:
        done = work.attempt(client)
        return Routed(done, done.answers)
    return run(Route(small, client), work)
