"""The agent loop.

An agent is a loop, and a short one. Send the question and the tools; the model either answers
or asks for tools; run them; send the results back; repeat.

It is written out here rather than taken from a framework because the loop is the short part.
What surrounds it is not, and that is where this project's requirements live:

- It must send the model's own reasoning back untouched, or the provider refuses the next call.
- It must return every tool's reply in one message, or the model learns to stop asking for
  tools in parallel.
- It must stop. A loop with no step cap and no spending cap is an open invoice, and a model
  that cannot make progress will keep trying.
- It must keep a record. When an answer is wrong, the steps are the only way to see why.

The loop itself never decides what a tool may do: it asks the Toolbox, which asks the pack.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from vardex.budgets import BudgetError
from vardex.llm import (
    Answer,
    Message,
    ModelClient,
    ModelError,
    Request,
    ToolCall,
    ToolReply,
)
from vardex.pricing import Usage
from vardex.prompts import AGENT
from vardex.tools import Toolbox, ToolResult

MAX_STEPS = 12  # model calls in one run; beyond this a question is not getting answered
MAX_TOKENS = 4096

# Why a run ended. Only `answered` means the question was answered.
Stop = Literal["answered", "steps", "budget", "refused", "failed", "truncated"]


@dataclass(frozen=True)
class Step:
    """One turn of the loop: what the model said, what it asked for, and what it got."""

    number: int
    text: str  # what it said before asking for tools, which is often its plan
    answer: Answer
    calls: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.answer.seconds


@dataclass(frozen=True)
class Run:
    """A whole run: every step, the answer, and what it cost."""

    question: str
    answer: str
    stop: Stop
    steps: list[Step]
    note: str = ""  # why it stopped, when it did not answer

    @property
    def tool_calls(self) -> list[tuple[ToolCall, ToolResult]]:
        return [call for step in self.steps for call in step.calls]

    @property
    def usage(self) -> Usage:
        """Every token the run used, added up."""
        return Usage(
            input_tokens=sum(s.answer.usage.input_tokens for s in self.steps),
            output_tokens=sum(s.answer.usage.output_tokens for s in self.steps),
            cache_write_tokens=sum(s.answer.usage.cache_write_tokens for s in self.steps),
            cache_read_tokens=sum(s.answer.usage.cache_read_tokens for s in self.steps),
        )

    @property
    def cost_usd(self) -> Decimal | None:
        """What the run cost, or None if any call had no price."""
        costs = [step.answer.cost_usd for step in self.steps]
        return None if None in costs else sum(costs, Decimal(0))

    @property
    def seconds(self) -> float:
        return sum(step.seconds for step in self.steps)

    @property
    def answered(self) -> bool:
        return self.stop == "answered"


def run_agent(
    client: ModelClient,
    toolbox: Toolbox,
    question: str,
    *,
    system: str = AGENT,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    on_text: Callable[[str], None] | None = None,
    on_call: Callable[[ToolCall, ToolResult], None] | None = None,
) -> Run:
    """Answer one question by letting the model call the toolbox's tools.

    Returns a Run whatever happens: answered, out of steps, out of budget, or stopped by the
    model itself. A caller gets the steps either way, because a run that failed half way is
    exactly the one you need to read.
    """
    messages = [Message("user", question)]
    steps: list[Step] = []
    specs = toolbox.specs  # built once: a tool list that changes costs the whole prompt cache
    for number in range(1, max_steps + 1):
        request = Request(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            tools=specs,
            cache_system=True,
            cache_conversation=True,
        )
        try:
            answer = client.send(request, on_text=on_text)
        except BudgetError as err:
            return _stopped(question, steps, "budget", str(err))
        except ModelError as err:
            if err.retryable:
                # The provider's own retries are already spent; stopping keeps the steps and
                # the spending visible instead of hiding them inside a longer wait.
                return _stopped(question, steps, "failed", str(err))
            raise
        step = Step(number=number, text=answer.text, answer=answer)
        steps.append(step)
        if answer.stop_reason == "refusal":
            return _stopped(question, steps, "refused", "The model refused the request.")
        if answer.stop_reason == "max_tokens":
            return _stopped(
                question, steps, "truncated", f"The answer reached max_tokens ({max_tokens})."
            )
        if answer.stop_reason != "tool_use" or not answer.tool_calls:
            return Run(question, answer.text, "answered", steps)
        # The turn goes back whole: the reasoning, the text and the calls, in that order.
        messages.append(answer.as_message())
        replies = []
        for call in answer.tool_calls:
            result = toolbox.call(call)
            step.calls.append((call, result))
            if on_call is not None:
                on_call(call, result)
            replies.append(ToolReply(call.id, result.content, is_error=result.is_error))
        messages.append(Message("user", "", tool_replies=tuple(replies)))
    return _stopped(
        question,
        steps,
        "steps",
        f"Stopped after {max_steps} steps without an answer.",
    )


def _stopped(question: str, steps: list[Step], stop: Stop, note: str) -> Run:
    """A run that ended without answering: keep the last thing the model said, and why."""
    text = steps[-1].text if steps else ""
    return Run(question, text, stop, steps, note=note)


def transcript(run: Run) -> str:
    """The run as something a person can read: each step, its calls and what came back."""
    lines = [f"Question: {run.question}"]
    for step in run.steps:
        lines.append(f"\nStep {step.number} ({step.seconds:.1f} s)")
        if step.text.strip():
            lines.append(f"  {step.text.strip()}")
        for call, result in step.calls:
            arguments = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            mark = "!" if result.is_error else " "
            first = result.content.strip().splitlines()[0] if result.content.strip() else ""
            lines.append(f" {mark} {call.name}({arguments})")
            lines.append(f"     → {first[:100]}")
    if run.answered:
        lines.append(f"\nAnswer: {run.answer}")
    else:
        lines.append(f"\nStopped ({run.stop}): {run.note}")
    return "\n".join(lines)
