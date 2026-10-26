"""The agent loop: what it sends, what it stops for, and what it records."""

import time
from decimal import Decimal

import pytest
from fakes import FakeModel, Turn

from vardex.agent import MAX_STEPS, run_agent, transcript
from vardex.budgets import Budget
from vardex.llm import ModelError, ToolCall
from vardex.permissions import Budgets, Permissions
from vardex.replay import dump_answer, load_answer
from vardex.tools import (
    Invocation,
    Tool,
    ToolFailure,
    ToolResult,
    arguments_schema,
    build_toolbox,
)


def counting_tool(calls: list[dict]) -> Tool:
    def run(invocation: Invocation) -> ToolResult:
        calls.append(invocation.arguments)
        return ToolResult(f"The answer is {len(calls)}.")

    return Tool(
        name="look_up",
        summary="Look one thing up.",
        description="Looks one thing up.",
        schema=arguments_schema({"what": {"type": "string"}}, ["what"]),
        run=run,
    )


def toolbox(calls: list[dict], **extra):
    return build_toolbox(
        Permissions(tools={"look_up": "read", "describe_tool": "read"}),
        extra=[counting_tool(calls)],
        **extra,
    )


def test_a_question_answered_without_tools_is_one_step():
    calls: list[dict] = []
    model = FakeModel("Nothing to look up: it is 42.")
    run = run_agent(model, toolbox(calls), "What is it?")
    assert run.answered and run.answer == "Nothing to look up: it is 42."
    assert len(run.steps) == 1
    assert calls == []


def test_a_tool_call_is_run_and_its_reply_sent_back():
    calls: list[dict] = []
    model = FakeModel(
        Turn("Let me look that up.", [("look_up", {"what": "revenue"})]),
        "Revenue was 1.",
    )
    run = run_agent(model, toolbox(calls), "What was revenue?")
    assert run.answered and run.answer == "Revenue was 1."
    assert calls == [{"what": "revenue"}]
    # The second request carries the assistant turn and the reply to its call
    second = model.requests[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[1].tool_calls[0].name == "look_up"
    assert second.messages[2].tool_replies[0].content == "The answer is 1."


def test_the_models_reasoning_goes_back_untouched():
    # A provider signs its reasoning and refuses an edited block, so the loop must carry it.
    calls: list[dict] = []
    model = FakeModel(Turn("", [("look_up", {"what": "x"})]), "Done.")
    run_agent(model, toolbox(calls), "Anything?")
    sent_back = model.requests[1].messages[1].reasoning
    assert [r.text for r in sent_back] == ["Thinking about it."]
    assert [r.signature for r in sent_back] == ["signed"]


def test_several_calls_in_one_turn_come_back_in_one_message():
    # Splitting the replies over several messages teaches the model to stop asking in parallel.
    calls: list[dict] = []
    model = FakeModel(
        Turn("Two at once.", [("look_up", {"what": "a"}), ("look_up", {"what": "b"})]),
        "Both done.",
    )
    run = run_agent(model, toolbox(calls), "Two things?")
    assert calls == [{"what": "a"}, {"what": "b"}]
    assert len(model.requests[1].messages[2].tool_replies) == 2
    assert len(run.steps[0].calls) == 2


def test_the_tool_list_is_built_once_and_sent_unchanged():
    calls: list[dict] = []
    model = FakeModel(Turn("", [("look_up", {"what": "x"})]), "Done.")
    run_agent(model, toolbox(calls), "Anything?")
    first, second = (request.tools for request in model.requests)
    assert first == second  # a tool list that changes costs the whole prompt cache


def test_it_stops_after_the_step_cap_and_says_so():
    calls: list[dict] = []
    turns = [Turn("Again.", [("look_up", {"what": "x"})])] * (MAX_STEPS + 2)
    run = run_agent(FakeModel(*turns), toolbox(calls), "Go on for ever")
    assert not run.answered and run.stop == "steps"
    assert len(run.steps) == MAX_STEPS
    assert "without an answer" in run.note


def test_a_refused_tool_is_a_step_the_model_can_recover_from():
    calls: list[dict] = []
    # look_up is allowed; write_it is not even listed, so the call comes back as an error
    model = FakeModel(
        Turn("I will write it.", [("write_it", {})]),
        "I may not write, so here is what I would have written.",
    )
    run = run_agent(model, toolbox(calls), "Write it")
    assert run.answered
    reply = model.requests[1].messages[2].tool_replies[0]
    assert reply.is_error and "no tool called" in reply.content


def test_a_run_adds_up_its_tokens_time_and_cost():
    calls: list[dict] = []
    model = FakeModel(Turn("", [("look_up", {"what": "x"})]), "Done.")
    run = run_agent(model, toolbox(calls), "Anything?")
    assert run.usage.input_tokens == 2000
    assert run.usage.output_tokens == 400
    assert run.cost_usd == Decimal("0.008")
    assert run.seconds == 3.0


def test_a_budget_stops_the_run_rather_than_the_question_stopping_it():
    calls: list[dict] = []
    model = FakeModel(*([Turn("Again.", [("look_up", {"what": "x"})])] * 5))
    # Three calls at $0.004 each fit in a cap of $0.02; the fourth cannot be guaranteed to
    budget = Budget(model, caps=Budgets(per_run=Decimal("0.02")))
    run = run_agent(budget, toolbox(calls), "Go on")
    assert not run.answered and run.stop == "budget"
    assert "budget for this question is spent" in run.note
    assert len(run.steps) < 5
    assert budget.run_usd <= Decimal("0.02")


def test_a_run_that_fails_keeps_its_steps():
    class Failing(FakeModel):
        def send(self, request, on_text=None):
            if self.steps >= 1:
                raise ModelError("overloaded", retryable=True)
            return super().send(request, on_text)

    calls: list[dict] = []
    model = Failing(Turn("", [("look_up", {"what": "x"})]), "never reached")
    run = run_agent(model, toolbox(calls), "Anything?")
    assert run.stop == "failed"
    assert len(run.steps) == 1  # what it did before failing is still there to read
    assert calls == [{"what": "x"}]


def test_a_model_error_that_will_not_pass_is_raised():
    class Broken(FakeModel):
        def send(self, request, on_text=None):
            raise ModelError("the request is wrong", retryable=False)

    with pytest.raises(ModelError):
        run_agent(Broken("unused"), toolbox([]), "Anything?")


def test_a_refusal_ends_the_run():
    model = FakeModel("I will not.", stop_reason="refusal")
    run = run_agent(model, toolbox([]), "Something off")
    assert run.stop == "refused"


def test_the_transcript_shows_every_call_and_what_came_back():
    calls: list[dict] = []
    model = FakeModel(Turn("Looking.", [("look_up", {"what": "revenue"})]), "It is 1.")
    run = run_agent(model, toolbox(calls), "What was revenue?")
    text = transcript(run)
    assert "Step 1" in text
    assert "look_up(what='revenue')" in text
    assert "→ The answer is 1." in text
    assert "Answer: It is 1." in text


def test_a_stopped_run_says_why_in_the_transcript():
    turns = [Turn("Again.", [("look_up", {"what": "x"})])] * (MAX_STEPS + 1)
    run = run_agent(FakeModel(*turns), toolbox([]), "For ever")
    assert "Stopped (steps)" in transcript(run)


def test_a_recorded_turn_that_called_tools_replays_as_itself():
    """A recording is JSON, so a turn's calls and reasoning come back as dictionaries unless
    load_answer rebuilds them. Without this the loop hands a dict to the toolbox."""
    model = FakeModel(Turn("Looking.", [("look_up", {"what": "revenue"})]), "It is 1.")
    live = run_agent(model, toolbox([]), "What was revenue?")
    asking = live.steps[0].answer
    assert asking.tool_calls, "the fake should have asked for a tool"

    replayed = load_answer(dump_answer(asking))
    assert replayed == asking
    assert isinstance(replayed.tool_calls[0], ToolCall)
    assert replayed.tool_calls[0].name == "look_up"


def sleeping_tool(seconds: float) -> Tool:
    """A tool that takes its time, so a turn's calls can be shown to overlap."""

    def run(invocation: Invocation) -> ToolResult:
        name = invocation.arguments["what"]
        if name == "boom":
            raise ToolFailure("there is no such thing")
        time.sleep(seconds)
        return ToolResult(f"Looked up {name}.")

    return Tool(
        name="look_up",
        summary="Look one thing up, slowly.",
        description="Looks one thing up, slowly.",
        schema=arguments_schema({"what": {"type": "string"}}, ["what"]),
        run=run,
    )


def sleeping_toolbox(seconds: float):
    return build_toolbox(
        Permissions(tools={"look_up": "read", "describe_tool": "read"}),
        extra=[sleeping_tool(seconds)],
    )


def test_a_turns_calls_run_at_the_same_time():
    nap = 0.3
    asked = [("look_up", {"what": w}) for w in ("one", "two", "three")]
    model = FakeModel(Turn("Three at once.", asked), "Done.")
    started = time.monotonic()
    run = run_agent(model, sleeping_toolbox(nap), "Three things?")
    elapsed = time.monotonic() - started
    assert run.answered
    assert len(run.steps[0].calls) == 3
    # Serial would be 3 naps; concurrent is one, with room for the threads themselves.
    assert elapsed < nap * 2, f"three {nap}s calls took {elapsed:.2f}s, so they ran one by one"


def test_replies_come_back_in_the_order_the_model_asked():
    """The API matches replies by id, but a stable order keeps transcripts and recordings the
    same from run to run, even when the quickest tool finishes first."""
    asked = [("look_up", {"what": w}) for w in ("slow", "quick", "middling")]
    model = FakeModel(Turn("Three at once.", asked), "Done.")
    toolbox = sleeping_toolbox(0.05)
    run = run_agent(model, toolbox, "Three things?")
    assert [call.arguments["what"] for call, _ in run.steps[0].calls] == [
        "slow",
        "quick",
        "middling",
    ]
    assert [call.arguments["what"] for call, _ in toolbox.calls] == ["slow", "quick", "middling"]


def test_a_failing_call_does_not_stop_its_neighbours():
    asked = [("look_up", {"what": w}) for w in ("one", "boom", "three")]
    model = FakeModel(Turn("Three at once.", asked), "Done.")
    run = run_agent(model, sleeping_toolbox(0.0), "Three things?")
    results = [result for _, result in run.steps[0].calls]
    assert [r.is_error for r in results] == [False, True, False]
    assert "no such thing" in results[1].content
    assert results[2].content == "Looked up three."


def test_every_call_records_how_long_it_took():
    nap = 0.2
    model = FakeModel(Turn("Looking.", [("look_up", {"what": "revenue"})]), "Done.")
    run = run_agent(model, sleeping_toolbox(nap), "How long?")
    call, result = run.steps[0].calls[0]
    assert result.seconds >= nap
    assert f"[{result.seconds:.1f} s]" in transcript(run)
