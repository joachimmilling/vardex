"""What an agent run cost, beside whether it passed: the figures, and that a replay keeps them."""

from decimal import Decimal

from fakes import FakeModel, Turn

from vardex.evals import (
    AgentCase,
    CaseResult,
    Grade,
    Outcome,
    Report,
    agent_case,
    agent_figures,
    format_report,
    grade_agent,
    median,
)
from vardex.permissions import Permissions
from vardex.replay import Recording, RecordingClient, ReplayClient
from vardex.tools import Invocation, Tool, ToolResult, arguments_schema, build_toolbox


def look_up() -> Tool:
    def run(invocation: Invocation) -> ToolResult:
        return ToolResult("DEMO LAKS AS had 1 480 million.")

    return Tool(
        name="look_up",
        summary="Look one figure up.",
        description="Looks one figure up.",
        schema=arguments_schema({"what": {"type": "string"}}, ["what"]),
        run=run,
    )


def toolbox():
    return build_toolbox(
        Permissions(tools={"look_up": "read", "describe_tool": "read"}),
        extra=[look_up()],
    )


def case(id: str = "a1") -> AgentCase:
    return AgentCase(id=id, type="figure", question="What was revenue?", expect="answer")


def test_a_runs_steps_calls_and_cost_land_in_the_outcomes_values():
    model = FakeModel(
        Turn("Looking.", [("look_up", {"what": "revenue"})]),
        "It was 1 480 million.",
    )
    result = agent_case(model, toolbox, case(), [grade_agent])
    values = result.outcome.values
    assert values is not None
    assert values["steps"] == 2
    assert values["tool_calls"] == 1
    assert values["tools"] == ["look_up"]
    # A string, not a Decimal: `values` is written to a recording as JSON.
    assert isinstance(values["cost_usd"], str)
    assert Decimal(values["cost_usd"]) > 0


def test_the_figures_are_json_so_they_survive_a_recording():
    """`values` is the only part of an outcome a recording carries, which is why the figures
    go there rather than being read back off the Run."""
    import json

    model = FakeModel(
        Turn("Looking.", [("look_up", {"what": "revenue"})]),
        "It was 1 480 million.",
    )
    result = agent_case(model, toolbox, case(), [grade_agent])
    again = json.loads(json.dumps(result.outcome.values))
    assert again == result.outcome.values


def outcome(steps: int, calls: int, cost: str | None, tools: list[str]) -> Outcome:
    return Outcome(
        "answer",
        "It was 1 480 million.",
        None,
        [],
        {"steps": steps, "tool_calls": calls, "cost_usd": cost, "tools": tools, "stop": "answered"},
    )


def report_of(*runs: tuple[int, int, str | None, list[str]]) -> Report:
    results = [
        CaseResult(case(f"a{n}"), outcome(*one), [Grade("agent", True)])
        for n, one in enumerate(runs, start=1)
    ]
    return Report(suite="agent", model="fake", results=results)


def test_the_report_gives_the_median_and_the_worst_of_each():
    report = report_of(
        (2, 1, "0.010", ["look_up"]),
        (4, 3, "0.030", ["look_up", "run_sql", "look_up"]),
        (9, 8, "0.400", ["run_sql"]),
    )
    lines = agent_figures(report)
    assert "Steps: median 4, worst 9" in lines
    assert "Cost: median $0.030, worst $0.400, total $0.44" in lines
    assert "Tools used: look_up 3, run_sql 2" in lines


def test_the_median_is_not_moved_by_one_runaway_run():
    assert median([2, 2, 2, 2, 40]) == 2


def test_a_suite_with_nothing_priced_says_so_rather_than_showing_zero():
    report = report_of((2, 1, None, ["look_up"]), (3, 2, None, ["look_up"]))
    assert "Cost: not priced" in agent_figures(report)


def test_details_show_each_case_on_its_own_line():
    report = report_of((2, 1, "0.010", ["look_up"]), (9, 8, "0.400", ["run_sql"]))
    lines = agent_figures(report, details=True)
    assert "  a1: 2 steps, 1 tool call, $0.010" in lines
    assert "  a2: 9 steps, 8 tool calls, $0.400" in lines


def test_a_suite_that_is_not_an_agent_suite_gets_no_figures():
    """Only an agent run records steps, so an answer or search suite prints nothing here."""
    results = [CaseResult(case("a1"), Outcome("answer", "x", None, []), [Grade("g", True)])]
    assert agent_figures(Report(suite="answer", model="fake", results=results)) == []


def test_the_figures_do_not_move_the_gate():
    """This is information, not a threshold: a dear run still passes."""
    report = report_of((12, 20, "9.99", ["run_sql"]))
    assert report.gate(threshold=0.9) == []
    assert "Cost: median $9.990" in format_report(report)


def test_the_figures_come_out_the_same_on_a_replay():
    """The real test of where the figures live: record a run, replay it, and the steps, the
    calls and the cost must match. They are recomputed from the recorded answers, so a cost
    that came only from a live client would quietly become None here."""
    recording = Recording("claude-sonnet-5-5", "none", "duckdb", "2026-10-26")
    live = FakeModel(
        Turn("Looking.", [("look_up", {"what": "revenue"})]),
        "It was 1 480 million.",
    )
    recorded = agent_case(RecordingClient(live, recording), toolbox, case(), [grade_agent])
    replayed = agent_case(ReplayClient(recording), toolbox, case(), [grade_agent])

    assert replayed.passed == recorded.passed
    assert replayed.outcome.values == recorded.outcome.values
    assert replayed.outcome.values is not None
    assert replayed.outcome.values["cost_usd"] is not None
