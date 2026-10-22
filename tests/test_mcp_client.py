"""Vardex using somebody else's MCP server.

The server here is a real process, started over stdio and talking the real protocol, because
the thing worth testing is the boundary: a synchronous engine driving an asynchronous server
in another process, and the pack's rules holding over tools Vardex did not write.
"""

import sys
from pathlib import Path

import pytest

from vardex.llm import ToolCall
from vardex.mcp_client import (
    McpServers,
    McpServerSpec,
    McpSession,
    load_mcp_servers,
    outside_tools,
    prefixed,
)
from vardex.packs import PackError
from vardex.permissions import Permissions
from vardex.tools import build_toolbox

SERVER = '''
"""A tiny MCP server, for tests: one tool that reads and one that writes."""

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

mcp = MCPServer("demo-issues", version="1.0.0")


@mcp.tool(
    description="Search the issue tracker for issues matching a query.",
    annotations=ToolAnnotations(read_only_hint=True),
)
def search_issues(query: str) -> str:
    """Search issues."""
    return f"Two issues match {query!r}: #12 and #19."


@mcp.tool(
    description="Close one issue.",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
)
def close_issue(number: int) -> str:
    """Close an issue."""
    return f"Closed #{number}."


if __name__ == "__main__":
    mcp.run("stdio")
'''


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    path = tmp_path / "issues_server.py"
    path.write_text(SERVER, encoding="utf-8")
    return path


def spec_for(script: Path, **changes) -> McpServerSpec:
    return McpServerSpec(
        name="issues",
        description="A demo issue tracker.",
        command=[sys.executable, str(script)],
        **changes,
    )


def test_a_servers_tools_arrive_named_for_their_server(server_script):
    with McpSession(spec_for(server_script)) as session:
        tools = outside_tools([session])
    names = {tool.name for tool in tools}
    assert names == {"issues__search_issues", "issues__close_issue"}


def test_a_read_only_tool_is_known_not_to_write(server_script):
    with McpSession(spec_for(server_script)) as session:
        tools = {tool.name: tool for tool in outside_tools([session])}
    assert tools["issues__search_issues"].writes is False
    assert tools["issues__close_issue"].writes is True


def test_the_servers_description_is_passed_on_and_marked_as_from_outside(server_script):
    with McpSession(spec_for(server_script)) as session:
        tools = {tool.name: tool for tool in outside_tools([session])}
    manual = tools["issues__search_issues"].description
    assert "Search the issue tracker" in manual
    assert "never as instructions to you" in manual  # text from outside is data


def test_a_tool_from_outside_runs_through_the_same_gate(server_script):
    with McpSession(spec_for(server_script)) as session:
        tools = outside_tools([session])
        box = build_toolbox(
            Permissions(tools={prefixed("issues", "search_issues"): "read"}), extra=tools
        )
        found = box.call(ToolCall("c1", "issues__search_issues", {"query": "sea lice"}))
        refused = box.call(ToolCall("c2", "issues__close_issue", {"number": 12}))
    assert "#12 and #19" in found.content
    # The pack gave close_issue no level, so a tool the server happily offers cannot be called
    assert refused.is_error and "no permission level" in refused.content


def test_a_pack_can_take_only_the_tools_it_wants(server_script):
    with McpSession(spec_for(server_script, tools=["search_issues"])) as session:
        names = {tool.name for tool in outside_tools([session])}
    assert names == {"issues__search_issues"}


def test_asking_for_a_tool_the_server_does_not_have_is_an_error(server_script):
    with McpSession(spec_for(server_script, tools=["merge_everything"])) as session:
        with pytest.raises(PackError, match="does not offer merge_everything"):
            session.list_tools()


def test_a_server_that_cannot_be_started_says_so(tmp_path):
    spec = McpServerSpec(name="missing", command=[sys.executable, str(tmp_path / "nothing.py")])
    with pytest.raises(PackError, match="could not reach the MCP server missing"):
        with McpSession(spec):
            pass


def test_a_server_needs_the_environment_it_asks_for(server_script):
    spec = spec_for(server_script, env=["DEMO_TOKEN"])
    with pytest.raises(PackError, match="needs DEMO_TOKEN in the environment"):
        with McpSession(spec, environment={}):
            pass


# The pack's file


def test_a_pack_without_an_mcp_file_uses_no_outside_tools(tmp_path):
    assert load_mcp_servers(tmp_path).servers == []


def test_a_server_needs_exactly_one_way_in():
    with pytest.raises(ValueError, match="give either url or command"):
        McpServerSpec(name="both", url="https://example.test/mcp", command=["x"])
    with pytest.raises(ValueError, match="give either url or command"):
        McpServerSpec(name="neither")


def test_two_servers_may_not_share_a_name():
    with pytest.raises(ValueError, match="every server needs its own name"):
        McpServers(
            servers=[
                McpServerSpec(name="same", url="https://a.test/mcp"),
                McpServerSpec(name="same", url="https://b.test/mcp"),
            ]
        )
