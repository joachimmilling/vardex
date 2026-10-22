"""Vardex as an MCP server: what another agent sees, and how approval travels.

The client here is the real SDK client, talking to the real server object in the same
process, so the tests go through the protocol rather than around it.
"""

import asyncio
from typing import Any

import mcp.types as mcp_types
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from vardex.mcp_server import build_server
from vardex.permissions import Permissions
from vardex.tools import Invocation, Tool, ToolResult, arguments_schema, build_toolbox

WRITTEN: list[str] = []


def look_up() -> Tool:
    def run(invocation: Invocation) -> ToolResult:
        return ToolResult(f"It is {invocation.argument('what')}.", structured={"value": 42})

    return Tool(
        name="look_up",
        summary="Look one thing up.",
        description="The whole manual for looking one thing up.",
        schema=arguments_schema({"what": {"type": "string"}}, ["what"]),
        run=run,
    )


def raise_limit() -> Tool:
    def run(invocation: Invocation) -> ToolResult:
        sentence = f"Raise the limit for {invocation.argument('key')} to 100."
        if invocation.dry_run:
            return ToolResult(f"Dry run: {sentence}", structured={"sentence": sentence})
        if not invocation.may_write:
            return ToolResult(f"Proposed: {sentence}", structured={"sentence": sentence})
        WRITTEN.append(sentence)
        return ToolResult(f"Done: {sentence}", structured={"sentence": sentence}, wrote=True)

    return Tool(
        name="raise_limit",
        summary="Raise a credit limit.",
        description="Raises one customer's credit limit.",
        schema=arguments_schema({"key": {"type": "string"}}, ["key"]),
        run=run,
        writes=True,
    )


def server(**levels: str) -> Any:
    def build():
        return build_toolbox(Permissions(tools=levels), extra=[look_up(), raise_limit()])

    return build_server("demo", build)


def with_client(server: Any, work: Any) -> Any:
    """Run one piece of work against the server, over the protocol, in this process."""

    async def main() -> Any:
        async with Client(
            server, client_info=mcp_types.Implementation(name="test-agent", version="1")
        ) as client:
            return await work(client)

    return asyncio.run(main())


def test_the_tools_a_pack_allows_are_the_tools_offered():
    async def work(client):
        return await client.list_tools()

    result = with_client(server(look_up="read", describe_tool="read"), work)
    offered = {tool.name for tool in result.tools}
    assert offered == {"look_up", "describe_tool"}  # raise_limit has no level, so it is absent


def test_a_tool_carries_its_whole_manual_and_its_schema():
    async def work(client):
        return await client.list_tools()

    result = with_client(server(look_up="read"), work)
    tool = next(t for t in result.tools if t.name == "look_up")
    assert "The whole manual" in (tool.description or "")
    assert tool.input_schema["properties"]["what"]["type"] == "string"
    assert tool.annotations.read_only_hint is True
    assert tool.meta == {"io.vardex/permission": "read"}


def test_a_writing_tool_is_marked_as_one():
    async def work(client):
        return await client.list_tools()

    result = with_client(server(raise_limit="act"), work)
    tool = next(t for t in result.tools if t.name == "raise_limit")
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.destructive_hint is True


def test_the_tool_list_may_be_cached_and_is_the_same_for_everyone():
    async def work(client):
        return await client.list_tools()

    result = with_client(server(look_up="read"), work)
    assert result.ttl_ms == 300_000
    assert result.cache_scope == "public"


def test_calling_a_tool_returns_text_and_the_data_behind_it():
    async def work(client):
        return await client.call_tool("look_up", {"what": "revenue"})

    result = with_client(server(look_up="read"), work)
    assert result.is_error is False
    assert result.content[0].text == "It is revenue."
    assert result.structured_content == {"value": 42}


def test_a_tool_with_no_level_cannot_be_called_even_by_name():
    async def work(client):
        return await client.call_tool("raise_limit", {"key": "100000008"})

    result = with_client(server(look_up="read"), work)
    assert result.is_error
    assert "no permission level" in result.content[0].text


def test_at_draft_the_server_proposes_and_writes_nothing():
    WRITTEN.clear()

    async def work(client):
        return await client.call_tool("raise_limit", {"key": "100000008"})

    result = with_client(server(raise_limit="draft"), work)
    assert not result.is_error
    assert result.content[0].text.startswith("Proposed:")
    assert WRITTEN == []


def test_at_approve_the_server_asks_the_calling_agent_first():
    # The approval is not a question to the model: the protocol makes the client ask a person.
    WRITTEN.clear()
    asked: list[str] = []

    async def approve(context, params):
        asked.append(params.message)
        return mcp_types.ElicitResult(action="accept", content={"approved": True})

    async def main():
        async with Client(
            server(raise_limit="approve"),
            client_info=mcp_types.Implementation(name="test-agent", version="1"),
            elicitation_callback=approve,
        ) as client:
            return await client.call_tool("raise_limit", {"key": "100000008"})

    result = asyncio.run(main())
    assert asked == ["Raise the limit for 100000008 to 100."]
    assert result.content[0].text.startswith("Done:")
    assert WRITTEN == ["Raise the limit for 100000008 to 100."]


def test_at_approve_a_refusal_means_nothing_is_written():
    WRITTEN.clear()

    async def refuse(context, params):
        return mcp_types.ElicitResult(action="decline")

    async def main():
        async with Client(
            server(raise_limit="approve"),
            client_info=mcp_types.Implementation(name="test-agent", version="1"),
            elicitation_callback=refuse,
        ) as client:
            return await client.call_tool("raise_limit", {"key": "100000008"})

    result = asyncio.run(main())
    assert result.is_error
    assert "refused" in result.content[0].text
    assert WRITTEN == []


def test_an_agent_that_cannot_ask_a_person_does_not_get_the_change():
    # A client with no way to put a question to its user cannot answer the input request, and
    # the SDK raises rather than guessing. The important half is the last line: nothing moved.
    WRITTEN.clear()

    async def work(client):
        return await client.call_tool("raise_limit", {"key": "100000008"})

    # The SDK runs the call inside a task group, so the error arrives in a group.
    refused = pytest.RaisesExc(MCPError, match="Elicitation not supported")
    with pytest.RaisesGroup(pytest.RaisesGroup(refused)):
        with_client(server(raise_limit="approve"), work)
    assert WRITTEN == []


def test_the_server_says_who_it_is_without_a_handshake():
    # Since 2026-07-28 there is no initialize: identity rides every request and response.
    async def work(client):
        await client.list_tools()
        return client.server_info

    info = with_client(server(look_up="read"), work)
    assert info is not None and info.name == "vardex-demo"
