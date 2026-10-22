"""Vardex as an MCP server: the same tools, offered to somebody else's agent.

Most people will never open a Vardex screen. They will ask the assistant they already use —
Claude, ChatGPT, Copilot, their editor — and it will reach Vardex through the Model Context
Protocol. So MCP is not an export feature; it is a main entrance, and it gets the same gate
as every other one.

This module is a translation layer and nothing else. It holds no permission logic, no SQL and
no retrieval: it turns `Toolbox` into MCP's vocabulary and back. If you find yourself adding a
rule here, it belongs in the Toolbox, where the CLI will get it too.

Two parts of the protocol are worth seeing:

- Since the 2026-07-28 revision MCP is stateless. There is no handshake and no session: every
  request carries its own protocol version, client identity and capabilities, and a server
  must not remember anything between them. That is why the toolbox is built per request here.
- A tool that needs a person's approval does not ask the model. It answers `input_required`
  with an elicitation, and the calling agent is obliged to put the question to its user and
  call again with the answer. The approval ladder is therefore not a Vardex invention bolted
  onto the protocol; it is how the protocol already works.

This is the only module that imports the MCP SDK's server side. A test enforces it.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import mcp.types as mcp_types
from mcp.server import InitializationOptions, ServerRequestContext
from mcp.server.lowlevel import Server

from vardex import __version__
from vardex.llm import ToolCall
from vardex.permissions import Permission
from vardex.tools import Tool, Toolbox

APPROVAL = "approval"  # the key the elicitation goes out and comes back under
LIST_TTL_MS = 300_000  # how long a client may cache the tool list: five minutes
# Vardex's permission levels, as MCP's hints about a tool's behaviour. The hints are advice
# to the client, not a control: a client is told to treat them as untrusted, and the real
# decision is made here, by the pack. They are set so a trusting client behaves sensibly.
ANNOTATIONS: dict[Permission, mcp_types.ToolAnnotations] = {
    "read": mcp_types.ToolAnnotations(read_only_hint=True, destructive_hint=False),
    "draft": mcp_types.ToolAnnotations(read_only_hint=False, destructive_hint=False),
    "approve": mcp_types.ToolAnnotations(read_only_hint=False, destructive_hint=True),
    "act": mcp_types.ToolAnnotations(read_only_hint=False, destructive_hint=True),
}


def as_mcp_tool(tool: Tool, permission: Permission) -> mcp_types.Tool:
    """One Vardex tool as MCP describes it. The whole manual goes in the description here:
    an MCP client caches the list and discloses it its own way, so there is nothing to save
    by sending only the summary."""
    annotations = ANNOTATIONS[permission].model_copy(update={"title": tool.name})
    return mcp_types.Tool(
        name=tool.name,
        title=tool.name,
        description=f"{tool.summary}\n\n{tool.description}",
        input_schema=tool.schema,
        annotations=annotations,
        meta={"io.vardex/permission": permission},
    )


def approval_request(tool: Tool, call: ToolCall, sentence: str) -> mcp_types.InputRequiredResult:
    """Ask the calling agent to put one question to its user, and call again with the answer."""
    schema = {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "True to carry this out, false to refuse it.",
            }
        },
        "required": ["approved"],
    }
    request = mcp_types.ElicitRequest(
        params=mcp_types.ElicitRequestFormParams(message=sentence, requested_schema=schema)
    )
    return mcp_types.InputRequiredResult(
        input_requests={APPROVAL: request},
        # The protocol has no session, so what the server needs on the retry travels with the
        # question and comes back with the answer. Nothing is remembered here.
        request_state=json.dumps({"tool": call.name, "arguments": call.arguments}),
    )


def approved_in(params: mcp_types.CallToolRequestParams) -> bool | None:
    """Whether the person said yes, when this call is the retry of an approval. None when no
    answer came with it, which means the question has not been put yet."""
    answers = params.input_responses or {}
    answer = answers.get(APPROVAL)
    if not isinstance(answer, mcp_types.ElicitResult):
        return None
    if answer.action != "accept":
        return False
    content = answer.content or {}
    return bool(content.get("approved"))


def as_mcp_result(content: str, is_error: bool, structured: Any = None) -> mcp_types.CallToolResult:
    """A Vardex result as MCP's. The text is what a model reads; the structured value is for
    a client that would rather have the data."""
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=content)],
        structured_content=structured,
        is_error=is_error,
    )


@dataclass
class VardexServer:
    """An MCP server over one pack's toolbox.

    `build` is called per request, not once, because MCP is stateless and because a warehouse
    connection or an index handle should not be held open between calls from a client that may
    never come back.
    """

    pack: str
    build: Callable[[], Toolbox]  # a fresh toolbox, with its connections

    def tools(self) -> list[mcp_types.Tool]:
        box = self.build()
        return [
            as_mcp_tool(tool, level)
            for name, tool in box.tools.items()
            if (level := box.level(name)) is not None  # a tool with no level is not offered
        ]

    def call(self, params: mcp_types.CallToolRequestParams) -> mcp_types.CallToolResult | (
        mcp_types.InputRequiredResult
    ):
        box = self.build()
        call = ToolCall(id=params.name, name=params.name, arguments=dict(params.arguments or {}))
        tool = box.tools.get(params.name)
        level = box.level(params.name)
        if tool is not None and level == "approve" and tool.writes:
            answered = approved_in(params)
            if answered is None:
                # Say what is about to happen in one sentence, and stop. The client must put
                # it to a person: the protocol gives it no other way to continue.
                sentence = _sentence(box, call, tool)
                return approval_request(tool, call, sentence)
            box.approve = lambda _call, _tool, given=answered: given
        result = box.call(call)
        return as_mcp_result(result.content, result.is_error, result.structured)


def _sentence(box: Toolbox, call: ToolCall, tool: Tool) -> str:
    """The one sentence a person is asked to approve: what the tool would do, if it can say
    so without doing it, and a plain description of the call if it cannot."""
    box.dry_run = True
    try:
        dry = box.call(call)
    finally:
        box.dry_run = False
    if not dry.is_error and dry.structured and "sentence" in dry.structured:
        return str(dry.structured["sentence"])
    arguments = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return f"Allow Vardex to run {tool.name}({arguments})?"


def build_server(pack: str, build: Callable[[], Toolbox], *, instructions: str = "") -> Server[Any]:
    """An MCP server offering one pack's tools."""
    vardex = VardexServer(pack, build)

    async def on_list_tools(
        context: ServerRequestContext[Any], params: mcp_types.PaginatedRequestParams | None
    ) -> mcp_types.ListToolsResult:
        # The list is the same for every caller and changes only when the pack does, so it is
        # worth caching; a client that caches it also keeps its own prompt cache warm.
        return mcp_types.ListToolsResult(
            tools=vardex.tools(), ttl_ms=LIST_TTL_MS, cache_scope="public"
        )

    async def on_call_tool(
        context: ServerRequestContext[Any], params: mcp_types.CallToolRequestParams
    ) -> mcp_types.CallToolResult | mcp_types.InputRequiredResult:
        return vardex.call(params)

    return Server(
        f"vardex-{pack}",
        version=__version__,
        title=f"Vardex: {pack}",
        instructions=instructions
        or (
            f"Answer questions about the {pack} pack. Call list_tables before writing SQL and "
            "search_documents before quoting a report. Every number must come from a tool "
            "result; never calculate one yourself."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def serve_stdio(server: Server[Any]) -> None:
    """Run the server on standard input and output, which is how a desktop agent starts one."""
    import anyio
    from mcp.server.stdio import stdio_server

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, initialization_options(server))

    anyio.run(main)


def initialization_options(server: Server[Any]) -> InitializationOptions:
    """What the server says about itself when a client asks."""
    return server.create_initialization_options()


def tool_handlers(
    server: Server[Any],
) -> tuple[Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]]:
    """The server's two tool handlers, for tests that call them without a transport."""
    return server.get_request_handler(mcp_types.ListToolsRequest), server.get_request_handler(
        mcp_types.CallToolRequest
    )
