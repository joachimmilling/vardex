# 0028 Speak MCP as a server and as a client, and govern both with the pack

Status: Accepted
Date: 2026-10-21

## Context

Most people will use Vardex through an assistant they already have, not through a Vardex
screen. The Model Context Protocol is how those assistants reach tools, and since the
2026-07-28 revision it is a stateless request/response protocol with a defined way for a
server to ask a question back mid-call.

The same protocol runs the other way: the tools a company already exposes — GitHub, Slack,
Drive, a wiki — are MCP servers Vardex's own agent could use.

## Options

- Server only: let other agents in, but write every connector ourselves.
- Client only: use other people's tools, and keep Vardex behind its own CLI.
- Both, with outside tools treated as ours.
- Both, with outside tools governed by the pack like any other tool.

## Decision

Both directions, in two modules: `mcp_server.py` offers the pack's tools, `mcp_client.py`
uses tools from the servers a pack lists in `mcp.yaml`. A tool from outside arrives named for
its server (`notes__search_notes`), needs a level in `permissions.yaml` like every other tool,
and its calls count against the same budget.

Approval travels on the protocol: a tool at `approve` answers `input_required` with an
elicitation, and the calling agent must put the question to its user before it can retry. We
do not ask the model for permission.

## Consequences

- Vardex is useful from a chat window on day one, with no UI of our own.
- A server's claim about its own tools (`readOnlyHint` and the rest) is recorded and never
  trusted; the pack decides. The specification tells clients to do exactly this.
- A tool description from outside is text we did not write, so it is passed to the model
  marked as information from outside, never as instructions.
- MCP is asynchronous and Vardex is not, so each session runs on its own thread with its own
  event loop. That is a real boundary, and it is one class wide.
- Only the variables a pack names are passed to a server it starts: a server Vardex launches
  has no business reading the model's API key.

## When to revisit

When the context cost of many servers' tool definitions outgrows the progressive disclosure in
`describe_tool` — then a tool search or a code-execution channel, where the agent writes code
against the tools instead of calling them one at a time, is the next step.
