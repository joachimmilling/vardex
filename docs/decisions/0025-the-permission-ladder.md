# 0025 Four permission levels, declared by the pack, enforced in one place

Status: Accepted
Date: 2026-10-21

## Context

An agent that can only read is a reporting tool. The moment a tool can change something
outside Vardex, somebody has to say which tools, how far, and who agreed. The engine cannot
decide: the same tool is a draft in one company and an automatic action in another.

Model providers and MCP both offer hints about a tool's behaviour (`readOnlyHint`,
`destructiveHint`). They are hints, written by whoever wrote the tool, and MCP's own
specification tells clients to treat them as untrusted.

## Options

- A boolean: tools that read, tools that write.
- Levels declared in code, on each tool.
- Levels declared by the pack, in `permissions.yaml`, with no default.
- Trust the provider's or the server's annotations.

## Decision

Four levels, a ladder: `read`, `draft`, `approve`, `act`. The pack sets the level per tool in
`permissions.yaml`. A tool with no level cannot be called. `Toolbox.call` is the only place
that decides, so the CLI, the MCP server and the evals all get the same answer.

`draft` is the rung that matters: the tool runs, produces exactly what it would have done, and
writes it down for a person instead of doing it. A pack can ship a writing capability and turn
it on later by editing one line.

## Consequences

- Adding a tool to the engine grants it to nobody. Every pack must opt in, and `vardex check`
  fails a pack whose tools and permissions have drifted apart.
- A tool from an MCP server is governed the same way: the server's own annotations are
  recorded and ignored for the decision.
- Four levels is more to explain than two, and the difference between `approve` and `act` only
  means something once there is somewhere for a person to approve things. There is not yet.
- A refused call is a tool result the model reads, not an exception, so the agent can tell the
  person what it would have done.

## When to revisit

When permissions need to depend on who is asking rather than on the pack — a level per role,
or per customer. That changes where the level comes from, not the ladder itself.
