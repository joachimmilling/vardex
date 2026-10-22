# 0027 Three channels over one set of engine functions

Status: Accepted
Date: 2026-10-21

## Context

People will reach Vardex in at least three ways: a person at a terminal, an assistant they
already use (Claude, ChatGPT, Copilot) that speaks MCP, and other programs over HTTP. The
easy mistake is to grow a second implementation behind the second entrance, and a third
behind the third, each with its own idea of what is allowed.

## Options

- The CLI as the only entrance, and a wrapper script for anything else.
- One implementation per channel.
- One set of engine functions, and thin channels over them.

## Decision

The engine is `Toolbox` and `run_agent`. A channel may build a toolbox, call it, and format
what comes back. It may not decide what a tool is allowed to do, what a budget is, or what an
error means. An architecture test fails a channel that looks as though it does.

The CLI gets what a program needs as well as what a person needs: `--json` for the whole run
as data, an error code in that JSON for a program to branch on, and `--dry-run` on anything
that writes. A coding agent can then drive Vardex without a person reading the output.

## Consequences

- A rule written once holds in every channel. The permission tests in `test_tools.py` and the
  protocol tests in `test_mcp_server.py` check the same decisions from two sides.
- The MCP server is a translation layer of about a hundred lines, and an HTTP channel should
  be about the same.
- Each channel still needs its own tests, because each has its own way of being wrong: the
  CLI's exit codes, MCP's input-required round trip.
- Keeping logic out of channels sometimes means pushing something into the engine that only
  one channel needs, such as the dry run.

## When to revisit

If a channel needs something the engine genuinely cannot express — streaming partial results
to a browser, say — before bending the engine to fit it.
