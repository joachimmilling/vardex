# 0024 Build the agent loop by hand, not on a framework

Status: Accepted
Date: 2026-10-21

## Context

Vardex needs an agent: a model that chooses tools, reads what they return and keeps going
until it can answer. Several frameworks do this, and the provider's own SDK has a tool runner
that drives the loop from a list of decorated functions.

The loop itself is about forty lines. What surrounds it is not: budgets that must hold, a
permission gate every call passes, a record of every step for the evals and for debugging, and
three channels that must behave identically. Those are the parts we are judged on.

## Options

- A framework (LangGraph, Pydantic AI, the Agent SDK). Less code to write; graph, memory,
  retries and tracing provided.
- The provider's tool runner. Least code; the loop is maintained by the provider.
- Our own loop over the existing `ModelClient`.

## Decision

Write the loop in `agent.py`, over `ModelClient`, with `Toolbox` deciding what may run.

The loop owns four things a general framework would own for us: when to stop, what goes back
to the provider untouched (signed reasoning blocks), how tool replies are batched into one
message, and what is recorded. Each of those is where our requirements live.

## Consequences

- No new dependency, and nothing between us and the provider's API to debug through.
- The loop is provider-neutral: it speaks `Request`, `Answer`, `ToolCall`, so a second
  provider needs a client, not a second loop. A framework would have been the only thing in
  Vardex that knew a provider's shape.
- We write our own retries, our own step cap and our own transcript, and we will write our own
  parallel tool execution when a run is slow enough to need it.
- A framework's community features — graph visualisation, checkpointing, hosted tracing —
  are ours to build or to do without.

## When to revisit

When the loop grows past a page or two of real logic: branching plans, several agents talking
to each other, or resumable runs that outlive a process. At that point a graph library earns
its place, and the loop here becomes one node in it.
