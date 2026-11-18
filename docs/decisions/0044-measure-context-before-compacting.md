# 0044 Measure an agent's context before compacting it

Status: Accepted
Date: 2026-11-18

## Context

An agent sends its whole conversation again on every step. On a model run on our own server,
the window is a setting (32,768 tokens in the Ollama we run), and a conversation longer than
it is not refused: Ollama drops the oldest messages and answers without them. Compaction
(replacing old tool results with a summary) and handles (keeping a large result out of the
conversation, behind a reference) both keep the context short, and both add a moving part that
can drop exactly the detail the next step needs.

## Options

- Compact from the start, on every model.
- Keep large results behind handles, with a tool to read parts of them.
- Measure how long the context grows on the agent suite, for each model we run, and build
  either only when a run comes near the window or the budget.

## Decision

Measure first. The agent suite reports, for every run, the prompt each step sent (cached or
not). On the suite as recorded with Claude Sonnet 5.5, the first step sends a median of 2,223
tokens; a run grows by a median of 1,371 tokens a step; the largest prompt of any run is 20,192
tokens, under seven steps. We have not yet run the agent suite on a local model. That is within the 32,768-token window we give a local
model, with room for the answer, and far within Claude's. No compaction is built.

## Consequences

- No summary can lose a tool result the answer needs, because none is made.
- The largest run uses about two thirds of a local window: a question that makes twice as many
  calls, or a tool that returns twice as much, would not fit. Ollama would not say so.
- The agent's step cap (12) bounds the growth, at about 18,000 tokens of growth beyond the
  first step at the median rate.

## When to revisit

When the largest prompt of a suite run on a model we serve passes half its window, when a tool
starts returning results of thousands of tokens (whole documents, large query results), or when
the agent's step cap is raised.
