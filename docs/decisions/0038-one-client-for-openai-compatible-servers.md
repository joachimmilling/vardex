# 0038 One client for every OpenAI-compatible server

Status: Accepted
Date: 2026-11-09

## Context

Vardex should run on open-weight models: on a laptop for development, on GPUs a company
rents in Norway, and through hosted providers. Ollama, LM Studio, vLLM, SGLang and most hosted
providers of open models all speak OpenAI's Chat Completions API, each with its own gaps:
Ollama reads `max_tokens` and ignores `max_completion_tokens`, its Homebrew build has no
structured outputs for models it runs on MLX, and none of them caches prompts as Anthropic's
API does.

## Options

- One client per server, each with its own SDK.
- One client for the Chat Completions API, configured by address and key.
- A library that wraps many providers behind one call, such as LiteLLM.

## Decision

`openai_client.OpenAIClient`, on the `openai` SDK, behind two entry points: `openai` (any
server, at `OPENAI_BASE_URL`) and `ollama` (the same client at Ollama's address, no key, no
price per token, a ten-minute timeout). It sends `max_tokens`, maps Vardex's five efforts onto
three, sends strict JSON schemas, and reads reasoning under either name servers use. A
server's own options go in `Request.extra`, passed through untouched, as Anthropic's client
now passes them too.

## Consequences

- Every server that speaks the API is reachable with no new code: only the address changes.
- What a server does not support, it may ignore rather than refuse. A request that relies on
  structured outputs must be checked on each server; Vardex validates every answer anyway.
- There are no cache breakpoints: a server caches as it caches, and the cost figures say how
  much it read from its cache when it reports it.
- Counting tokens costs a one-token call, since the API has no way to ask.

## When to revisit

When a server Vardex needs speaks only OpenAI's newer Responses API, or a feature Vardex
depends on (tool calls with reasoning sent back) works only there.
