# 0002 Talk to models through one provider-neutral interface

Status: Accepted
Date: 2026-10-01

## Context

Vardex calls language models for answers and extraction. The first provider is Anthropic, but
customers may need another vendor, a model hosted in Norway, or one running on their own
machines. Code that uses one vendor's SDK everywhere would have to change in every module.

## Options

- Call the vendor's SDK wherever a model is needed.
- Use a third-party library that wraps many providers.
- Define Vardex's own small interface, `ModelClient`, with one client module per provider.

## Decision

The engine speaks only Vardex's own types: `Request`, `Answer`, `StopReason` and `ModelError`
in `vardex.llm`. `AnthropicClient` in `anthropic_client.py` is the only module that imports
`anthropic`; it translates requests, stop reasons and errors. `cli.connect` is the one place
that chooses a provider. A test enforces the import rule.

## Consequences

- A second provider is one new module and one line in `connect`.
- Tests run the whole engine with a small fake client and never touch the network.
- A provider feature appears in Vardex only when it fits the interface; some quirks, such as a
  model that rejects a setting, are absorbed inside the client.

## When to revisit

If a needed feature exists at only one provider and cannot be expressed in Vardex's own terms,
decide whether to add it to the interface or keep it out.
