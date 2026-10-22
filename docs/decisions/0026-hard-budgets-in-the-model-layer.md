# 0026 Enforce spending caps in the model client, on the worst case

Status: Accepted
Date: 2026-10-21

## Context

An agent decides how many model calls to make. A loop that cannot make progress will keep
trying, and a question that should cost two cents can cost five dollars. Providers offer a
token budget the model is shown and asked to pace itself against; it is advice, and the model
may ignore it.

## Options

- A step cap only.
- The provider's advisory token budget.
- A cap checked after each call, on what has been spent.
- A cap checked before each call, on the worst that call could cost.

## Decision

`Budget` wraps a `ModelClient` and refuses a call whose worst case — every input token plus
the whole of `max_tokens` as output — would take the total past the cap. Caps come from the
pack's `permissions.yaml`: one per run, one per day. The day's spending is kept in a file
beside the pack's data, so the cap holds across commands and channels.

The input tokens are estimated locally, pessimistically, at two characters per token. The cap
then costs nothing, adds no round trip, and still holds when the provider's token counter is
unreachable.

## Consequences

- Everything in Vardex reaches a model through a `ModelClient`, so there is no path that
  escapes the cap: not a prompt, not a tool, not a new channel.
- Pricing the worst case refuses some calls that would have fitted. That is the price of a cap
  that cannot be stepped over, and the message says how much was left.
- The estimate is deliberately high, so the cap bites a little early. `vardex tokens` counts
  exactly when it matters.
- The day's record is a file, so two commands running at the same moment can both read it
  before either writes. The error is at most one call's worth, and the step cap bounds that.
- A model with no price in `pricing.py` cannot be counted; those calls are tallied separately
  rather than treated as free.

## When to revisit

When several machines share a budget, or when a cap has to be per user rather than per pack.
Then the record belongs in the warehouse or a small service, not in a file.
