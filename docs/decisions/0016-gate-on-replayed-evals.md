# 0016 Gate every pull request on a replayed eval, with a threshold and critical cases

Status: Accepted
Date: 2026-10-13

## Context

A prompt, a glossary entry or a change to the checks can make answers worse without failing a
single unit test. The change must be caught before it is merged. But a live eval in CI needs a
key and a warehouse, and its result moves by a few cases from run to run, so a gate on it would
sometimes fail a good change and pass a bad one.

## Options

- No gate: run evals by hand before a release.
- A live eval on every pull request, with a key from the repository's secrets.
- A replayed eval on every pull request: the author records a live run after any change to
  what the model is sent, and CI replays it and applies the gate.

## Decision

CI runs `vardex eval PACK --replay` on every push and pull request. It fails when a recording is
stale, when a suite's pass rate is below its `threshold`, or when a case marked `critical`
fails, such as a request to delete data that must always be refused. The threshold is set
below the measured pass rate by about the run-to-run noise, so it catches a real drop and not
chance. The report shows a 95% Wilson interval beside every rate.

## Consequences

- A change to a prompt cannot be merged without a new live run: the stale recording fails CI.
- The gate costs nothing to run and never flickers: a replay is the same every time.
- The recording is made by the author, so the gate trusts the author's run. A reviewer sees the
  new recording and the new report in the pull request.
- With 63 cases, the interval is several points either side. A gate catches a drop of a few
  cases or more; a smaller one needs more cases, or repeated runs.

## When to revisit

When deployments need a live gate of their own, against the model and warehouse in production,
or when the golden set is large enough to gate on each type separately.
