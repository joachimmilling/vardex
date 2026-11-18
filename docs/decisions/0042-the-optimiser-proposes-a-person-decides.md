# 0042 The prompt optimiser proposes; a person decides

Status: Accepted
Date: 2026-11-09

## Context

A small model fails work for reasons a sentence in the pack would fix: a word it misreads, a
pattern it has not seen. Programs such as
DSPy's optimisers and GEPA find such sentences by running a model against a metric and
letting a stronger model rewrite the prompt from the failures. They also find ways to pass a
test that teach nothing, such as copying the failing question into the examples.

## Options

- Tune the pack's files by hand.
- Use DSPy, which rewrites prompts it owns, in its own program structure.
- A small loop of Vardex's own over the pack file that instructs a capability, checked on
  held-out cases, whose result is a file a person reviews.

## Decision

`vardex optimise`: the golden set is split in two by a hash of each case's id. The teacher
sees failures from one half only, and rewrites the file that instructs the suite's
capability, never the engine's prompts: questions.yaml (glossary, style rules, examples) for
an answer suite, `extractions/<name>.yaml` (instructions, examples) for an extraction suite.
Each kind of file says how it is validated, which of its examples copy a golden input, and
how a failure is shown to the teacher. A rewrite is tried on a batch of that half and kept
when more pass; one that does not validate, or that has an example close to a golden input,
is refused before it is tried. The best rewrite runs once on the other half, and is
written only if it passes more there. It is never committed: a person reads the diff and
opens a pull request.

## Consequences

- Each round costs one teacher call and one batch of the small model's calls; with a model on
  a laptop, that is minutes per case.
- Code graders only (behaviour and rows, or behaviour and fields): a judge in the loop would be
  optimised against too.
- A held-out half of a golden set of sixty is small, so "better" is a few cases and McNemar's
  test is reported beside it.
- The rewrite changes what Claude is sent too: the suites must be recorded again and passed
  before the change is merged.

## When to revisit

When packs keep instructions per model, so a rewrite for a small model need not change what
the large one reads.
