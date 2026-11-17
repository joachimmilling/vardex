# 0041 Shadow mode runs after the live result, on the same work

Status: Accepted
Date: 2026-11-09

## Context

A golden set measures a candidate model on the cases someone wrote down. Before a candidate
does work for people, it should do the work people actually bring (their questions, their
documents), with nobody seeing the result, and the two models' results should be compared.

## Options

- Replay production work from the traces to the candidate, later, in a batch.
- Give the work to both models at once, and show whichever is the live one.
- Give the candidate the same work after the live result has been given, with the same
  settings, and keep both.

## Decision

The third, with `VARDEX_SHADOW_MODEL` or `--shadow-model` on `vardex answer` and `vardex
extract`. Shadow mode knows no capability: each one says, through `capabilities.py`, how to do
its work with a given model and how two results differ. For answers that is the same kind of
reply and the same rows; for extractions, the same value in every field. The candidate's
result is never printed, never acted on and never raised: a failing candidate is recorded as
failing. Each piece of work goes as one line to `data/<pack>/shadow.jsonl`, with the
capability, what each model did, its time and its cost; `vardex shadow` reports how often they
agree, with a Wilson interval, and everything on which they differ.

## Consequences

- The person waits for nothing, but the process does: a command-line answer exits only after
  the shadow is done. A server would run it in the background.
- The shadow costs what the candidate costs; on a local model that is time and electricity.
- Agreement is not correctness: the live model is not the reference. The questions on which
  they differ are what a person reads, and what becomes cases.
- Answers and extractions are shadowed. An agent in shadow must run with every writing tool
  on a dry run, so a candidate never acts; that is one more capability adapter.

## When to revisit

When answers are served by a web API: the shadow belongs in a background task there, with a
sample rate so a busy service shadows a share of its traffic rather than all of it.
