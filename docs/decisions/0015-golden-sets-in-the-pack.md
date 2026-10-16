# 0015 Keep golden sets in the pack, with reference SQL instead of reference answers

Status: Accepted
Date: 2026-10-13

## Context

What a right answer is depends on the domain: which accounts count, what "last year" means,
which companies are seafood companies. The engine knows none of this. The data also moves: the
register changes every night, and a number that was right on Monday is wrong on Tuesday. A
golden set must survive both.

## Options

- Golden sets in the engine's tests, with the expected numbers written in.
- Golden sets in the pack, with the expected numbers written in.
- Golden sets in the pack, each case with the SQL that answers it, checked by hand, or with the
  behaviour expected instead: asking back or refusing.

## Decision

A pack keeps its golden sets in `evals/<suite>.yaml`. A case has an id, a type, a language,
the question, and either reference SQL or `expect: clarify` or `expect: refuse`. The reference
SQL runs on the same warehouse at the same time as the answer, so the two always see the same
data. Each suite fixes `today`, so "last year" means the same in every run. A case must not
repeat an example from `questions.yaml`, and its SQL must pass Vardex's own SQL checks; both are
checked before every run and in the pack's tests.

## Consequences

- The engine runs any pack's golden sets; only the cases and the graders' inputs differ.
- References stay right as the data changes; only a change to the meaning of a table breaks one.
- A case is only as right as the person who wrote its SQL. Every case is checked by running it
  and reading the rows before it is added.
- A golden set written by the same people who tune the prompts will flatter them. New cases
  should come from questions people really asked.

## When to revisit

When a pack's answers come from documents rather than tables, and the reference must be a
passage or a set of facts instead of a query.
