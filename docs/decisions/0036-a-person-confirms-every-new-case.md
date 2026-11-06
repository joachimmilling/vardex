# 0036 Feedback makes candidates; a person makes cases

Status: Accepted
Date: 2026-10-28

## Context

The golden set is how a change is judged: a pull request that lowers its score does not
merge. It should hold the questions people actually ask, including the ones Vardex gets
wrong, and production is where those are found. But a thumbs down says only that somebody
was unhappy, and the SQL in the trace is, by the same token, probably wrong. A case whose
reference is wrong makes every later run look worse for being right.

## Options

- Add every flagged answer to the golden set automatically, with the SQL that ran.
- Keep feedback as a number on the dashboard, and write cases by hand.
- Turn flagged answers, and a random sample of ordinary ones, into candidates that wait,
  and add one to the golden set only when a person has seen the rows of its reference.

## Decision

`vardex feedback --bad` records the verdict and makes a candidate: the trace's question, the
SQL that ran, the answer given, and the day it was asked. `vardex sample` adds randomly
chosen answered questions as candidates, because people report only some failures.
`vardex promote` shows the reference's rows — the person's own SQL, or Vardex's if they agree
with it — and asks for a yes at the keyboard (or `--yes`), then adds the case through
`evals.add_case`, which still checks it. `vardex dismiss` closes one with a reason. Every
step is a line appended to a file, so it can be read back.

## Consequences

- No case enters the golden set without a person who looked at its rows.
- The set grows from real traffic, in the words people use, including Nynorsk.
- Promoting changes a suite, so its recording must be made again; the command says so.
- A question that depends on its date ("last year") is checked against the suite's `today`,
  and the person is warned when the years differ.

## When to revisit

When the candidates outnumber the people who review them: then rank them, by how often a
question is asked or how sure a judge is that the answer was wrong, without letting a judge
decide alone.
