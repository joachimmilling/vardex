# 0029 Checkpoint runs to files, rather than adopt a durable-execution framework

Status: Accepted
Date: 2026-10-26

## Context

Work that runs on a schedule outlives the process that starts it. A weekly watch over a few
thousand companies makes dozens of model calls over several minutes, writes proposals a person
must answer, and will sooner or later be stopped half way: the budget runs out, the machine
sleeps, the provider is overloaded, somebody closes the laptop.

Decision 0024 built the agent loop by hand and said this was where to revisit it: "resumable
runs that outlive a process". This is that moment.

The field has good answers. Temporal, Restate and Inngest journal every completed step and
replay the function, returning recorded results instead of doing the work again; DBOS writes
state in a transaction at each step boundary; LangGraph checkpoints a graph after every
super-step and resumes from an `interrupt` with a stored value. All of them are more capable
than what is written here.

## Options

- A durable-execution engine (Temporal, Restate, Inngest, DBOS).
- LangGraph, with a Postgres checkpointer.
- A table in the pack's warehouse.
- One JSON file per run, written whole after every step, with a journal of finished work.

## Decision

A run is a file. `runs.py` writes the whole run — its steps, what it has spent, its journal
and the facts one step leaves for the next — through a temporary file after every step and
every case. Resuming loads the file and skips what is recorded. A lock file keeps two runs of
one workflow from overlapping.

The journal is keyed by the tool's name and its arguments, so a resumed run that asks for the
same change is handed the first answer back. That is the same mechanism the engines use, cut
down to the one guarantee that matters here: a writing call happens at most once per run.

## Consequences

- No new dependency, no server to run, nothing to learn before reading a run. A run's file is
  a few kilobytes of JSON that a person can open; `vardex runs` prints it.
- Durability stops at the filesystem. One machine, one disk. There is no distributed clock, no
  fan-out across workers, no exactly-once across processes, and no replay of a step whose
  inputs have changed.
- The unit that resumes is a step, and inside a step a case. A case that died half way is
  attempted again from the start of that case; the journal is what keeps its writes from
  happening twice.
- Checkpointing after every case costs one small file write per case. On a weekly run of a few
  dozen cases this is nothing. On thousands of items a run it would be the wrong design.
- The model is not replayed. A resumed case asks the model again, so it may reason
  differently and read fresher numbers. Only the writing is pinned.

## When to revisit

When runs need to survive the machine rather than the process: several workers sharing a
queue, a run that waits days for an approval, or fan-out wide enough that per-item retries
matter. Then the journal belongs in a database and the loop belongs in an engine that owns it,
and the step kinds here become the activities it schedules.
