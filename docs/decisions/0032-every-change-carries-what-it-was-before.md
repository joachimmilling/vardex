# 0032 Every change carries what it was before, and a connector that cannot undo says so

Status: Accepted
Date: 2026-10-26

## Context

An approved change is made by a program against somebody else's system. Three things then go
wrong in practice: the change was approved on figures that were wrong; the change was right but
somebody wants it back; the other system was already changed by somebody else in between.

"Undo" is easy to promise and hard to mean. A connector that reverses a change by writing what
it guesses the old value was is worse than one that refuses: it makes a second unapproved
change, wearing the first one's reference.

## Options

- No undo. Make the change and leave it.
- Undo by re-running the opposite action, with the value a person types.
- Have the connector return what it changed *and what it was*, and undo from that.
- Write a compensating action for each action, as a saga would.

## Decision

`carry_out` returns a `Change`: the other system's own reference, the fields after, and the
fields *before*. `undo` is given that `Change` back. A connector that cannot say what the
values were leaves `before` empty, and the undo then restores nothing rather than inventing
something. A connector that cannot reverse a change at all sets `undoable = False`, and Vardex
refuses the undo and says what to do instead.

The undo also refuses when the current values are not the ones this change left behind.
Somebody has been there since, and putting "as you were" over their work is not an undo.

A dry run is the other half of the same idea: `--dry-run` runs every step and every tool with
writing turned off, so the run says what it would have changed and changes nothing.

## Consequences

- Undo is a property of the connector, not a promise of the engine. Reading `undoable` tells
  you the truth before you rely on it.
- `before` makes an undo exact for the fields the action sets. It says nothing about what else
  the other system did as a consequence — a letter it sent, a workflow it started. Those need a
  compensating action, which is the action a pack writes itself.
- The refusal when something has changed since is strict, and will sometimes block an undo a
  person would have wanted. The message names the change so they can decide and act on purpose.
- Every change and every undo is an entry in the inbox log, so the audit trail survives being
  reversed.
- The stand-in file connector keeps the current values itself, which is what lets `before` be
  real without a CRM. A real connector reads them from the system it talks to, in the same call
  if that system offers it.

## When to revisit

When a pack's actions have consequences the other system takes on its own. Then an undo is not
enough and each action needs its own named compensation, declared in the ontology beside it,
and the inbox needs to record which of the two it ran.
