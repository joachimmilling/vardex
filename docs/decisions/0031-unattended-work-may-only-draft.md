# 0031 Unattended work may only draft; the approval carries the change out

Status: Accepted
Date: 2026-10-26

## Context

Decision 0025 set four permission levels, and `approve` meant "does it, once a person has said
yes to this exact call". That was asked at the keyboard, by whoever started the run.

Scheduled work has no keyboard. The run happens at half past six on a Monday with nobody
there, which left a tool at `approve` with nothing to do but refuse — so the most useful thing
a monitor could do was tell nobody what it would have done.

Separating the question from the answer also raises a question the keyboard never did: where
is the record of who agreed to what, and what the proposal said at the time?

## Options

- Refuse, as before: an unattended run cannot act, full stop.
- Let a scheduled run act on its own, and review afterwards.
- Write the proposal into the other system as a draft record there.
- Draft into Vardex's own inbox, and have a separate human-run command carry it out.

## Decision

A `Toolbox` with no approver hands a writing call the rung below: the tool is invoked with
`draft`, so it writes a proposal instead of refusing or acting. The proposal goes into the
pack's inbox, an append-only JSON-lines log, with the sentence a person has to agree to, the
reason the run gave, and a deadline from the pack's `approvals` rules.

`vardex approve` then does three things in order: it checks the person is one the pack names,
it records the decision, and it asks the connector to make the change. The change therefore
happens in a different process, started by a human being, after a deliberate act.

Nothing reaches the other system before that — not even a draft record. The ladder is
unchanged; what changed is what a rung means when nobody is there.

## Consequences

- A monitor is useful without being trusted. It can run every week and the most it can do on
  its own is fill an inbox.
- A pack that allows an action only to `draft` cannot have its proposals carried out at all,
  by anybody, through any channel. `vardex approve` refuses and says so. That is the honest
  reading of `draft`, and it makes shipping a writing capability in two stages possible.
- The state of a proposal is a replay of its log, which is slower than a status column and
  gives the one answer auditors want: who, when, and what did it say then. Undo is another
  entry rather than an update, so a reversed change keeps its history.
- The inbox is a file, so two people approving the same proposal in the same second could both
  be recorded. The second `carry_out` would then make the change twice. A real installation
  puts this behind a database with a unique constraint on the reference.
- `approvers` is a list of names typed on the command line. It is honest on one person's
  machine and nowhere near enough for a company, where it belongs behind real identity.
- A proposal nobody answers expires rather than waiting for ever, because the figures it rests
  on go stale.

## When to revisit

When approvals have to happen where people already work — Teams, Slack, email — or when who
may approve what depends on the amount. The inbox becomes a small service with real identity
behind it, and these commands become one client of it.
