# 0030 The run decides whether it is due, from the last run that finished

Status: Accepted
Date: 2026-10-26

## Context

Scheduled work has to be started by something. All the usual somethings miss: launchd does not
fire while a laptop is asleep and does not catch up afterwards; a hosted CI schedule is queued
at the top of the hour and sometimes dropped, with a five-minute floor on how often it may be
asked; a container restarts mid-run; a clock goes back an hour twice a year, so the same wall
time happens twice.

A job that does the work whenever it is told therefore runs the wrong number of times: twice in
the hour that repeats, not at all on the Monday the machine was off, and twice when a person
also runs it by hand.

## Options

- Trust the tick: whatever fires it, run.
- Keep a cron expression in the pack and evaluate it ourselves.
- Compute the last moment the workflow should have run, and run only if no finished run
  has happened since.

## Decision

The pack declares its schedule in words — `every: week`, `weekday: monday`, `at: "06:30"`,
`timezone: Europe/Oslo` — and `Schedule.due_before(now)` gives the last moment it should have
run. `is_due` is then one comparison: has a run that *finished* started since that moment?

Whatever fires Vardex calls `vardex run PACK --if-due` often — every fifteen minutes is fine —
and Vardex decides. A missed slot is picked up at the next call. The hour that happens twice
asks for the same slot both times, and the second time the work is already done.

A time zone is a place, not an offset, so "half past six" means half past six in both halves
of the year.

## Consequences

- The scheduler becomes replaceable and uninteresting: launchd, cron, a CI schedule, a person,
  anything that can run a command on a timer.
- Firing twice is harmless, which is the property that makes the whole thing safe to automate.
- A slot that is missed entirely is not run twice to catch up. The next run reads the feed from
  its cursor, so nothing is lost, but there is no separate run per missed week.
- A run that failed leaves the workflow due, so the next tick tries again. A run that is
  `stopped` on its budget also leaves it due, which is right: the work was not done.
- `every: hour` with a scheduler that only fires every fifteen minutes runs up to fifteen
  minutes late. For anything finer than that, this is the wrong mechanism.
- Only three cadences exist. A pack that needs "the first Tuesday of the month" cannot say so.

## When to revisit

When a pack needs a cadence that words cannot express, or when something must run within
seconds of an event rather than at a time. The first wants a cron expression parsed into
`due_before`; the second wants the event to start the run, with the feed as its trigger.
