# 0003 Store money as Decimal, never float

Status: Accepted
Date: 2026-10-01

## Context

Vardex adds up API costs in fractions of a cent and reads amounts from annual reports and
registers. A binary float cannot hold most decimal fractions exactly: 0.1 + 0.2 is not 0.3, and
the error grows as values are added. Amounts that do not add up are not trusted.

## Options

- `float`, and round when printing.
- Integers of the smallest unit, such as micro-dollars or øre.
- `decimal.Decimal` in Python and exact decimal types in the database.

## Decision

Every amount of money is a `Decimal` in Python, written as a string in JSON so no reader turns
it into a float. Prices live in `pricing.py` with the date they were checked. In the warehouse,
amounts are `DECIMAL` columns, never `DOUBLE`.

## Consequences

- Sums and comparisons are exact, and tests can compare amounts with `==`.
- Every boundary needs care: JSON numbers, model output and SQL drivers may hand over floats,
  which must be converted from their text, not from the float.

## When to revisit

Not expected to change. If a library forces floats at a boundary, convert at that boundary.
