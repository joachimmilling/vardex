# 0013 Grade answers with code first, and with a model only for the words

Status: Accepted
Date: 2026-10-13

## Context

An eval needs a grader for every case: something that says pass or fail. Text-to-SQL produces
three things that can be wrong in different ways: what it did (answered, asked back or
refused), the query it ran, and the words it wrote from the rows. Code can check the first two
exactly and cheaply. Code cannot tell whether a sentence such as "Mowi's margin was 15.6%"
says what the rows say; a person can, and a model can, at a price and with its own errors.

## Options

- Code only: compare what it did and the rows of its query with the case's reference.
- A model as judge only: ask a model whether the answer is right, given the reference.
- People only: someone reads every answer of every run.
- Code for what code can check, a model for the words, and people to check the model.

## Decision

Every case is graded by up to three graders behind one interface, `Grader`: a function of the
case, the outcome and the reference rows that returns a `Grade` or nothing. `grade_behaviour`
and `grade_rows` are code: the second compares result sets, ignores extra columns and their
names, compares whole numbers exactly and other numbers to six significant digits, and checks
order only when the case says it matters. `Judge` asks a model, at low effort, for a binary
verdict with its reasoning first, from the question, the reference SQL, the reference rows and
the answer. A case passes when every grader passes. The judge uses the same model client as
the run, so it is recorded and replayed with it.

## Consequences

- Most failures are found by code that is exact, free and never changes its mind.
- The judge can be wrong. Its verdicts on failing cases are read during error analysis, and a
  judge that disagrees with a person is fixed before its numbers are trusted.
- A judge of the same model may be lenient towards its own answers; the rows grader limits the
  harm, because a wrong query fails whatever the judge says.
- A pass/fail verdict loses nuance a scale would keep, and gains agreement between runs and
  between people.

## When to revisit

When a cheaper model agrees with people as often as the main model does on the judge's task,
or when answers grow beyond what reference rows can settle, such as explanations of documents.
