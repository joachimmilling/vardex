# 0040 A judge chosen per suite, and checked against people

Status: Accepted
Date: 2026-11-09

## Context

The answer and cite suites end with a model judge (0013). It was the model under test: Claude
judged Claude's answers, and an open model would have judged its own. Models favour text
they wrote, and a weak model is a weak judge. Nothing measured whether the judge agreed with a
person at all.

## Options

- Keep judging with the model under test.
- Always judge with the strongest model available.
- Name the judge in each suite (`judge:`), override it with `--judge`, warn when it is the
  model under test, and check it against verdicts a person made by hand.

## Decision

The third. A judge is a model, `provider/model`, or a judge plug-in: any grader with the
grader signature, such as a decision model that is not a chat model. Labels a person makes
with `vardex label` are kept in the pack's `evals/labels/`; `vardex eval --check-judge`
runs the judge on those answers and reports agreement and Cohen's kappa, which discounts the
agreement a judge that passes everything would get by chance. The pack's answer and cite
suites are judged by Claude Opus 5.5.

## Consequences

- The judge stays the same when the model under test changes, so two models' rates compare.
- Changing the judge changes the recorded calls: the suites were recorded again.
- A plug-in judge is called live; a suite judged by one cannot be replayed.
- The labels are only as good as the person who made them, and only as many: a kappa from
  twenty labels has a wide interval.

## When to revisit

When the labels show a judge's failures cluster in one type of question: then the judge
prompt, or the judge, is what to change for that type.
