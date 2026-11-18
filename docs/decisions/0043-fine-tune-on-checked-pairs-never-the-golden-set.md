# 0043 Fine-tune on checked pairs, never on the golden set

Status: Accepted
Date: 2026-11-18

## Context

A small open model, fine-tuned on a pack's own work, may do it better than the same model
prompted: write a pack's SQL, read its documents. A fine-tune needs pairs of inputs and right
outputs. There are few approved results yet, the golden set is the obvious and forbidden source, and a provider's
terms usually forbid training a model on its model's output.

## Options

- Train on the golden set.
- Train on Claude's production output.
- Train on work people approved, plus the pack's own examples and, where it is safe, pairs
  generated from them, each checked by the capability's own checks; nothing the golden set
  holds.

## Decision

The third, built by `vardex train`. Each capability has a lesson (`lessons.py`) that says
where its pairs come from, which it leaves out, and how it writes them; the dataset, the
golden rule and the trainer are shared. For the SQL writer, a generated pair swaps a value of an ontology label or
values column for another from the warehouse, or shifts its years, and is kept only if its
query passes `check_sql` and returns rows. A pair whose question is within 0.95 of a golden
question, or whose query is a golden reference query, is left out; a test proves no golden
question gets through. For an extraction (`--extraction`), the pairs are its examples and the
documents people approved whose values pass the checks; none are generated, since a document
with a changed figure is a document that does not exist, and a golden document is left out.
Pairs are written as the exact chat Vardex sends, with the right output as the answer. Training is a plug-in; the first is LoRA with mlx-lm on Apple silicon. An adapter
replaces the prompted model only if it passes more of the golden set, case by case.

## Consequences

- No model writes the training answers, so no provider's terms on its output apply, and no
  model's mistakes are taught. A person's approval is the other source; it grows with use.
- Generated pairs are only as varied as the examples: the model learns the pack's patterns,
  not new ones, and no refusals or questions back, which the examples do not hold.
- Swapped names include people's names where the register holds sole proprietorships: the
  dataset is personal data, kept in the pack's data folder and never committed.
- Training prompts must match what is served, word for word: a change to the engine's prompt
  makes an adapter stale.
- An adapter is served for every call a capability makes, not only the one it was trained on.
  A LoRA adapter for Gemma 4 E4B, trained on 126 plan pairs, wrote plans where the second call
  wanted a sentence, and never refused or asked back: 13 of 66 answer cases passed, against 47
  for the same model prompted. Its validation loss was 0.003. The dataset has to cover every
  call it will answer, refusals and questions back included, before an adapter is worth testing.
- Variants of one example are split to the same side, so the validation loss is not measured on
  near-copies of the training pairs.

## When to revisit

When there are some hundreds of approved answers: then they, not generated pairs, are the
dataset, and preference training on approved against rejected answers becomes possible.
