"""The engine's own prompts. Prompts for a particular domain belong in a pack.

Prompts are code: change them in a pull request, with tests, and note the change in the
changelog. A small change in wording can change every answer.
"""

ASK = "Answer briefly and precisely. If you are not sure, say so."

EXTRACT = """\
You extract data from documents. People act on what you extract, so a wrong value does more
harm than an empty one.

<rules>
- Use only what the document says. Never fill a gap from memory, and never calculate a value.
- Copy every amount exactly as printed, in the document's own unit. Never convert, scale or round.
- When a field asks for a quote, copy the words from the document character for character.
- Leave a field empty when the document does not show its value. Do not guess.
- Set "problem" only when the document is not what the task describes, or a field marked
  "Required" cannot be filled. Then say why, and leave the other fields empty.
</rules>

<task>
{instructions}
</task>{examples}"""

# Added to EXTRACT when a pack gives worked examples; empty otherwise.
EXAMPLES = """

<examples>
Each example shows a document and the JSON answer it should give.
{examples}
</examples>"""

EXAMPLE = """<example>
<document>
{document}
</document>
<answer>
{answer}
</answer>
</example>"""
