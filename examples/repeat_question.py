"""Ask the same question several times and compare the answers.

Run with, for example:
    uv run python examples/repeat_question.py "Which Norwegian county has the most salmon farms?"
    uv run python examples/repeat_question.py "..." --model claude-haiku-4-5 --temperature 0

This script calls the API directly, because Vardex itself does not expose temperature:
models released after Claude Opus 4.6 reject any value other than the default.
"""

from collections import Counter
from typing import Annotated

import anthropic
import typer

from vardex.config import load_settings
from vardex.llm import SYSTEM_PROMPT, MissingAPIKeyError, make_client


def main(
    question: Annotated[str, typer.Argument(help="The question to repeat.")],
    runs: Annotated[int, typer.Option(min=2, help="How many times to ask.")] = 5,
    model: Annotated[str | None, typer.Option(help="Defaults to VARDEX_MODEL.")] = None,
    temperature: Annotated[float | None, typer.Option(help="Only older models accept it.")] = None,
) -> None:
    """Ask one question several times and count how many different answers come back."""
    settings = load_settings()
    model = model or settings.model
    try:
        client = make_client(settings)
    except MissingAPIKeyError as err:
        typer.secho(str(err), fg="red", err=True)
        raise typer.Exit(1) from err

    # The SDK no longer has a temperature argument, so it goes into the request body directly.
    extra_body = {"temperature": temperature} if temperature is not None else None
    answers = []
    for run in range(1, runs + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question}],
                extra_body=extra_body,
            )
        except anthropic.APIStatusError as err:
            typer.secho(f"The API refused the request: {err.message}", fg="red", err=True)
            raise typer.Exit(1) from err
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        answers.append(text)
        typer.echo(f"--- run {run} · {response.usage.output_tokens} output tokens\n{text}\n")

    distinct = Counter(answers)
    typer.echo(f"{model}: {len(distinct)} different answers in {runs} runs.")


if __name__ == "__main__":
    typer.run(main)