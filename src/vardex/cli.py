"""The vardex command-line tool."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import anthropic
import typer

from vardex import __version__
from vardex.config import load_settings
from vardex.llm import Answer, Effort, MissingAPIKeyError, count_tokens
from vardex.llm import ask as ask_model
from vardex.packs import PackError, load_pack
from vardex.pricing import (
    PRICES,
    UnknownModelError,
    Usage,
    cost_usd,
    format_usd,
    monthly_cost_usd,
)

app = typer.Typer(
    help="Vardex: blocks, recipes and packs for enterprise AI apps.",
    no_args_is_help=True,
)

ModelOption = Annotated[str | None, typer.Option(help="Use this model instead of VARDEX_MODEL.")]


def fail(message: str) -> typer.Exit:
    """Print an error in red and return the exit to raise."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def describe(answer: Answer) -> str:
    """One line of statistics about an answer."""
    out = f"{answer.usage.output_tokens:,} out"
    if answer.thinking_tokens:
        out += f" ({answer.thinking_tokens:,} thinking)"
    cost = format_usd(answer.cost_usd) if answer.cost_usd is not None else "cost unknown"
    return (
        f"{answer.model} · {answer.usage.input_tokens:,} in · {out}"
        f" · {answer.seconds:.1f} s · {cost}"
    )


@app.command()
def version() -> None:
    """Print the Vardex version."""
    typer.echo(__version__)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to ask.")],
    model: ModelOption = None,
    effort: Annotated[
        Effort | None, typer.Option(help="How much work the model puts in. Not on Haiku 4.5.")
    ] = None,
    stats: Annotated[bool, typer.Option("--stats", help="Show tokens, time and cost.")] = False,
) -> None:
    """Send a question to the model and print the answer."""
    settings = load_settings()
    if model:
        settings = replace(settings, model=model)
    try:
        answer = ask_model(question, settings, effort=effort)
    except MissingAPIKeyError as err:
        raise fail(str(err)) from err
    except anthropic.APIError as err:
        raise fail(f"The API refused the request: {err.message}") from err

    typer.echo(answer.text)
    if answer.truncated:
        typer.secho("Warning: the answer was cut off at max_tokens.", fg="yellow", err=True)
    if stats:
        typer.secho(describe(answer), fg="bright_black", err=True)


@app.command()
def tokens(
    file: Annotated[Path, typer.Argument(help="A text file.", exists=True, dir_okay=False)],
    model: ModelOption = None,
) -> None:
    """Count the tokens in a text file, and what sending it once costs."""
    settings = load_settings()
    if model:
        settings = replace(settings, model=model)
    text = file.read_text(encoding="utf-8")
    try:
        n = count_tokens(text, settings)
    except MissingAPIKeyError as err:
        raise fail(str(err)) from err
    except anthropic.APIError as err:
        raise fail(f"The API refused the request: {err.message}") from err

    words = len(text.split())
    try:
        cost = format_usd(cost_usd(settings.model, Usage(input_tokens=n, output_tokens=0)))
    except UnknownModelError:
        cost = "cost unknown"
    typer.echo(f"{file.name}  ({settings.model})")
    typer.echo(f"  tokens          {n:>8,}")
    typer.echo(f"  words           {words:>8,}")
    typer.echo(f"  characters      {len(text):>8,}")
    typer.echo(f"  tokens per word {n / max(words, 1):>8.2f}")
    typer.echo(f"  cost as input   {cost:>8}")


@app.command()
def cost(
    per_day: Annotated[int, typer.Option("--per-day", min=1, help="Requests per day.")],
    input_tokens: Annotated[int, typer.Option("--input", min=0, help="Input tokens per request.")],
    output_tokens: Annotated[
        int, typer.Option("--output", min=0, help="Output tokens per request, thinking included.")
    ],
    cached: Annotated[
        int, typer.Option(min=0, help="How many of the input tokens are read from the cache.")
    ] = 0,
    days: Annotated[int, typer.Option(min=1, help="Days per month.")] = 30,
) -> None:
    """Estimate the monthly cost of an LLM feature on every model in the price table."""
    if cached > input_tokens:
        raise fail("--cached cannot be larger than --input.")
    usage = Usage(
        input_tokens=input_tokens - cached,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
    )
    typer.echo(
        f"{per_day:,} requests a day for {days} days. Each request: "
        f"{input_tokens:,} input tokens ({cached:,} cached), {output_tokens:,} output tokens."
    )
    typer.echo(f"\n{'model':<20}{'per request':>14}{'per month':>14}")
    for model in sorted(PRICES, key=lambda m: cost_usd(m, usage)):
        per_request = format_usd(cost_usd(model, usage))
        per_month = format_usd(monthly_cost_usd(model, usage, per_day, days))
        typer.echo(f"{model:<20}{per_request:>14}{per_month:>14}")


@app.command()
def validate(
    folder: Annotated[Path, typer.Argument(help="A folder that contains a pack.yaml.")],
) -> None:
    """Check that a pack follows the pack format."""
    try:
        pack = load_pack(folder)
    except PackError as err:
        raise fail(str(err)) from err
    recipes = ", ".join(pack.recipes)
    typer.secho(f"OK  {pack.name} {pack.version}  recipes: {recipes}", fg=typer.colors.GREEN)
