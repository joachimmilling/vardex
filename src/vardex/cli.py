"""The vardex command-line tool."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from vardex import __version__
from vardex.anthropic_client import AnthropicClient
from vardex.calllog import log_call
from vardex.config import Settings, load_settings
from vardex.extraction import ExtractionError, build_request, extract, load_extraction
from vardex.llm import Answer, Effort, ModelClient, ModelError, Request
from vardex.llm import ask as ask_model
from vardex.packs import PackError, load_pack
from vardex.pricing import (
    PRICES,
    UnknownModelError,
    Usage,
    cost_usd,
    format_usd,
    monthly_cost_usd,
    price_for,
    stale_prices_warning,
)

app = typer.Typer(
    help="Vardex: blocks, recipes and packs for enterprise AI apps.",
    no_args_is_help=True,
)

ModelOption = Annotated[str | None, typer.Option(help="Use this model instead of VARDEX_MODEL.")]
StatsOption = Annotated[bool, typer.Option("--stats", help="Show tokens, time and cost.")]
CALL_LOG = Path("logs/calls.jsonl")


def fail(message: str) -> typer.Exit:
    """Print an error in red and return the exit to raise."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def settings_for(model: str | None) -> Settings:
    """The settings, with the model replaced when one was given on the command line."""
    settings = load_settings()
    return replace(settings, model=model) if model else settings


def connect(settings: Settings) -> ModelClient:
    """The model client. This is the one place that chooses a provider."""
    return AnthropicClient(settings)


def warn_if_prices_are_stale() -> None:
    """Print a yellow warning when the price table is too old to trust."""
    warning = stale_prices_warning()
    if warning:
        typer.secho(warning, fg="yellow", err=True)


def echo_piece(text: str) -> None:
    """Print part of a streamed answer, without a line break."""
    typer.echo(text, nl=False)


def describe(answer: Answer) -> str:
    """One line of statistics about an answer."""
    tokens_in = f"{answer.usage.input_tokens:,} in"
    if answer.usage.cache_read_tokens or answer.usage.cache_write_tokens:
        tokens_in += (
            f" (+{answer.usage.cache_read_tokens:,} cache read,"
            f" {answer.usage.cache_write_tokens:,} cache write)"
        )
    out = f"{answer.usage.output_tokens:,} out"
    if answer.thinking_tokens:
        out += f" ({answer.thinking_tokens:,} thinking)"
    seconds = f"{answer.seconds:.1f} s"
    if answer.first_text_seconds is not None:
        seconds += f" (first text after {answer.first_text_seconds:.1f} s)"
    cost = format_usd(answer.cost_usd) if answer.cost_usd is not None else "cost unknown"
    return f"{answer.model} · {tokens_in} · {out} · {seconds} · {cost}"


def record(answer: Answer, stats: bool) -> None:
    """Log a call, warn if it was cut off, and show its statistics when asked."""
    try:
        log_call(answer, CALL_LOG)
    except OSError as err:
        typer.secho(f"Warning: could not write to {CALL_LOG}: {err}", fg="yellow", err=True)
    if answer.truncated:
        typer.secho("Warning: the answer was cut off at max_tokens.", fg="yellow", err=True)
    if stats:
        typer.secho(describe(answer), fg="bright_black", err=True)


@app.command()
def version() -> None:
    """Print the Vardex version."""
    typer.echo(__version__)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to ask.")],
    model: ModelOption = None,
    effort: Annotated[
        Effort | None, typer.Option(help="How much work the model puts in. Ignored on Haiku 4.5.")
    ] = None,
    stats: StatsOption = False,
    stream: Annotated[
        bool, typer.Option("--stream", help="Print the answer as it is written.")
    ] = False,
) -> None:
    """Send a question to the model and print the answer."""
    try:
        client = connect(settings_for(model))
        answer = ask_model(client, question, effort=effort, on_text=echo_piece if stream else None)
    except ModelError as err:
        raise fail(str(err)) from err

    typer.echo("" if stream else answer.text)  # a streamed answer only needs its final line break
    if stats:
        warn_if_prices_are_stale()
    record(answer, stats)


@app.command(name="extract")
def extract_command(
    pack: Annotated[Path, typer.Argument(help="A pack folder.", exists=True, file_okay=False)],
    name: Annotated[str, typer.Argument(help="An extraction in the pack, such as key-figures.")],
    files: Annotated[
        list[Path], typer.Argument(help="Text files to extract from.", exists=True, dir_okay=False)
    ],
    model: ModelOption = None,
    effort: Annotated[
        Effort, typer.Option(help="How much work the model puts in. Ignored on Haiku 4.5.")
    ] = "low",
    stats: StatsOption = False,
    show_prompt: Annotated[
        bool,
        typer.Option(
            "--show-prompt", help="Print the request for the first file and send nothing."
        ),
    ] = False,
) -> None:
    """Extract values from documents. Prints one JSON line per document that passes the checks."""
    try:
        spec = load_extraction(pack, name)
        if show_prompt:
            show(build_request(spec, files[0].read_text(encoding="utf-8"), effort=effort))
            return
        client = connect(settings_for(model))
    except (PackError, ModelError) as err:
        raise fail(str(err)) from err

    failed = 0
    for file in files:
        try:
            result = extract(client, spec, file.read_text(encoding="utf-8"), effort=effort)
        except ExtractionError as err:
            failed += 1
            typer.secho(f"{file.name}: {err}", fg=typer.colors.RED, err=True)
            answer = err.answer
        except ModelError as err:
            raise fail(f"{file.name}: {err}") from err
        else:
            # default=str writes amounts as exact strings, never as floats
            typer.echo(
                json.dumps({"file": file.name, **result.values}, default=str, ensure_ascii=False)
            )
            answer = result.answer
        if answer is not None:
            record(answer, stats)
    if failed:
        raise fail(f"{failed} of {len(files)} documents failed.")


def show(request: Request) -> None:
    """Print a request under headings: the system prompt, each message and the output schema."""
    typer.secho("=== System prompt ===", bold=True)
    typer.echo(request.system)
    for number, message in enumerate(request.messages, start=1):
        typer.secho(f"\n=== Message {number} ({message.role}) ===", bold=True)
        typer.echo(message.text)
    if request.output_schema is not None:
        typer.secho("\n=== Output schema ===", bold=True)
        typer.echo(json.dumps(request.output_schema.model_json_schema(), indent=2))


@app.command()
def tokens(
    file: Annotated[Path, typer.Argument(help="A text file.", exists=True, dir_okay=False)],
    model: ModelOption = None,
) -> None:
    """Count the tokens in a text file, and what sending it once costs."""
    settings = settings_for(model)
    text = file.read_text(encoding="utf-8")
    try:
        n = connect(settings).count_tokens(text)
    except ModelError as err:
        raise fail(str(err)) from err

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
    model: Annotated[str | None, typer.Option(help="Show only this model.")] = None,
    batch: Annotated[
        bool, typer.Option("--batch", help="Use Message Batches API prices: 50% off.")
    ] = False,
) -> None:
    """Estimate the monthly cost of an LLM feature on every model in the price table."""
    if cached > input_tokens:
        raise fail("--cached cannot be larger than --input.")
    if model:
        try:
            price_for(model)
        except UnknownModelError as err:
            raise fail(err.args[0]) from err
    usage = Usage(
        input_tokens=input_tokens - cached,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
    )
    typer.echo(
        f"{per_day:,} requests a day for {days} days. Each request: "
        f"{input_tokens:,} input tokens ({cached:,} cached), {output_tokens:,} output tokens."
    )
    if batch:
        typer.echo("Message Batches API prices: 50% off, answers within 24 hours.")
    warn_if_prices_are_stale()
    typer.echo(f"\n{'model':<20}{'per request':>14}{'per month':>14}")
    models = [model] if model else sorted(PRICES, key=lambda m: cost_usd(m, usage))
    for name in models:
        per_request = format_usd(cost_usd(name, usage, batch=batch))
        per_month = format_usd(monthly_cost_usd(name, usage, per_day, days, batch=batch))
        typer.echo(f"{name:<20}{per_request:>14}{per_month:>14}")


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


@app.command()
def packs(
    folder: Annotated[
        Path, typer.Argument(help="A folder of packs.", exists=True, file_okay=False)
    ] = Path("packs"),
) -> None:
    """List every pack in a folder, with the error for each invalid one."""
    subfolders = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not subfolders:
        typer.echo(f"No packs found in {folder}")
        return
    invalid = 0
    for sub in subfolders:
        try:
            pack = load_pack(sub)
        except PackError as err:
            invalid += 1
            typer.secho(f"ERROR  {sub.name}", fg=typer.colors.RED)
            for line in str(err).splitlines():
                typer.echo(f"       {line}")
            continue
        recipes = ", ".join(pack.recipes)
        typer.secho(f"OK     {pack.name} {pack.version}  recipes: {recipes}", fg=typer.colors.GREEN)
    if invalid:
        raise typer.Exit(code=1)
