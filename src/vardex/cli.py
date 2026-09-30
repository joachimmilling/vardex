"""The vardex command-line tool."""

from pathlib import Path
from typing import Annotated

import typer

from vardex import __version__
from vardex.config import load_settings
from vardex.llm import MissingAPIKeyError
from vardex.llm import ask as ask_model
from vardex.packs import PackError, load_pack

app = typer.Typer(
    help="Vardex: blocks, recipes and packs for enterprise AI apps.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the Vardex version."""
    typer.echo(__version__)


@app.command()
def ask(question: Annotated[str, typer.Argument(help="The question to ask.")]) -> None:
    """Send a question to the model and print the answer."""
    try:
        typer.echo(ask_model(question, load_settings()))
    except MissingAPIKeyError as err:
        typer.secho(str(err), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err


@app.command()
def validate(
    folder: Annotated[Path, typer.Argument(help="A folder that contains a pack.yaml.")],
) -> None:
    """Check that a pack follows the pack format."""
    try:
        pack = load_pack(folder)
    except PackError as err:
        typer.secho(str(err), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from err
    recipes = ", ".join(pack.recipes)
    typer.secho(f"OK  {pack.name} {pack.version}  recipes: {recipes}", fg=typer.colors.GREEN)
