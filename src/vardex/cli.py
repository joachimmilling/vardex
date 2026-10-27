"""The vardex command-line tool."""

import json
import os
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from vardex import __version__, evals, rag, text_to_sql
from vardex.actions import Connector, FileConnector, action_tools, connector_path
from vardex.agent import MAX_STEPS, Run, run_agent
from vardex.anthropic_client import AnthropicClient
from vardex.approvals import ApprovalError, Inbox, Item, inbox_path
from vardex.budgets import Budget, BudgetError, Spend, spend_path
from vardex.calllog import log_call
from vardex.charts import charts_path
from vardex.config import Settings, load_settings
from vardex.documents import (
    document_path,
    find_page,
    load_documents,
    page_numbers,
    pages_text,
    read_document,
    tesseract,
)
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.embeddings import Embedder, Reranker
from vardex.extraction import (
    ExtractionError,
    ExtractionSpec,
    build_request,
    extract,
    load_extraction,
)
from vardex.fastembed_models import FastEmbedder, FastReranker
from vardex.indexing import index_pack
from vardex.ingest import http_client
from vardex.llm import (
    Answer,
    Conversation,
    Effort,
    Ledger,
    Message,
    MissingAPIKeyError,
    ModelClient,
    ModelError,
    ToolCall,
)
from vardex.llm import ask as ask_model
from vardex.mcp_client import McpSession, load_mcp_servers, outside_tools
from vardex.mcp_server import build_server, serve_stdio
from vardex.packs import Language, PackError, load_pack
from vardex.permissions import Permissions, load_permissions, unlisted
from vardex.pgvector_index import PostgresIndex
from vardex.postgres_warehouse import PostgresWarehouse
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
from vardex.prompts import AGENT
from vardex.provenance import Figure, Quote
from vardex.replay import (
    NotRecorded,
    Recording,
    RecordingClient,
    RecordingEmbedder,
    RecordingIndex,
    RecordingReranker,
    RecordingWarehouse,
    ReplayClient,
    ReplayEmbedder,
    ReplayIndex,
    ReplayReranker,
    ReplayWarehouse,
)
from vardex.retrieval import Filters, Index, Search, SearchError, search
from vardex.runs import Lock, RunError, RunStore, runs_path
from vardex.runs import Run as WorkflowRun
from vardex.semantic import Knowledge, check, check_documents, find_key, load_knowledge
from vardex.sql import QueryError, Result, Warehouse, format_table, format_value
from vardex.text_to_sql import AnswerError
from vardex.tools import Approver, Tool, Toolbox, ToolResult, build_toolbox
from vardex.warehouse import WarehouseError, build, warehouse_path
from vardex.workflows import (
    Connections,
    Watermarks,
    WorkflowError,
    WorkflowSpec,
    check_workflow,
    is_due,
    load_workflow,
    load_workflows,
    marks_path,
    reports_path,
    run_workflow,
    start,
    waiting_summary,
    workflow_names,
)

app = typer.Typer(
    help="Vardex: blocks, recipes and packs for enterprise AI apps.",
    no_args_is_help=True,
)

ModelOption = Annotated[str | None, typer.Option(help="Use this model instead of VARDEX_MODEL.")]
PackArgument = Annotated[Path, typer.Argument(help="A pack folder.", exists=True, file_okay=False)]
StatsOption = Annotated[bool, typer.Option("--stats", help="Show tokens, time and cost.")]
WarehouseOption = Annotated[
    str | None,
    typer.Option(
        "--warehouse",
        help="A PostgreSQL URL or a DuckDB file. Default: VARDEX_WAREHOUSE_URL, else the "
        "pack's own warehouse.",
    ),
]
CALL_LOG = Path("logs/calls.jsonl")
MAX_ASKED_BACK = 2  # questions the model may ask back before it must answer or give up


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


def connect_warehouse(url: str | None, pack: Path, settings: Settings) -> Warehouse:
    """The warehouse. This is the one place that chooses a database."""
    url = url or settings.warehouse_url
    timeout = settings.query_timeout_seconds
    if url and url.startswith(("postgresql://", "postgres://")):
        return PostgresWarehouse(url, timeout_seconds=timeout)
    path = Path(url) if url else warehouse_path(settings.data_dir, load_pack(pack).name)
    return DuckDBWarehouse(path, timeout_seconds=timeout)


def connect_embedder(settings: Settings) -> Embedder:
    """The embedding model. This is the one place that chooses one."""
    return FastEmbedder(settings.embedding_model, settings.data_dir / "models")


def connect_reranker(settings: Settings, rerank: bool = True) -> Reranker | None:
    """The reranker, or None when it is turned off. The one place that chooses one."""
    if not rerank or settings.rerank_model is None:
        return None
    return FastReranker(settings.rerank_model, settings.data_dir / "models")


def connect_index(settings: Settings, pack: Path, embedder: Embedder) -> Index:
    """The index of the pack's documents. This is the one place that chooses where it is."""
    if not settings.index_url:
        raise SearchError(
            "VARDEX_INDEX_URL is not set: it must name a PostgreSQL database with pgvector"
        )
    name = load_pack(pack).name
    return PostgresIndex(settings.index_url, name, embedder.model, embedder.dimensions)


def interactive() -> bool:
    """True when a person is at the keyboard to answer a question back."""
    return sys.stdin.isatty()


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


@app.command()
def chat(
    model: ModelOption = None,
    stats: StatsOption = False,
) -> None:
    """Talk with the model. The history is sent with every message; an empty line ends it."""
    try:
        conversation = Conversation(connect(settings_for(model)))
    except ModelError as err:
        raise fail(str(err)) from err
    typer.secho("Chat with Vardex. Press Enter on an empty line to stop.", fg="bright_black")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        try:
            answer = conversation.say(text, on_text=echo_piece)
        except ModelError as err:
            typer.secho(str(err), fg=typer.colors.RED, err=True)
            continue
        typer.echo("\n")
        record(answer, stats)
    total = conversation.cost_usd
    turns = len(conversation.answers)
    cost = format_usd(total) if total is not None else "cost unknown"
    typer.secho(f"{turns} turns · {cost}", fg="bright_black", err=True)


def pages_to_extract(
    pack: Path, spec: ExtractionSpec, document: str, settings: Settings, n: int = 3
) -> tuple[Path, set[int]]:
    """An indexed document's file, and the pages of the n passages a search finds best for
    the extraction: its description and every field's, searched within the document."""
    found = {d.id: d for d in load_documents(pack).documents}
    if document not in found:
        raise PackError(f"No document {document} in {pack}")
    embedder = connect_embedder(settings)
    index = connect_index(settings, pack, embedder)
    try:
        if document not in index.documents():
            raise PackError(f"{document} is not indexed: run vardex index {pack}")
        query = " ".join([spec.description, *(f.description for f in spec.fields.values())])
        result = search(
            index,
            embedder,
            query,
            Filters(document=document),
            limit=n,
            reranker=connect_reranker(settings),
        )
    finally:
        index.close()
    path = document_path(settings.data_dir, load_pack(pack).name, found[document])
    return path, {passage.page for passage in result.passages}


@app.command(name="extract")
def extract_command(
    pack: PackArgument,
    name: Annotated[str, typer.Argument(help="An extraction in the pack, such as key-figures.")],
    files: Annotated[
        list[Path] | None,
        typer.Argument(help="Text or PDF files to extract from.", exists=True, dir_okay=False),
    ] = None,
    pages: Annotated[
        str | None,
        typer.Option(help="In a PDF, read only these pages, such as 4 or 4-6. Default: all."),
    ] = None,
    document: Annotated[
        str | None,
        typer.Option(
            help="An indexed document, by its id, instead of files: the pages a "
            "search finds for the extraction are read."
        ),
    ] = None,
    model: ModelOption = None,
    effort: Annotated[
        Effort, typer.Option(help="How much work the model puts in. Ignored on Haiku 4.5.")
    ] = "low",
    repairs: Annotated[
        int, typer.Option(min=0, max=3, help="How often to ask again after a failed check.")
    ] = 1,
    show_prompt: Annotated[
        bool,
        typer.Option("--show-prompt", help="Print the request for the first file; call nothing."),
    ] = False,
    stats: StatsOption = False,
) -> None:
    """Extract values from documents. Prints one JSON line per document that passes the checks.
    From a PDF, each quoted value also gets the page its quote is on."""
    if bool(files) == bool(document):
        raise fail("Give files to extract from, or --document, not both.")
    try:
        spec = load_extraction(pack, name)
        wanted = page_numbers(pages) if pages else None
        if document:
            path, wanted = pages_to_extract(pack, spec, document, settings_for(model))
            typer.secho(f"{document}: pages {', '.join(map(str, sorted(wanted)))}", err=True)
            files = [path]
    except (PackError, ValueError, SearchError) as err:
        raise fail(str(err)) from err
    files = files or []
    if show_prompt:
        request = build_request(spec, files[0].read_text(encoding="utf-8"), effort)
        schema = request.output_schema.model_json_schema() if request.output_schema else None
        typer.echo(f"=== system ===\n{request.system}\n")
        for message in request.messages:
            typer.echo(f"=== {message.role} ===\n{message.text}\n")
        typer.echo(f"=== output schema ===\n{json.dumps(schema, indent=2, ensure_ascii=False)}")
        return
    try:
        client = connect(settings_for(model))
    except ModelError as err:
        raise fail(str(err)) from err

    failed = 0
    for file in files:
        ledger = Ledger(client)  # the calls for this file, still logged if one fails
        read = None
        try:
            if file.suffix.lower() == ".pdf":
                read = read_document(file, ledger, tesseract, wanted).pages
                text = pages_text(read)
            else:
                text = file.read_text(encoding="utf-8")
            result = extract(ledger, spec, text, effort=effort, repairs=repairs)
        except ExtractionError as err:
            failed += 1
            typer.secho(f"{file.name}: {err}", fg=typer.colors.RED, err=True)
        except (ModelError, PackError) as err:
            for answer in ledger.answers:
                record(answer, stats)
            raise fail(f"{file.name}: {err}") from err
        else:
            values = {"document" if document else "file": document or file.name, **result.values}
            for field, spec_field in spec.fields.items():
                quote = result.values.get(f"{field}_quote")
                if read is not None and spec_field.quote and quote:
                    values[f"{field}_page"] = find_page(read, quote)
            # default=str writes amounts as exact strings, never as floats
            typer.echo(json.dumps(values, default=str, ensure_ascii=False))
        for answer in ledger.answers:  # the pages read from pictures, and the extraction
            record(answer, stats)
    if failed:
        raise fail(f"{failed} of {len(files)} documents failed.")


@app.command()
def ingest(
    pack: PackArgument,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Fetch every source, even those fetched recently.")
    ] = False,
) -> None:
    """Fetch the pack's sources that are due, and rebuild its warehouse from scratch."""

    def say(line: str) -> None:
        typer.echo(line, err=True)

    try:
        with http_client() as client:
            result = build(pack, load_settings().data_dir, client, refresh=refresh, say=say)
    except (PackError, WarehouseError) as err:
        raise fail(str(err)) from err
    for warning in result.warnings:
        typer.secho(f"Warning: {warning}", fg="yellow", err=True)
    for stale in result.stale:
        typer.secho(f"Stale: {stale}", fg=typer.colors.RED, err=True)
    typer.echo(result.path)
    if result.stale:
        raise typer.Exit(code=1)  # the warehouse is in place, but a scheduler should notice


@app.command(name="sql")
def sql_command(
    pack: PackArgument,
    statement: Annotated[str, typer.Argument(help="The SQL to run, such as 'select 1'.")],
    warehouse: WarehouseOption = None,
    max_rows: Annotated[int, typer.Option(min=1, help="Fetch at most this many rows.")] = 1000,
) -> None:
    """Run SQL on a pack's warehouse, read-only, and print the result."""
    try:
        connection = connect_warehouse(warehouse, pack, load_settings())
    except (PackError, QueryError) as err:
        raise fail(str(err)) from err
    try:
        typer.echo(format_table(connection.run(statement, max_rows=max_rows)))
    except QueryError as err:
        raise fail(f"{connection.name}: {err}") from err
    finally:
        connection.close()


@app.command(name="check")
def check_command(pack: PackArgument, warehouse: WarehouseOption = None) -> None:
    """Check the pack's semantic layer against a warehouse, and print what does not hold."""
    try:
        knowledge = load_knowledge(pack)
        connection = connect_warehouse(warehouse, pack, load_settings())
    except (PackError, QueryError) as err:
        raise fail(str(err)) from err
    try:
        problems = check(connection, knowledge)
        # The documents are checked against the ontology's tables once those hold
        if not problems and (pack / "documents" / "sources.yaml").is_file():
            problems += check_documents(connection, knowledge.ontology, load_documents(pack))
        problems += check_permissions(pack, connection)
        problems += check_workflows(pack, knowledge)
    except PackError as err:
        raise fail(str(err)) from err
    finally:
        connection.close()
    if problems:
        raise fail(f"{connection.name}:\n" + "\n".join(f"  {p}" for p in problems))
    typer.secho(
        f"OK  {connection.name}: {len(knowledge.tables)} models, {len(knowledge.metrics)} "
        f"metrics, {len(knowledge.ontology.objects)} objects, "
        f"{len(knowledge.ontology.links)} links, {len(knowledge.ontology.actions)} actions, "
        f"{len(knowledge.questions.examples)} examples, "
        f"{len(workflow_names(pack))} workflows",
        fg=typer.colors.GREEN,
    )


def check_workflows(pack: Path, knowledge: Knowledge) -> list[str]:
    """Every workflow the pack declares, checked without running any of it."""
    if not workflow_names(pack):
        return []
    permissions = load_permissions(pack)
    problems = []
    for spec in load_workflows(pack):
        problems += check_workflow(spec, knowledge, permissions)
    return problems


def check_permissions(pack: Path, connection: Warehouse) -> list[str]:
    """Every tool with no permission level, and every level for a tool that is not there.

    A pack whose permissions and tools have drifted apart fails in the worst way: silently,
    at the moment someone asks the agent to do the thing nobody granted. A pack with no
    permissions.yaml has no agent, and there is nothing here to check.
    """
    if not (pack / "permissions.yaml").is_file():
        return []
    settings = load_settings()
    knowledge = load_knowledge(pack)
    permissions = load_permissions(pack)
    tools = build_toolbox(
        permissions,
        warehouse=connection,
        knowledge=knowledge,
        charts=charts_path(settings.data_dir, pack.name),
        extra=action_tools(
            knowledge,
            connection,
            connect_connectors(settings, pack),
            connect_inbox(settings, pack),
        ),
    )
    # The document tools exist whenever the pack has documents, index or no index
    names = list(tools.tools) + (
        ["search_documents", "read_passage"]
        if (pack / "documents" / "sources.yaml").is_file()
        else []
    )
    for spec in load_mcp_servers(pack).servers:
        # Without starting the server, only the tools the pack named can be checked
        names += [f"{spec.name}__{tool}" for tool in spec.tools]
    return unlisted(permissions, names)


def show_sources(result: Result, figures: list[Figure], err: bool = False) -> None:
    """Print what an answer rests on: the SQL, the rows, and where each number came from."""
    typer.echo(f"\nSQL, run on {result.warehouse} in {result.seconds:.2f} s", err=err)
    typer.echo(result.sql.strip(), err=err)
    typer.echo(f"\nRows\n{format_table(result)}", err=err)
    if figures:
        typer.echo("\nFigures", err=err)
    shown = set()
    for figure in figures:
        if figure.text in shown:  # each number once, where it is first written
            continue
        shown.add(figure.text)
        if figure.cells:
            cell = figure.cells[0]
            where = f"row {cell.row}, {cell.column} = {format_value(cell.value)}"
            if len(figure.cells) > 1:
                where += f" (and {len(figure.cells) - 1} more)"
        else:
            where = (
                "the query"
                if figure.in_sql
                else "the question"
                if figure.in_question
                else "the assumptions"
            )
        typer.echo(f"  {figure.text:<16} {where}", err=err)


JsonOption = Annotated[
    bool, typer.Option("--json", help="Print the result as one JSON object, for a program.")
]


def result_as_json(result: Result) -> dict[str, Any]:
    """The query an answer rests on: what ran, and what came back."""
    return {
        "sql": result.sql.strip(),
        "warehouse": result.warehouse,
        "seconds": round(result.seconds, 3),
        "columns": list(result.columns),
        "rows": [list(row) for row in result.rows],
        "truncated": result.truncated,
    }


def figure_as_json(figure: Figure) -> dict[str, Any]:
    """One number in an answer, and everywhere it can be traced to."""
    return {
        "text": figure.text,
        "traced": figure.traced,
        "cells": [
            {"row": cell.row, "column": cell.column, "value": cell.value} for cell in figure.cells
        ],
        "in_sql": figure.in_sql,
        "in_question": figure.in_question,
        "in_assumptions": figure.in_assumptions,
        "quotes": [quote_as_json(quote) for quote in figure.quotes],
    }


def quote_as_json(quote: Quote) -> dict[str, Any]:
    """Words copied from a page, and where a person can read them."""
    return {
        "document": quote.document,
        "title": quote.title,
        "page": quote.page,
        "text": " ".join(quote.text.split()),
        "link": quote.link,
    }


def reply_as_json(reply: text_to_sql.Reply) -> dict[str, Any]:
    """A warehouse answer as data: what it said, what it ran, and what it rests on."""
    return {
        "kind": reply.kind,
        "answer": reply.text,
        "assumptions": list(reply.assumptions),
        "warnings": list(reply.warnings),
        "query": result_as_json(reply.result) if reply.result else None,
        "figures": [figure_as_json(figure) for figure in reply.figures],
    }


@app.command(name="answer")
def answer_command(
    pack: PackArgument,
    question: Annotated[str, typer.Argument(help="A question about the pack's data.")],
    warehouse: WarehouseOption = None,
    model: ModelOption = None,
    effort: Annotated[
        Effort, typer.Option(help="How much work the model puts into the SQL.")
    ] = "medium",
    repairs: Annotated[
        int, typer.Option(min=0, max=3, help="How often to ask again after a failed check.")
    ] = 1,
    max_rows: Annotated[
        int, typer.Option(min=1, help="Give the model at most this many rows.")
    ] = 100,
    show_prompt: Annotated[
        bool, typer.Option("--show-prompt", help="Print the request for the SQL; call nothing.")
    ] = False,
    no_input: Annotated[
        bool, typer.Option("--no-input", help="Never wait for an answer to a question back.")
    ] = False,
    as_json: JsonOption = False,
    stats: StatsOption = False,
) -> None:
    """Answer a question from the pack's warehouse, with the SQL it ran and the rows it used.
    When the model asks a question back, answer it at the keyboard; without a keyboard, or
    with --no-input, the question is printed and the exit code is 2."""
    settings = settings_for(model)
    try:
        knowledge = load_knowledge(pack)
        connection = connect_warehouse(warehouse, pack, settings)
    except (PackError, QueryError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    spent: list[Answer] = []  # the calls for the question in hand, still logged if one fails
    try:
        if show_prompt:
            request = text_to_sql.build_request(
                knowledge,
                connection,
                question,
                effort=effort,
                max_rows=max_rows,
                embedder=connect_embedder(settings),
            )
            system = "\n\n".join(part for part in (request.system, request.system_tail) if part)
            typer.echo(f"=== system ===\n{system}\n\n=== user ===\n{question}")
            return
        client = connect(settings)
        embedder = connect_embedder(settings)
        earlier: list[Message] = []
        while True:
            spent.clear()
            reply = text_to_sql.answer(
                Ledger(client, spent),
                connection,
                knowledge,
                question,
                effort=effort,
                repairs=repairs,
                max_rows=max_rows,
                earlier=earlier,
                embedder=embedder,
            )
            for call in reply.answers:
                record(call, stats)
            asked_back = len(earlier) // 2
            if reply.kind != "clarify" or no_input or as_json or asked_back == MAX_ASKED_BACK:
                break
            if not interactive():
                break
            typer.echo(reply.text)
            earlier += [Message("user", question), Message("assistant", reply.answers[-1].text)]
            question = typer.prompt(">", prompt_suffix=" ")
    except (ModelError, QueryError) as err:
        for call in spent:
            record(call, stats)
        raise (fail_json(err) if as_json else fail(str(err))) from err
    except AnswerError as err:
        for call in err.answers:
            record(call, stats)
        if as_json:
            raise fail_json(
                err,
                data={"query": result_as_json(err.result)} if err.result else None,
            ) from err
        if err.result:
            show_sources(err.result, [], err=True)
        raise fail(str(err)) from err
    finally:
        connection.close()

    if as_json:
        # A question back is the one outcome that is not an answer, and it already exits 2.
        if reply.kind == "clarify":
            raise fail_json(reply.text, code="clarify_needed", exit_code=2)
        ok_json(reply_as_json(reply))
        return

    typer.echo(reply.text)
    for warning in reply.warnings:
        typer.secho(f"Warning: {warning}", fg="yellow", err=True)
    if reply.assumptions:
        typer.echo("\nAssumptions")
        for assumption in reply.assumptions:
            typer.echo(f"- {assumption}")
    if reply.result:
        show_sources(reply.result, reply.figures)
    elif reply.kind == "clarify":
        note = "(a question back: ask again, with the answer in your question)"
        typer.secho(note, fg="bright_black", err=True)
        raise typer.Exit(code=2)
    else:
        typer.secho("(refused: no query was run)", fg="bright_black", err=True)


AboutOption = Annotated[
    str | None, typer.Option(help="Only documents about this, such as an organisation number.")
]
KindOption = Annotated[str | None, typer.Option(help="Only this kind, such as annual-report.")]
PeriodOption = Annotated[str | None, typer.Option(help="Only this period, such as 2025.")]
LanguageOption = Annotated[Language | None, typer.Option(help="Only documents in this language.")]
DocumentOption = Annotated[str | None, typer.Option(help="Only this document, by its id.")]
CompanyOption = Annotated[
    str | None, typer.Option(help="Only documents about this company, by name, such as Mowi.")
]
RerankOption = Annotated[
    bool, typer.Option("--rerank/--no-rerank", help="Reorder the passages with the reranker.")
]


@app.command(name="index")
def index_command(
    pack: PackArgument,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Read and index every document, changed or not.")
    ] = False,
    model: ModelOption = None,
    stats: StatsOption = False,
) -> None:
    """Fetch the pack's documents, read them page by page, and index their passages. Only new
    or changed documents are read. A scanned page is read by the model, and checked by OCR."""

    def say(line: str) -> None:
        typer.echo(line, err=True)

    settings = settings_for(model)
    try:
        load_documents(pack)
        embedder = connect_embedder(settings)
        index = connect_index(settings, pack, embedder)
    except (PackError, SearchError, QueryError) as err:
        raise fail(str(err)) from err
    try:
        client: ModelClient | None = connect(settings)
    except ModelError:
        client = None  # scanned pages are left unread, and the user is warned
    try:
        with http_client() as http:
            done = index_pack(
                pack,
                settings.data_dir,
                http,
                index,
                embedder,
                client,
                tesseract,
                rebuild=rebuild,
                say=say,
            )
    except (PackError, SearchError, ModelError) as err:
        raise fail(str(err)) from err
    finally:
        index.close()
    for answer in done.answers:
        record(answer, stats)
    for warning in done.warnings:
        typer.secho(f"Warning: {warning}", fg="yellow", err=True)
    typer.echo(
        f"{done.documents} indexed, {done.unchanged} unchanged, {done.removed} removed: "
        f"{done.pages:,} pages ({done.scanned} read from a picture), {done.passages:,} passages"
    )


def document_filters(
    pack: Path,
    settings: Settings,
    company: str | None,
    about: str | None,
    rest: tuple[str | None, str | None, str | None, str | None],
) -> Filters:
    """The filters for a search. A company's name is looked up through the pack's ontology,
    among the companies that have documents, and becomes the about filter."""
    if company and about:
        raise PackError("Give --company or --about, not both.")
    if company:
        documents = load_documents(pack)
        connection = connect_warehouse(None, pack, settings)
        try:
            ontology = load_knowledge(pack).ontology
            keys = [d.about for d in documents.documents]
            about = find_key(connection, ontology, documents.about, company, keys)
        finally:
            connection.close()
    return Filters(about, *rest)


def show_search(found: Search) -> None:
    """Print a search's passages, best first, each with how it was ranked and where to read it."""
    for rank, hit in enumerate(found.hits, start=1):
        p = hit.passage
        how = [
            f"meaning {hit.meaning_rank or '-'}",
            f"words {hit.words_rank or '-'}",
            *([f"rerank {hit.rerank:.2f}"] if hit.rerank is not None else []),
        ]
        typer.echo(f"{rank:>2}  {p.document.title}, page {p.page}   ({' · '.join(how)})")
        typer.echo(f"    {p.link}")
        snippet = " ".join(p.text.split())
        typer.echo(f"    {snippet[:200]}{'…' if len(snippet) > 200 else ''}\n")


def search_as_json(found: Search) -> dict[str, Any]:
    """What a search returned: each passage, and every rank and score behind its place."""
    return {
        "query": found.query,
        "hits": [
            {
                "place": place,
                "passage_id": hit.passage.id,
                "document": hit.passage.document.id,
                "title": hit.passage.document.title,
                "page": hit.passage.page,
                "link": hit.passage.link,
                "text": hit.passage.text,
                "meaning_rank": hit.meaning_rank,
                "words_rank": hit.words_rank,
                "fused": round(hit.fused, 6),
                "rerank": hit.rerank,
            }
            for place, hit in enumerate(found.hits, start=1)
        ],
    }


def cite_as_json(reply: rag.Reply) -> dict[str, Any]:
    """A cited answer as data: what it said, and the page behind every claim."""
    return {
        "kind": reply.kind,
        "answer": reply.text,
        "citations": [
            {"number": citation.number, **quote_as_json(citation.quote)}
            for citation in reply.citations
        ],
    }


@app.command(name="search")
def search_command(
    pack: PackArgument,
    query: Annotated[str, typer.Argument(help="What to look for, in any words.")],
    company: CompanyOption = None,
    about: AboutOption = None,
    kind: KindOption = None,
    period: PeriodOption = None,
    language: LanguageOption = None,
    document: DocumentOption = None,
    limit: Annotated[int, typer.Option(min=1, max=50, help="How many passages to show.")] = 10,
    rerank: RerankOption = True,
    as_json: JsonOption = False,
) -> None:
    """Search the pack's documents, by meaning and by words, and print the best passages."""
    settings = load_settings()
    try:
        filters = document_filters(
            pack, settings, company, about, (kind, period, language, document)
        )
        embedder = connect_embedder(settings)
        index = connect_index(settings, pack, embedder)
    except (PackError, SearchError, QueryError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    try:
        found = search(
            index,
            embedder,
            query,
            filters,
            limit=limit,
            reranker=connect_reranker(settings, rerank),
        )
    except SearchError as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    finally:
        index.close()
    if not found.hits:
        note = "No passage matches; try fewer filters."
        raise fail_json(note, code="no_match") if as_json else fail(note)
    if as_json:
        ok_json(search_as_json(found))
        return
    show_search(found)


@app.command(name="cite")
def cite_command(
    pack: PackArgument,
    question: Annotated[str, typer.Argument(help="A question the pack's documents answer.")],
    company: CompanyOption = None,
    about: AboutOption = None,
    kind: KindOption = None,
    period: PeriodOption = None,
    language: LanguageOption = None,
    document: DocumentOption = None,
    passages: Annotated[
        int, typer.Option(min=1, max=20, help="How many passages the model reads.")
    ] = 8,
    rerank: RerankOption = True,
    model: ModelOption = None,
    as_json: JsonOption = False,
    stats: StatsOption = False,
) -> None:
    """Answer a question from the pack's documents, citing the page of every claim."""
    settings = settings_for(model)
    try:
        filters = document_filters(
            pack, settings, company, about, (kind, period, language, document)
        )
        embedder = connect_embedder(settings)
        index = connect_index(settings, pack, embedder)
    except (PackError, SearchError, QueryError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    try:
        client = connect(settings)
        reply = rag.answer(
            client,
            index,
            embedder,
            question,
            filters,
            reranker=connect_reranker(settings, rerank),
            passages=passages,
        )
    except (ModelError, SearchError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    except rag.CiteError as err:
        for call in err.answers:
            record(call, stats)
        raise (fail_json(err) if as_json else fail(str(err))) from err
    finally:
        index.close()
    for call in reply.answers:
        record(call, stats)
    if as_json:
        ok_json(cite_as_json(reply))
        return
    typer.echo(reply.text)
    if reply.kind == "not_found":
        typer.secho("(not in the documents: nothing is cited)", fg="bright_black", err=True)
        return
    typer.echo("\nSources")
    shown = set()
    for citation in reply.citations:
        quote = citation.quote
        if citation.number not in shown:
            shown.add(citation.number)
            typer.echo(f"[{citation.number}] {quote.title}, page {quote.page}")
            typer.echo(f"    {quote.link}")
        typer.echo(f'    "{" ".join(quote.text.split())}"')


@dataclass
class EvalConnections:
    """What an eval suite runs on. Each is live, live and recorded, or replayed, and only what
    the suite's capability needs is there."""

    client: ModelClient | None = None
    warehouse: Warehouse | None = None
    embedder: Embedder | None = None
    index: Index | None = None
    reranker: Reranker | None = None
    recording: Recording | None = None

    def close(self) -> None:
        if self.warehouse:
            self.warehouse.close()
        if self.index:
            self.index.close()


# What each kind of suite needs: a model, a warehouse, an embedder, an index, a reranker.
NEEDS: dict[type, set[str]] = {
    evals.Suite: {"model", "warehouse", "embedder"},
    evals.ExtractionSuite: {"model"},
    evals.SearchSuite: {"embedder", "index", "reranker"},
    evals.CiteSuite: {"model", "embedder", "index", "reranker"},
    evals.AgentSuite: {"model", "warehouse", "embedder", "index", "reranker"},
}


def eval_connections(
    pack: Path,
    name: str,
    suite: evals.AnySuite,
    settings: Settings,
    url: str | None,
    record: bool,
    replay: bool,
) -> EvalConnections:
    """The connections a suite runs on: live, live and recorded, or replayed."""
    needs = NEEDS[type(suite)]
    path = evals.recording_path(pack, name)
    if replay:
        if not path.is_file():
            raise PackError(f"No recording at {path}. Make one with --record.")
        recording = Recording.load(path)
        return EvalConnections(
            client=ReplayClient(recording) if "model" in needs else None,
            warehouse=ReplayWarehouse(recording) if "warehouse" in needs else None,
            embedder=ReplayEmbedder(recording) if "embedder" in needs else None,
            index=ReplayIndex(recording) if "index" in needs else None,
            reranker=ReplayReranker(recording) if recording.reranker else None,
            recording=recording,
        )
    live = EvalConnections()
    try:
        if "warehouse" in needs:
            live.warehouse = connect_warehouse(url, pack, settings)
        if "embedder" in needs:
            live.embedder = connect_embedder(settings)
        if "index" in needs and live.embedder:
            live.index = connect_index(settings, pack, live.embedder)
        if "reranker" in needs:
            live.reranker = connect_reranker(settings)
        if "model" in needs:
            live.client = connect(settings)
    except (PackError, QueryError, SearchError, ModelError):
        live.close()
        raise
    if not record:
        return live
    recording = Recording(
        live.client.model if live.client else "",
        live.warehouse.name if live.warehouse else "",
        live.warehouse.dialect if live.warehouse else "",
        date.today().isoformat(),
    )
    return EvalConnections(
        client=RecordingClient(live.client, recording) if live.client else None,
        warehouse=RecordingWarehouse(live.warehouse, recording) if live.warehouse else None,
        embedder=RecordingEmbedder(live.embedder, recording) if live.embedder else None,
        index=RecordingIndex(live.index, recording) if live.index else None,
        reranker=RecordingReranker(live.reranker, recording) if live.reranker else None,
        recording=recording,
    )


def check_suite_for(pack: Path, suite: evals.AnySuite, knowledge: Knowledge | None) -> list[str]:
    """Everything that makes a suite unfit to measure with, for its kind of suite."""
    if isinstance(suite, evals.ExtractionSuite):
        return evals.check_extraction_suite(suite, load_extraction(pack, suite.extraction), pack)
    if isinstance(suite, evals.SearchSuite | evals.CiteSuite):
        return evals.check_document_suite(suite, load_documents(pack))
    if isinstance(suite, evals.AgentSuite):
        return evals.check_agent_suite(suite, load_permissions(pack))
    return evals.check_suite(suite, knowledge)  # type: ignore[arg-type]


def eval_suite(
    pack: Path,
    name: str,
    suite: evals.AnySuite,
    knowledge: Knowledge | None,
    settings: Settings,
    *,
    warehouse: str | None,
    effort: Effort,
    only: list[str] | None,
    record_run: bool,
    replay_run: bool,
    details: bool,
    repeat: int = 1,
    baseline: Path | None = None,
) -> bool:
    """Run one suite and print its report. True when it fails its gate."""
    try:
        problems = check_suite_for(pack, suite, knowledge)
    except PackError as err:
        raise fail(str(err)) from err
    if problems:
        raise fail(f"Suite {name} cannot be used:\n" + "\n".join(f"  {p}" for p in problems))
    try:
        c = eval_connections(pack, name, suite, settings, warehouse, record_run, replay_run)
    except (PackError, QueryError, ModelError, SearchError) as err:
        raise fail(str(err)) from err
    graders: list[evals.Grader] = []
    if isinstance(suite, evals.Suite) and c.client:
        graders = [evals.grade_behaviour, evals.grade_rows, evals.Judge(c.client)]
    elif isinstance(suite, evals.ExtractionSuite):
        graders = [evals.grade_behaviour, evals.grade_fields]
    elif isinstance(suite, evals.CiteSuite) and c.client:
        graders = [evals.grade_behaviour, evals.grade_citations, evals.FaithfulnessJudge(c.client)]
    elif isinstance(suite, evals.AgentSuite):
        # Every grader here is code: whether it answered, which tools it used, what the
        # answer says and how many steps it took. A run costs model calls; grading it does not.
        graders = [evals.grade_agent, evals.grade_tools, evals.grade_contains, evals.grade_steps]

    def agent_toolbox() -> Toolbox:
        """The agent's tools for one case. No MCP servers and nobody to approve: an eval
        measures what the pack can do on its own, with nothing outside it and no person.
        What it drafts goes to an inbox of its own, never the one people read."""
        knowledge_now = knowledge or load_knowledge(pack)
        return build_toolbox(
            load_permissions(pack),
            warehouse=c.warehouse,
            knowledge=knowledge_now,
            index=c.index,
            embedder=c.embedder,
            reranker=c.reranker,
            charts=charts_path(settings.data_dir, pack.name),
            extra=action_tools(
                knowledge_now,
                c.warehouse,
                connect_connectors(settings, pack),
                Inbox(Path(tempfile.mkdtemp(prefix="vardex-eval-")) / "inbox.jsonl"),
            )
            if c.warehouse
            else [],
        )

    started: list[str] = []  # the ids of the cases run so far, to name one that stops the run

    def run_case(case: evals.AnyCase) -> evals.CaseResult:
        started.append(case.id)
        if isinstance(case, evals.ExtractionCase) and isinstance(suite, evals.ExtractionSuite):
            spec = load_extraction(pack, suite.extraction)
            return evals.extract_case(c.client, pack, spec, case, graders)  # type: ignore[arg-type]
        if isinstance(case, evals.SearchCase) and isinstance(suite, evals.SearchSuite):
            return evals.search_case(c.index, c.embedder, c.reranker, suite, case)  # type: ignore[arg-type]
        if isinstance(case, evals.CiteCase):
            return evals.cite_case(c.client, c.index, c.embedder, c.reranker, case, graders)  # type: ignore[arg-type]
        if isinstance(case, evals.AgentCase):
            return evals.agent_case(c.client, agent_toolbox, case, graders)  # type: ignore[arg-type]
        return evals.answer_case(
            c.client,
            c.warehouse,
            knowledge,
            suite,
            case,
            graders,  # type: ignore[arg-type]
            effort=effort,
            embedder=c.embedder,
        )

    def show(result: evals.CaseResult) -> None:
        mark, colour = ("ok", "green") if result.passed else ("FAIL", "red")
        typer.secho(f"{mark:<5}{result.case.id}", fg=colour, err=True)
        if not replay_run:
            for call in result.answers:
                record(call, stats=False)

    model = c.client.model if c.client else (c.embedder.model if c.embedder else "")
    recording = c.recording
    try:
        report = evals.run(name, suite, model, run_case, only=only, repeat=repeat, on_result=show)
    except NotRecorded as err:
        recorded_on = recording.recorded_on if recording else "?"
        raise fail(
            f"{name}, case {started[-1]}: {err}. A prompt, the pack, the code or the model has "
            f"changed since the recording of {recorded_on}: record it again with --record."
        ) from err
    except (PackError, QueryError, ModelError, SearchError) as err:
        raise fail(f"{name}: {err}") from err
    finally:
        c.close()
    if record_run and recording:
        recording.verdicts = evals.verdicts(report)
        recording.save(evals.recording_path(pack, name))

    source = f", replayed from the recording of {recording.recorded_on}" if replay_run else ""
    times = f", {repeat} runs each" if repeat > 1 else ""
    typer.echo(f"\n{name}: {len(report.cases)} cases{times} on {report.model}{source}\n")
    typer.echo(evals.format_report(report, details))
    if baseline:
        before = Recording.load(baseline).verdicts
        if not before:
            raise fail(f"{baseline} holds no verdicts: record it again with this version.")
        verdicts = {case: verdict["passed"] for case, verdict in before.items()}
        typer.echo("\n" + evals.format_comparison(evals.compare_runs(verdicts, report)))
    calls = sum(len(result.answers) for result in report.results)
    cost = format_usd(report.cost_usd) if report.cost_usd is not None else "cost unknown"
    paid = " when it was recorded" if replay_run else ""
    typer.secho(f"\n{calls} model calls · {cost}{paid}", fg="bright_black", err=True)
    if only:
        typer.secho("Only some cases ran, so the gate was not applied.", fg="yellow", err=True)
        return False
    problems = report.gate(suite.threshold)
    for problem in problems:
        typer.secho(f"FAILED: {problem}", fg=typer.colors.RED, err=True)
    if not problems:
        typer.secho(
            f"Passed: at least {suite.threshold:.0%}, and every critical case.",
            fg=typer.colors.GREEN,
            err=True,
        )
    return bool(problems)


@app.command(name="eval")
def eval_command(
    pack: PackArgument,
    suites: Annotated[
        list[str] | None,
        typer.Argument(help="Suites to run, such as answer. Default: every suite in the pack."),
    ] = None,
    warehouse: WarehouseOption = None,
    model: ModelOption = None,
    effort: Annotated[
        Effort, typer.Option(help="How much work the model puts into the SQL.")
    ] = "medium",
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Run only this case. Repeat for more.")
    ] = None,
    record_run: Annotated[
        bool,
        typer.Option("--record", help="Save every model answer and query result, to replay."),
    ] = False,
    replay_run: Annotated[
        bool,
        typer.Option("--replay", help="Answer from the recording: no API key, no warehouse."),
    ] = False,
    details: Annotated[
        bool,
        typer.Option("--details", help="Show each failure's question, SQL and reply."),
    ] = False,
    repeat: Annotated[int, typer.Option(min=1, max=10, help="Run every case this many times.")] = 1,
    baseline: Annotated[
        Path | None,
        typer.Option(help="A recording to compare with, case by case.", exists=True),
    ] = None,
) -> None:
    """Run the pack's golden sets and report how often Vardex gets them right. Exits with 1
    when a suite falls below its threshold or a critical case fails."""
    if record_run and (replay_run or case):
        raise fail("--record records whole suites, live: leave out --replay and --case.")
    settings = settings_for(model)
    try:
        found = evals.load_suites(pack)
    except PackError as err:
        raise fail(str(err)) from err
    names = suites or list(found)
    unknown = [name for name in names if name not in found]
    if unknown or not names:
        known = ", ".join(found) or "none"
        raise fail(f"No suite {', '.join(unknown)} in {pack}. Known: {known}")
    try:
        # Only text-to-SQL needs the semantic layer: a pack of documents alone has none
        sql = any(isinstance(found[name], evals.Suite) for name in names)
        knowledge = load_knowledge(pack) if sql else None
    except PackError as err:
        raise fail(str(err)) from err
    failed = [
        eval_suite(
            pack,
            name,
            found[name],
            knowledge,
            settings,
            warehouse=warehouse,
            effort=effort,
            only=case,
            record_run=record_run,
            replay_run=replay_run,
            details=details,
            repeat=repeat,
            baseline=baseline,
        )
        for name in names
    ]
    if any(failed):
        raise typer.Exit(code=1)


@app.command(name="add-case")
def add_case_command(
    pack: PackArgument,
    suite: Annotated[str, typer.Argument(help="The suite to add to, such as answer.")],
    question: Annotated[str, typer.Argument(help="The question, as a person asked it.")],
    type_: Annotated[str, typer.Option("--type", help="The case's type, such as lookup.")],
    language: Annotated[Language, typer.Option(help="The question's language.")] = "en",
    expect: Annotated[
        Literal["clarify", "refuse"] | None,
        typer.Option(help="Add a case that must be asked back or refused; nothing is run."),
    ] = None,
    warehouse: WarehouseOption = None,
    model: ModelOption = None,
) -> None:
    """Add a case to a golden set. Without --expect, the question is answered live and the
    case is added, with the SQL as its reference, only when you confirm the answer is right."""
    settings = settings_for(model)
    try:
        knowledge = load_knowledge(pack)
        suites = evals.load_suites(pack)
    except PackError as err:
        raise fail(str(err)) from err
    if suite not in suites:
        raise fail(f"No suite {suite} in {pack}. Known: {', '.join(suites) or 'none'}")
    taken = {case.id for case in suites[suite].cases}
    sql = None
    if expect is None:
        if not interactive():
            raise fail("Confirming an answer needs a keyboard. Nothing was added.")
        try:
            connection = connect_warehouse(warehouse, pack, settings)
        except (PackError, QueryError) as err:
            raise fail(str(err)) from err
        try:
            reply = text_to_sql.answer(
                connect(settings),
                connection,
                knowledge,
                question,
                today=suites[suite].today,
                embedder=connect_embedder(settings),
            )
        except (ModelError, QueryError, AnswerError) as err:
            raise fail(str(err)) from err
        finally:
            connection.close()
        if reply.kind != "answer" or reply.result is None:
            raise fail(f"Vardex did not answer ({reply.kind}): {reply.text}")
        typer.echo(reply.text)
        show_sources(reply.result, reply.figures)
        if not typer.confirm("\nIs this answer right?", default=False):
            typer.echo("Nothing was added.")
            return
        sql = reply.result.sql
    try:
        case = evals.Case(
            id=evals.new_case_id(question, taken),
            type=type_,
            language=language,
            question=question,
            expect=expect or "answer",
            sql=sql,
        )
        evals.add_case(pack / "evals" / f"{suite}.yaml", case, knowledge)
    except (ValueError, PackError) as err:
        raise fail(str(err)) from err
    typer.secho(f"Added case {case.id} to {suite}.", fg=typer.colors.GREEN)
    typer.secho(
        f"The suite changed: record it again with vardex eval {pack} {suite} --record",
        fg="yellow",
        err=True,
    )


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


# The agent, and the two channels it is reached through


def connect_connectors(settings: Settings, pack: Path) -> dict[str, Connector]:
    """The outside systems a pack's actions write to. The one place that chooses them."""
    name = load_pack(pack).name
    return {
        "crm": FileConnector(
            connector_path(settings.data_dir, name, "crm"), name="the demo CRM, a file"
        )
    }


def approve_at_keyboard(call: ToolCall, tool: Tool) -> bool:
    """Ask the person who started the run. Only used when somebody is there to ask."""
    arguments = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    typer.secho(f"\n{tool.name}({arguments})", fg=typer.colors.YELLOW, err=True)
    return typer.confirm("Allow this?", default=False, err=True)


def connect_inbox(settings: Settings, pack: Path) -> Inbox:
    """The pack's approval inbox. The one place that decides where proposals wait."""
    return Inbox(inbox_path(settings.data_dir, load_pack(pack).name))


@dataclass
class Built:
    """Everything one run of the agent can reach, and what has to be closed afterwards."""

    toolbox: Toolbox
    warehouse: Warehouse
    knowledge: Knowledge
    permissions: Permissions
    open_now: list[Warehouse | Index]


def build_agent(
    pack: Path,
    settings: Settings,
    *,
    warehouse_url: str | None = None,
    documents: bool = True,
    rerank: bool = True,
    dry_run: bool = False,
    approve: Approver | None = None,
    sessions: list[McpSession] | None = None,
    context: dict[str, str] | None = None,
) -> Built:
    """Everything the agent of one run can reach, and what has to be closed afterwards."""
    knowledge = load_knowledge(pack)
    permissions = load_permissions(pack)
    connection = connect_warehouse(warehouse_url, pack, settings)
    open_now: list[Warehouse | Index] = [connection]
    index = embedder = None
    if documents and (pack / "documents" / "sources.yaml").is_file():
        embedder = connect_embedder(settings)
        index = connect_index(settings, pack, embedder)
        open_now.append(index)
    extra = action_tools(
        knowledge,
        connection,
        connect_connectors(settings, pack),
        connect_inbox(settings, pack),
    )
    extra += outside_tools(sessions or [])
    toolbox = build_toolbox(
        permissions,
        warehouse=connection,
        knowledge=knowledge,
        index=index,
        embedder=embedder,
        reranker=connect_reranker(settings, rerank) if index is not None else None,
        charts=charts_path(settings.data_dir, pack.name),
        extra=extra,
        approve=approve,
        dry_run=dry_run,
        context=context or {},
    )
    return Built(toolbox, connection, knowledge, permissions, open_now)


ERROR_CODES: list[tuple[type[Exception], str]] = [
    (MissingAPIKeyError, "no_api_key"),
    (BudgetError, "budget_exceeded"),
    (ModelError, "model_failed"),
    (PackError, "pack_invalid"),
    (SearchError, "index_unavailable"),
    (WarehouseError, "warehouse_unavailable"),
    (QueryError, "query_failed"),
]


def error_code(err: Exception) -> str:
    """A name a program can branch on, for each way a command can fail."""
    for kind, code in ERROR_CODES:
        if isinstance(err, kind):
            return code
    return "failed"


def print_json(body: dict[str, Any]) -> None:
    """Print one JSON object on stdout, for a program reading our output.

    One object and nothing else. Everything a person would want to read — warnings, statistics,
    the rows in a table — goes to stderr, so `vardex … --json | jq` never chokes on prose.
    """
    typer.echo(json.dumps(body, ensure_ascii=False, default=str))


def ok_json(data: dict[str, Any]) -> None:
    """Report success as the one envelope every command shares."""
    print_json({"ok": True, "data": data})


def fail_json(
    err: Exception | str,
    code: str = "failed",
    exit_code: int = 1,
    data: dict[str, Any] | None = None,
) -> typer.Exit:
    """Report a failure as the same envelope, and exit with a code that matches `ok`.

    A program should never have to read the exit code and the body and find them disagreeing,
    so the two are set in one place: ok false here always means a non-zero exit.

    `data` carries whatever the command did manage to produce. A run that stopped on its
    budget still cost money, and a caller that only learns it failed cannot bill it.
    """
    if isinstance(err, Exception):
        code = error_code(err)
    body: dict[str, Any] = {"ok": False, "error": {"code": code, "message": str(err)}}
    if data is not None:
        body["data"] = data
    print_json(body)
    return typer.Exit(code=exit_code)


def run_as_json(run: Run, toolbox: Toolbox) -> dict[str, Any]:
    """A whole run as data: the answer, every call, and what it cost."""
    return {
        "stop": run.stop,
        "answer": run.answer,
        "note": run.note,
        "steps": len(run.steps),
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "permission": toolbox.level(call.name),
                "is_error": result.is_error,
                "wrote": result.wrote,
                "content": result.content,
                "data": result.structured,
            }
            for call, result in run.tool_calls
        ],
        "usage": {
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "cache_read_tokens": run.usage.cache_read_tokens,
            "cache_write_tokens": run.usage.cache_write_tokens,
        },
        "cost_usd": str(run.cost_usd) if run.cost_usd is not None else None,
        "seconds": round(run.seconds, 2),
    }


@app.command(name="agent")
def agent_command(
    pack: PackArgument,
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    model: ModelOption = None,
    warehouse: WarehouseOption = None,
    effort: Annotated[Effort | None, typer.Option(help="How much work the model puts in.")] = None,
    steps: Annotated[int, typer.Option(help="Most model calls in one run.")] = MAX_STEPS,
    documents: Annotated[
        bool, typer.Option("--documents/--no-documents", help="Let it search the pack's documents.")
    ] = True,
    rerank: Annotated[
        bool, typer.Option("--rerank/--no-rerank", help="Rerank search results.")
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what a tool would change, change nothing.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the whole run as JSON, for another program.")
    ] = False,
    show_steps: Annotated[
        bool, typer.Option("--steps-shown", help="Print each step as it runs.")
    ] = False,
    stats: StatsOption = False,
) -> None:
    """Answer a question by letting the model use the pack's tools."""
    settings = settings_for(model)
    if not as_json:
        warn_if_prices_are_stale()
    sessions: list[McpSession] = []
    try:
        specs = load_mcp_servers(pack).servers
        permissions = load_permissions(pack)
        with ExitStack() as stack:
            for spec in specs:
                sessions.append(
                    stack.enter_context(
                        McpSession(spec, environment=dict(os.environ), ask=ask_at_keyboard)
                    )
                )
            built = build_agent(
                pack,
                settings,
                warehouse_url=warehouse,
                documents=documents,
                rerank=rerank,
                dry_run=dry_run,
                # With nobody at the keyboard there is nobody to ask, so a tool that needs
                # approval drafts into the inbox instead of refusing. Same rung, and the
                # difference is only who answers.
                approve=approve_at_keyboard if interactive() else None,
                sessions=sessions,
            )
            toolbox = built.toolbox
            for closing in built.open_now:
                stack.callback(closing.close)
            budget = Budget(
                connect(settings),
                caps=permissions.budgets,
                spend=Spend(spend_path(settings.data_dir, load_pack(pack).name)),
            )
            run = run_agent(
                budget,
                toolbox,
                question,
                system=AGENT,
                max_steps=steps,
                on_call=show_call if show_steps and not as_json else None,
            )
    except (PackError, ModelError, SearchError, WarehouseError, QueryError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    if as_json:
        body = run_as_json(run, toolbox)
        if not run.answered:
            raise fail_json(run.note, code=f"agent_{run.stop}", data=body)
        ok_json(body)
        raise typer.Exit(code=0)
    for answer in (step.answer for step in run.steps):
        record(answer, stats=False)
    if not run.answered:
        typer.secho(run.note, fg=typer.colors.RED, err=True)
    typer.echo(run.answer)
    if stats:
        cost = format_usd(run.cost_usd) if run.cost_usd is not None else "cost unknown"
        typer.secho(
            f"{len(run.steps)} steps · {len(run.tool_calls)} tool calls · "
            f"{run.usage.input_tokens:,} in · {run.usage.output_tokens:,} out · "
            f"{run.seconds:.1f} s · {cost}",
            fg="bright_black",
            err=True,
        )
    if not run.answered:
        raise typer.Exit(code=1)


def show_call(call: ToolCall, result: ToolResult) -> None:
    """Print one tool call as it happens, so a person can watch the agent work."""
    arguments = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    colour = typer.colors.RED if result.is_error else typer.colors.BRIGHT_BLACK
    typer.secho(f"  {call.name}({arguments[:120]})", fg=colour, err=True)


def ask_at_keyboard(question: str) -> bool:
    """Put a yes-or-no question from an outside server to the person."""
    if not interactive():
        return False
    return typer.confirm(question, default=False, err=True)


@app.command(name="tools")
def tools_command(
    pack: PackArgument,
    warehouse: WarehouseOption = None,
    documents: Annotated[bool, typer.Option("--documents/--no-documents")] = True,
) -> None:
    """List the tools this pack's agent has, and what each may do."""
    settings = load_settings()
    sessions: list[McpSession] = []
    try:
        with ExitStack() as stack:
            for spec in load_mcp_servers(pack).servers:
                sessions.append(stack.enter_context(McpSession(spec, environment=dict(os.environ))))
            built = build_agent(
                pack, settings, warehouse_url=warehouse, documents=documents, sessions=sessions
            )
            for closing in built.open_now:
                stack.callback(closing.close)
            rows = [
                (name, built.toolbox.level(name) or "none", tool.summary)
                for name, tool in built.toolbox.tools.items()
            ]
    except (PackError, SearchError, WarehouseError, QueryError) as err:
        raise fail(str(err)) from err
    width = max(len(name) for name, _, _ in rows)
    for name, level, summary in rows:
        colour = typer.colors.RED if level == "none" else None
        typer.secho(f"{name.ljust(width)}  {level:<8}  {summary[:70]}", fg=colour)


@app.command(name="mcp")
def mcp_command(
    pack: PackArgument,
    warehouse: WarehouseOption = None,
    documents: Annotated[bool, typer.Option("--documents/--no-documents")] = True,
) -> None:
    """Offer this pack's tools to another agent, over MCP on standard input and output."""
    settings = load_settings()
    name = load_pack(pack).name

    def build() -> Toolbox:
        # A new toolbox per request: the protocol is stateless, and a connection should not
        # be held open for a client that may never come back.
        return build_agent(pack, settings, warehouse_url=warehouse, documents=documents).toolbox

    try:
        build()  # fail loudly here rather than on the first request from a client
    except (PackError, SearchError, WarehouseError, QueryError) as err:
        raise fail(str(err)) from err
    serve_stdio(build_server(name, build))


ByOption = Annotated[
    str,
    typer.Option(
        "--by",
        help="Who is deciding. Kept in the log, so it has to be a person, not a role.",
    ),
]
WorkflowArgument = Annotated[
    str | None,
    typer.Argument(help="One workflow. Without it, every workflow the pack has."),
]


def connect_workflow(
    pack: Path,
    settings: Settings,
    spec: WorkflowSpec,
    stack: ExitStack,
    *,
    warehouse: str | None,
    documents: bool,
    rerank: bool,
    dry_run: bool,
    spent: Decimal = Decimal(0),
) -> EvalConnections:
    """Everything one workflow run can reach. The one place a run is wired together.

    Note what the agent is given: no approver. There is nobody at the keyboard, so a tool
    that needs a person can only draft, which is exactly what we want of a run nobody is
    watching. And the budget starts at what the run has already spent, so resuming a run
    does not hand it a fresh allowance.
    """
    name = load_pack(pack).name
    sessions = [
        stack.enter_context(McpSession(server, environment=dict(os.environ)))
        for server in load_mcp_servers(pack).servers
    ]
    built = build_agent(
        pack,
        settings,
        warehouse_url=warehouse,
        documents=documents,
        rerank=rerank,
        dry_run=dry_run,
        approve=None,
        sessions=sessions,
    )
    for closing in built.open_now:
        stack.callback(closing.close)
    # The workflow may set its own cap for one run; the pack's cap for the day always holds.
    caps = built.permissions.budgets.model_copy(
        update={"per_run": spec.budget.per_run or built.permissions.budgets.per_run}
    )
    client = Budget(
        connect(settings),
        caps=caps,
        spend=Spend(spend_path(settings.data_dir, name)),
        run_usd=spent,
    )
    return Connections(
        pack=name,
        folder=str(pack),
        knowledge=built.knowledge,
        permissions=built.permissions,
        warehouse=built.warehouse,
        client=client,
        inbox=connect_inbox(settings, pack),
        store=RunStore(runs_path(settings.data_dir, name)),
        marks=Watermarks(marks_path(settings.data_dir, name, spec.name)),
        reports=reports_path(settings.data_dir, name),
        # One toolbox, copied per case with that case's context. The tools themselves hold
        # the open warehouse and index, so this costs nothing but the dataclass.
        toolbox=lambda context: replace(built.toolbox, context=context, calls=[]),
        http=http_client,
    )


def run_as_dict(run: WorkflowRun) -> dict[str, Any]:
    """A workflow run as data, for a program or a person reading --json."""
    return {
        "id": run.id,
        "workflow": run.workflow,
        "state": run.state,
        "trigger": run.trigger,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "attempts": run.attempts,
        "dry_run": run.dry_run,
        "note": run.note,
        "cost_usd": run.cost_usd,
        "steps": [
            {
                "id": step.id,
                "uses": step.uses,
                "state": step.state,
                "items": step.items,
                "note": step.note,
                "cost_usd": step.cost_usd,
            }
            for step in run.steps
        ],
        "report": run.facts.get("report"),
        "proposals": run.facts.get("proposals", []),
        "cases": len(run.facts.get("cases", [])),
    }


def show_run(run: WorkflowRun) -> None:
    """A run as a few lines on stderr, so stdout stays whatever the command prints."""
    colour = {"done": typer.colors.GREEN, "failed": typer.colors.RED}.get(
        run.state, typer.colors.YELLOW
    )
    typer.secho(f"{run.id}  {run.state}", fg=colour, err=True)
    for step in run.steps:
        mark = {"done": " ", "failed": "!", "stopped": "-", "skipped": "·"}.get(step.state, "·")
        typer.secho(
            f" {mark} {step.id:<12} {step.uses:<7} {step.note[:90]}", fg="bright_black", err=True
        )
    if run.note:
        typer.secho(f"   {run.note}", fg=colour, err=True)


def run_one_workflow(
    pack: Path,
    settings: Settings,
    spec: WorkflowSpec,
    *,
    trigger: str,
    warehouse: str | None,
    documents: bool,
    rerank: bool,
    dry_run: bool,
    resume: str | None,
    show_steps: bool,
) -> WorkflowRun:
    """Start or resume one run of one workflow, under its own lock."""
    name = load_pack(pack).name
    store = RunStore(runs_path(settings.data_dir, name))
    lock = Lock(runs_path(settings.data_dir, name) / f"{spec.name}.lock")
    with ExitStack() as stack:
        stack.enter_context(lock)
        earlier = store.load(resume) if resume else None
        connections = connect_workflow(
            pack,
            settings,
            spec,
            stack,
            warehouse=warehouse,
            documents=documents,
            rerank=rerank,
            dry_run=dry_run,
            spent=earlier.spent if earlier else Decimal(0),
        )
        run = earlier or start(spec, connections, trigger=trigger, dry_run=dry_run)
        return run_workflow(
            spec,
            run,
            connections,
            on_call=show_call if show_steps else None,
        )


@app.command(name="run")
def run_command(
    pack: PackArgument,
    workflow: WorkflowArgument = None,
    model: ModelOption = None,
    warehouse: WarehouseOption = None,
    if_due: Annotated[
        bool,
        typer.Option("--if-due", help="Run only what the schedule says is due. For a scheduler."),
    ] = False,
    resume: Annotated[
        str | None, typer.Option(help="Carry on an earlier run, by its id, from where it stopped.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what would change, change nothing.")
    ] = False,
    documents: Annotated[bool, typer.Option("--documents/--no-documents")] = True,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    show_steps: Annotated[
        bool, typer.Option("--steps-shown", help="Print each tool call as it happens.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the runs as JSON.")] = False,
) -> None:
    """Run the pack's scheduled work: on demand, or whatever the schedule says is due."""
    settings = settings_for(model)
    if not as_json:
        warn_if_prices_are_stale()
    try:
        if resume:
            earlier = RunStore(runs_path(settings.data_dir, load_pack(pack).name)).load(resume)
            specs = [load_workflow(pack, earlier.workflow)]
        else:
            specs = [load_workflow(pack, workflow)] if workflow else load_workflows(pack)
    except (PackError, RunError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    if not specs:
        message = f"{pack} declares no workflows: there is nothing to run."
        raise (fail_json(message, code="pack_invalid") if as_json else fail(message))
    runs: list[WorkflowRun] = []
    skipped: list[str] = []
    try:
        for spec in specs:
            store = RunStore(runs_path(settings.data_dir, load_pack(pack).name))
            if if_due and not resume and not is_due(spec, store):
                skipped.append(spec.name)
                continue
            runs.append(
                run_one_workflow(
                    pack,
                    settings,
                    spec,
                    trigger="schedule" if if_due else "manual",
                    warehouse=warehouse,
                    documents=documents,
                    rerank=rerank,
                    dry_run=dry_run,
                    resume=resume,
                    show_steps=show_steps and not as_json,
                )
            )
    except (PackError, RunError, WorkflowError, ModelError, SearchError, WarehouseError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    worst = next((run for run in runs if run.state in ("failed", "stopped")), None)
    if as_json:
        body = {"runs": [run_as_dict(run) for run in runs], "not_due": skipped}
        if worst is not None:
            raise fail_json(worst.note, code=f"workflow_{worst.state}", data=body)
        ok_json(body)
        raise typer.Exit(code=0)
    for name in skipped:
        typer.secho(f"{name}: not due yet", fg="bright_black", err=True)
    for run in runs:
        show_run(run)
        report = run.facts.get("report")
        if report:
            typer.echo(report)
    if runs:
        waiting = connect_inbox(settings, pack).items(state="proposed")
        typer.secho(waiting_summary(waiting), fg="bright_black", err=True)
    if worst is not None:
        raise typer.Exit(code=1)


@app.command(name="runs")
def runs_command(
    pack: PackArgument,
    run_id: Annotated[str | None, typer.Argument(help="One run, by its id.")] = None,
    workflow: Annotated[str | None, typer.Option(help="Only this workflow's runs.")] = None,
    limit: Annotated[int, typer.Option(help="How many to list.")] = 10,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List the pack's recent runs, or show one of them."""
    settings = load_settings()
    store = RunStore(runs_path(settings.data_dir, load_pack(pack).name))
    try:
        if run_id:
            run = store.load(run_id)
            if as_json:
                ok_json(run_as_dict(run))
                return
            show_run(run)
            typer.echo(json.dumps(run.facts, indent=2, ensure_ascii=False, default=str))
            return
        found = store.recent(limit=limit, workflow=workflow)
    except (PackError, RunError) as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    if as_json:
        ok_json({"runs": [run_as_dict(run) for run in found]})
        return
    if not found:
        typer.secho("No runs yet.", fg="bright_black")
        return
    for run in found:
        colour = {"done": typer.colors.GREEN, "failed": typer.colors.RED}.get(run.state, "yellow")
        typer.secho(
            f"{run.id:<34} {run.state:<8} {run.trigger:<8} ${Decimal(run.cost_usd):.4f}  "
            f"{run.note[:50]}",
            fg=colour,
        )


@app.command(name="schedule")
def schedule_command(
    pack: PackArgument, as_json: Annotated[bool, typer.Option("--json")] = False
) -> None:
    """What the pack's workflows are, when each last ran, and when each is next due."""
    settings = load_settings()
    try:
        specs = load_workflows(pack)
    except PackError as err:
        raise (fail_json(err) if as_json else fail(str(err))) from err
    store = RunStore(runs_path(settings.data_dir, load_pack(pack).name))
    now = datetime.now(UTC)
    rows = []
    for spec in specs:
        last = store.last_done(spec.name)
        rows.append(
            {
                "workflow": spec.name,
                "schedule": spec.schedule.describe() if spec.schedule else "by hand only",
                "last_done": last.started_at if last else None,
                "due_now": is_due(spec, store, now),
                "next_due": (
                    spec.schedule.next_after(now).isoformat(timespec="minutes")
                    if spec.schedule
                    else None
                ),
            }
        )
    if as_json:
        ok_json({"workflows": rows})
        return
    for row in rows:
        mark = "due now" if row["due_now"] else f"next {row['next_due'] or '—'}"
        typer.echo(
            f"{row['workflow']:<16} {row['schedule']:<34} "
            f"last {row['last_done'] or 'never':<22} {mark}"
        )


def item_as_json(item: Item) -> dict[str, Any]:
    return item.as_dict()


def show_item(item: Item, *, wide: bool = False) -> None:
    colour = {
        "proposed": typer.colors.YELLOW,
        "done": typer.colors.GREEN,
        "rejected": typer.colors.BRIGHT_BLACK,
        "failed": typer.colors.RED,
    }.get(item.state, None)
    overdue = " OVERDUE" if item.overdue() else ""
    typer.secho(f"{item.reference}  {item.state}{overdue}", fg=colour)
    typer.echo(f"  {item.proposal.sentence}")
    if wide:
        if item.proposal.why:
            typer.echo(f"  Why: {item.proposal.why}")
        typer.echo(f"  Proposed {item.made_at} by {item.history[0].by or 'unknown'}")
        if item.needed_by:
            typer.echo(f"  Answer by {item.needed_by}")
        for entry in item.history[1:]:
            typer.echo(f"  {entry.at}  {entry.state} {entry.by} {entry.note}".rstrip())


@app.command(name="inbox")
def inbox_command(
    pack: PackArgument,
    state: Annotated[str | None, typer.Option(help="Only this state, such as done.")] = None,
    overdue: Annotated[bool, typer.Option("--overdue", help="Only what is late.")] = False,
    wide: Annotated[bool, typer.Option("--wide", help="Show the history of each one.")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """What is waiting for a person to decide, and what has been decided."""
    settings = load_settings()
    inbox = connect_inbox(settings, pack)
    items = inbox.overdue() if overdue else inbox.items(state=state)  # type: ignore[arg-type]
    if as_json:
        ok_json({"items": [item_as_json(item) for item in items]})
        return
    if not items:
        typer.secho("Nothing to show.", fg="bright_black")
        return
    for item in items:
        show_item(item, wide=wide)
    waiting = inbox.items(state="proposed")
    typer.secho(waiting_summary(waiting), fg="bright_black", err=True)


def may_decide(pack: Path, by: str) -> None:
    """Refuse a decision by somebody the pack has not named as an approver."""
    permissions = load_permissions(pack)
    if not by.strip():
        raise fail("Say who is deciding: --by YOUR-NAME. It goes in the log.")
    if not permissions.may_approve(by):
        allowed = ", ".join(permissions.approvals.approvers)
        raise fail(f"{by} is not an approver for this pack. permissions.yaml names: {allowed}.")


@app.command(name="approve")
def approve_command(
    pack: PackArgument,
    reference: Annotated[str, typer.Argument(help="The proposal, or enough of its reference.")],
    by: ByOption = "",
    note: Annotated[str, typer.Option(help="Anything worth keeping with the decision.")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what approving would do, and do nothing.")
    ] = False,
) -> None:
    """Approve a proposal and carry it out. This is where a change to the outside happens."""
    settings = load_settings()
    may_decide(pack, by)
    inbox = connect_inbox(settings, pack)
    try:
        item = inbox.get(reference)
        permissions = load_permissions(pack)
        # The ladder still decides. A pack that allows this action only to draft never
        # agreed to it being carried out, whoever says yes.
        if not permissions.allows(item.proposal.action, "approve"):
            level = permissions.level(item.proposal.action) or "no level"
            raise fail(
                f"{item.proposal.action} has {level} in this pack's permissions.yaml, so an "
                "approval cannot carry it out. Raise it to approve, or reject this."
            )
        if dry_run:
            typer.secho("Dry run. Approving this would:", fg=typer.colors.YELLOW)
            typer.echo(f"  {item.proposal.sentence}")
            typer.echo(f"  in {connect_connectors(settings, pack)[_connector_of(pack, item)].name}")
            return
        inbox.decide(item.reference, approved=True, by=by, note=note)
        connector = connect_connectors(settings, pack)[_connector_of(pack, item)]
        done = inbox.carry_out(item.reference, connector, by=by)
    except (ApprovalError, PackError) as err:
        raise fail(str(err)) from err
    typer.secho(f"Done: {done.proposal.sentence}", fg=typer.colors.GREEN)
    if done.change:
        typer.echo(f"Reference {done.change.reference}. Undo with:")
        typer.echo(f"  vardex undo {pack} {done.reference} --by {by}")


def _connector_of(pack: Path, item: Item) -> str:
    """Which outside system an item's action is carried out in, from the ontology."""
    action = load_knowledge(pack).ontology.action(item.proposal.action)
    return action.connector


@app.command(name="reject")
def reject_command(
    pack: PackArgument,
    reference: Annotated[str, typer.Argument(help="The proposal, or enough of its reference.")],
    by: ByOption = "",
    note: Annotated[str, typer.Option(help="Why. Worth writing: it is what trains the pack.")] = "",
) -> None:
    """Say no to a proposal. Nothing is changed, and the reason is kept."""
    settings = load_settings()
    may_decide(pack, by)
    try:
        item = connect_inbox(settings, pack).decide(reference, approved=False, by=by, note=note)
    except ApprovalError as err:
        raise fail(str(err)) from err
    typer.secho(f"Rejected {item.reference}.", fg=typer.colors.YELLOW)


@app.command(name="undo")
def undo_command(
    pack: PackArgument,
    reference: Annotated[str, typer.Argument(help="The change, or enough of its reference.")],
    by: ByOption = "",
) -> None:
    """Put a carried-out change back as it was."""
    settings = load_settings()
    may_decide(pack, by)
    inbox = connect_inbox(settings, pack)
    try:
        item = inbox.get(reference)
        connector = connect_connectors(settings, pack)[_connector_of(pack, item)]
        undone = inbox.undo(item.reference, connector, by=by)
    except (ApprovalError, PackError) as err:
        raise fail(str(err)) from err
    typer.secho(f"Undone: {undone.proposal.sentence}", fg=typer.colors.GREEN)
