"""The vardex command-line tool."""

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from vardex import __version__, evals, text_to_sql
from vardex.anthropic_client import AnthropicClient
from vardex.calllog import log_call
from vardex.config import Settings, load_settings
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.extraction import ExtractionError, build_request, extract, load_extraction
from vardex.ingest import http_client
from vardex.llm import Answer, Conversation, Effort, Ledger, Message, ModelClient, ModelError
from vardex.llm import ask as ask_model
from vardex.packs import PackError, load_pack
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
from vardex.provenance import Figure
from vardex.replay import (
    NotRecorded,
    Recording,
    RecordingClient,
    RecordingWarehouse,
    ReplayClient,
    ReplayWarehouse,
)
from vardex.semantic import Knowledge, check, load_knowledge
from vardex.sql import QueryError, Result, Warehouse, format_table, format_value
from vardex.text_to_sql import AnswerError
from vardex.warehouse import WarehouseError, build, warehouse_path

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


@app.command(name="extract")
def extract_command(
    pack: PackArgument,
    name: Annotated[str, typer.Argument(help="An extraction in the pack, such as key-figures.")],
    files: Annotated[
        list[Path], typer.Argument(help="Text files to extract from.", exists=True, dir_okay=False)
    ],
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
    """Extract values from documents. Prints one JSON line per document that passes the checks."""
    try:
        spec = load_extraction(pack, name)
    except PackError as err:
        raise fail(str(err)) from err
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
        try:
            result = extract(
                ledger, spec, file.read_text(encoding="utf-8"), effort=effort, repairs=repairs
            )
        except ExtractionError as err:
            failed += 1
            typer.secho(f"{file.name}: {err}", fg=typer.colors.RED, err=True)
            answers = err.answers
        except ModelError as err:
            for answer in ledger.answers:
                record(answer, stats)
            raise fail(f"{file.name}: {err}") from err
        else:
            # default=str writes amounts as exact strings, never as floats
            typer.echo(
                json.dumps({"file": file.name, **result.values}, default=str, ensure_ascii=False)
            )
            answers = result.answers
        for answer in answers:
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
    finally:
        connection.close()
    if problems:
        raise fail(f"{connection.name}:\n" + "\n".join(f"  {p}" for p in problems))
    typer.secho(
        f"OK  {connection.name}: {len(knowledge.tables)} models, {len(knowledge.metrics)} "
        f"metrics, {len(knowledge.ontology.objects)} objects, "
        f"{len(knowledge.ontology.links)} links, {len(knowledge.questions.examples)} examples",
        fg=typer.colors.GREEN,
    )


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
        raise fail(str(err)) from err
    spent: list[Answer] = []  # the calls for the question in hand, still logged if one fails
    try:
        if show_prompt:
            request = text_to_sql.build_request(
                knowledge, connection, question, effort=effort, max_rows=max_rows
            )
            typer.echo(f"=== system ===\n{request.system}\n\n=== user ===\n{question}")
            return
        client = connect(settings)
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
            )
            for call in reply.answers:
                record(call, stats)
            asked_back = len(earlier) // 2
            if reply.kind != "clarify" or no_input or asked_back == MAX_ASKED_BACK:
                break
            if not interactive():
                break
            typer.echo(reply.text)
            earlier += [Message("user", question), Message("assistant", reply.answers[-1].text)]
            question = typer.prompt(">", prompt_suffix=" ")
    except (ModelError, QueryError) as err:
        for call in spent:
            record(call, stats)
        raise fail(str(err)) from err
    except AnswerError as err:
        for call in err.answers:
            record(call, stats)
        if err.result:
            show_sources(err.result, [], err=True)
        raise fail(str(err)) from err
    finally:
        connection.close()

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


def eval_connections(
    pack: Path, suite: str, settings: Settings, url: str | None, record: bool, replay: bool
) -> tuple[ModelClient, Warehouse, Recording | None]:
    """The model and warehouse an eval runs on: live, live and recorded, or replayed."""
    path = evals.recording_path(pack, suite)
    if replay:
        if not path.is_file():
            raise PackError(f"No recording at {path}. Make one with --record.")
        recording = Recording.load(path)
        return ReplayClient(recording), ReplayWarehouse(recording), recording
    connection = connect_warehouse(url, pack, settings)
    try:
        client = connect(settings)
    except ModelError:
        connection.close()
        raise
    if not record:
        return client, connection, None
    recording = Recording(
        client.model, connection.name, connection.dialect, date.today().isoformat()
    )
    return RecordingClient(client, recording), RecordingWarehouse(connection, recording), recording


def eval_suite(
    pack: Path,
    name: str,
    suite: evals.Suite,
    knowledge: Knowledge,
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
    problems = evals.check_suite(suite, knowledge)
    if problems:
        raise fail(f"Suite {name} cannot be used:\n" + "\n".join(f"  {p}" for p in problems))
    try:
        client, connection, recording = eval_connections(
            pack, name, settings, warehouse, record_run, replay_run
        )
    except (PackError, QueryError, ModelError) as err:
        raise fail(str(err)) from err
    graders: list[evals.Grader] = [evals.grade_behaviour, evals.grade_rows, evals.Judge(client)]

    started: list[str] = []  # the ids of the cases run so far, to name one that stops the run

    def run_case(case: evals.Case) -> evals.CaseResult:
        started.append(case.id)
        return evals.answer_case(client, connection, knowledge, suite, case, graders, effort=effort)

    def show(result: evals.CaseResult) -> None:
        mark, colour = ("ok", "green") if result.passed else ("FAIL", "red")
        typer.secho(f"{mark:<5}{result.case.id}", fg=colour, err=True)
        if not replay_run:
            for call in result.answers:
                record(call, stats=False)

    try:
        report = evals.run(
            name, suite, client.model, run_case, only=only, repeat=repeat, on_result=show
        )
    except NotRecorded as err:
        recorded_on = recording.recorded_on if recording else "?"
        raise fail(
            f"{name}, case {started[-1]}: {err}. A prompt, the pack, the code or the model has "
            f"changed since the recording of {recorded_on}: record it again with --record."
        ) from err
    except (PackError, QueryError, ModelError) as err:
        raise fail(f"{name}: {err}") from err
    finally:
        connection.close()
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
        knowledge = load_knowledge(pack)
        found = evals.load_suites(pack)
    except PackError as err:
        raise fail(str(err)) from err
    names = suites or list(found)
    unknown = [name for name in names if name not in found]
    if unknown or not names:
        known = ", ".join(found) or "none"
        raise fail(f"No suite {', '.join(unknown)} in {pack}. Known: {known}")
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
