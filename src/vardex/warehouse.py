"""The warehouse: a DuckDB file rebuilt from a pack's landed files and its dbt project.

A build never touches the warehouse people are reading. It loads the landed files into raw
tables in a new file, runs the pack's dbt project on it (models, then tests), checks the
pack's metrics, and only then puts the new file in place of the old one.
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import httpx2
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vardex.ingest import (
    IngestError,
    Source,
    describe_statuses,
    fetch_file,
    fetch_keys,
    fetched_at,
    is_due,
    landed,
    load_ingest,
    prune,
)
from vardex.packs import PackError, load_pack, read_yaml

Say = Callable[[str], None]  # prints one line of progress
# When a row's file was fetched, from its name: 2026-10-05T061502Z.csv.gz
FETCHED_AT = "strptime(split_part(parse_filename(filename), '.', 1), '%Y-%m-%dT%H%M%SZ')"
DBT = [sys.executable, "-c", "import sys; from dbt.cli.main import cli; sys.exit(cli())"]
KEYED_COLUMNS = "{'key': 'VARCHAR', 'status': 'INTEGER', 'body': 'JSON', 'error': 'VARCHAR'}"


class Metric(BaseModel):
    """One metric in metrics.yaml: a name, what it means, and the SQL that computes it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    table: str  # the model it is computed from
    sql: str  # an aggregate expression over that table, such as sum(revenue)
    unit: str  # what the number is counted in, such as NOK or companies
    notes: list[str] = []  # what to watch for when using it


class SemanticLayer(BaseModel):
    """The contents of warehouse/metrics.yaml: every metric a question may be answered with."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[Metric] = Field(min_length=1)

    @model_validator(mode="after")
    def names_are_unique(self) -> "SemanticLayer":
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("every metric needs its own name")
        return self


class NodeResult(BaseModel):
    """dbt's result for one model, test or source, from run_results.json or sources.json."""

    unique_id: str  # such as test.norwegian_companies.currency_is_a_known_code
    status: str  # success, error, pass, warn, fail, skipped or runtime error
    message: str | None = None
    failures: int | None = None  # for a test: the rows that broke it
    max_loaded_at: datetime | None = None  # for a source: its newest _fetched_at

    @property
    def name(self) -> str:
        """The kind and name, such as `test currency_is_a_known_code` or `source units`."""
        kind, _package, *rest = self.unique_id.split(".")  # a test's id ends in a hash
        return f"{kind} {rest[-1] if kind == 'source' else rest[0]}"


class DbtResults(BaseModel):
    results: list[NodeResult]


class WarehouseError(Exception):
    """Raised when a build fails. The warehouse in place is left as it was."""


@dataclass
class Build:
    """Where the new warehouse is, and what the user should hear about it."""

    path: Path
    warnings: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # sources older than the pack allows


def count(n: int, thing: str) -> str:
    """A number and a noun, such as `1 row` or `5,768 rows`."""
    return f"{n:,} {thing}{'' if n == 1 else 's'}"


def size(n_bytes: int) -> str:
    """A file size such as `154.8 MB` or `3.9 kB`."""
    return f"{n_bytes / 1e6:,.1f} MB" if n_bytes >= 100_000 else f"{n_bytes / 1e3:,.1f} kB"


def load_metrics(pack: Path) -> SemanticLayer:
    """Read and validate warehouse/metrics.yaml in a pack folder."""
    path = pack / "warehouse" / "metrics.yaml"
    if not path.is_file():
        raise PackError(f"No warehouse/metrics.yaml in {pack}")
    return read_yaml(path, SemanticLayer, "metrics file")


def warehouse_path(data_dir: Path, pack_name: str) -> Path:
    """Where a pack's warehouse lives."""
    return data_dir / pack_name / "warehouse.duckdb"


def load_raw(con: duckdb.DuckDBPyConnection, source: Source, files: list[Path]) -> int:
    """Load landed files into the table raw.<source>, as they are, and count its rows.

    Text stays text: a code such as 0301 must not become the number 301. Typing the columns
    is the job of the pack's staging models.
    """
    names = [str(path) for path in files]
    if source.keys:
        reader = f"read_json(?, format='newline_delimited', columns={KEYED_COLUMNS}, filename=true)"
    elif source.format == "csv":
        reader = "read_csv(?, all_varchar=true, filename=true)"
    else:
        reader = "read_json(?, filename=true)"
    con.execute(
        f"create or replace table raw.{source.name} as "
        f"select * exclude (filename), {FETCHED_AT} as _fetched_at from {reader}",
        [names],
    )
    return con.execute(f"select count(*) from raw.{source.name}").fetchone()[0]


def run_dbt(command: list[str], project: Path, warehouse: Path, work: Path) -> list[NodeResult]:
    """Run one dbt command on the pack's dbt project against a warehouse file, and return
    dbt's result for every model, test or source.

    dbt runs in a process of its own, so it closes the warehouse when it is done, and its
    results are read from the JSON files it writes, as dbt documents.
    """
    process = subprocess.run(
        [
            *DBT, *command,
            "--project-dir", str(project),
            "--profiles-dir", str(project),
            "--target-path", str(work / "target"),
            "--log-path", str(work / "logs"),
            "--no-send-anonymous-usage-stats",
            "--no-version-check",
            "--quiet",
        ],
        env={**os.environ, "VARDEX_WAREHOUSE": str(warehouse)},  # profiles.yml reads it
        capture_output=True,
        text=True,
    )  # fmt: skip
    results = work / "target" / ("sources.json" if command[0] == "source" else "run_results.json")
    if not results.is_file():  # dbt stopped before it ran anything, such as on a SQL typo
        output = (process.stdout + process.stderr).strip()
        raise WarehouseError(f"dbt {' '.join(command)} could not run:\n{output}")
    return DbtResults.model_validate_json(results.read_text(encoding="utf-8")).results


def check_metrics(con: duckdb.DuckDBPyConnection, layer: SemanticLayer) -> list[str]:
    """Every metric whose SQL does not run on the warehouse, with DuckDB's reason."""
    problems = []
    for metric in layer.metrics:
        try:
            con.execute(f"select {metric.sql} as {metric.name} from {metric.table} where false")
        except duckdb.Error as err:
            problems.append(f"{metric.name}: {err}")
    return problems


def fetch_and_load(
    con: duckdb.DuckDBPyConnection,
    client: httpx2.Client,
    source: Source,
    folder: Path,
    now: datetime,
    refresh: bool,
    say: Say,
) -> str | None:
    """Fetch one source if it is due, then load its landed files. A failed fetch falls back
    on the newest earlier file; the reason is returned as a warning."""
    warning = None
    if refresh or is_due(source, folder, now):
        try:
            if source.keys:
                keys = [str(row[0]) for row in con.execute(source.keys).fetchall()]
                statuses = fetch_keys(client, source, keys, folder, now)
                say(f"{source.name}: fetched {len(keys):,} keys: {describe_statuses(statuses)}")
            else:
                path = fetch_file(client, source, folder, now)
                say(f"{source.name}: fetched {size(path.stat().st_size)}")
            prune(source, folder)
        except (IngestError, duckdb.Error) as err:
            if not landed(folder):
                raise WarehouseError(str(err)) from err
            warning = f"{err}; using the copy from {fetched_at(landed(folder)[-1]):%Y-%m-%d}"
    files = landed(folder) if source.history else landed(folder)[-1:]
    rows = load_raw(con, source, files)
    say(f"{source.name}: {count(rows, 'row')} in raw.{source.name}")
    return warning


def build(
    pack: Path,
    data_dir: Path,
    client: httpx2.Client,
    *,
    refresh: bool = False,
    now: datetime | None = None,
    say: Say = lambda line: None,
) -> Build:
    """Fetch what is due, rebuild the pack's warehouse from scratch, test it, and put it in
    place. Raises WarehouseError, leaving the old warehouse untouched, if anything fails."""
    manifest = load_pack(pack)
    spec = load_ingest(pack)
    layer = load_metrics(pack)  # read every pack file first, so a typo fails before any fetch
    now = now or datetime.now(UTC)
    home = data_dir / manifest.name
    work = home.resolve() / "build"  # absolute: dbt reads relative paths from the project
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    new = work / "warehouse.duckdb"  # same file name as the real one: DuckDB names it after it

    result = Build(path=warehouse_path(data_dir, manifest.name))
    with duckdb.connect(new) as con:
        con.execute("create schema raw")
        for source in spec.sources:
            folder = home / "landing" / source.name
            warning = fetch_and_load(con, client, source, folder, now, refresh, say)
            if warning:
                result.warnings.append(warning)
    failures = []
    nodes = run_dbt(["build"], pack / "warehouse", new, work)
    for node in nodes:
        detail = count(node.failures, "failing row") if node.failures else node.message
        if node.status in ("error", "fail", "runtime error"):
            failures.append(f"{node.name}: {detail}")
        elif node.status == "warn":
            result.warnings.append(f"{node.name}: {detail}")
    if failures:
        raise WarehouseError("dbt build failed:\n" + "\n".join(f"  {f}" for f in failures))
    built = sum(node.unique_id.startswith("model.") for node in nodes)
    passed = sum(node.status == "pass" for node in nodes)
    say(f"dbt build: {count(built, 'model')} built, {count(passed, 'test')} passed")

    for node in run_dbt(["source", "freshness"], pack / "warehouse", new, work):
        if node.status != "pass":
            when = f"{node.max_loaded_at:%Y-%m-%d %H:%M}" if node.max_loaded_at else "never"
            result.stale.append(f"{node.name} ({node.status}): last fetched {when}")

    with duckdb.connect(new, read_only=True) as con:
        problems = check_metrics(con, layer)
    if problems:
        raise WarehouseError("metrics that do not run:\n" + "\n".join(f"  {p}" for p in problems))
    say(f"metrics: all {len(layer.metrics)} run on the new warehouse")

    os.replace(new, result.path)  # one step: readers see the old warehouse or the new one
    return result


def query(path: Path, sql: str) -> duckdb.DuckDBPyRelation:
    """Run SQL on a warehouse, read-only."""
    if not path.is_file():
        raise WarehouseError(f"No warehouse at {path}. Run vardex ingest first.")
    return duckdb.connect(path, read_only=True).sql(sql)
