"""Work that runs without being asked: what a pack declares, and what runs it.

An agent answers a question somebody asked. A workflow asks the questions. The pack says
what to look at, how often, and what may be proposed when something is wrong; the engine
decides nothing except how to run it safely.

The shape is deliberately boring. A workflow is a list of steps, in order, and each step is
one of four things the engine knows how to do:

- `watch`  — read the source's own feed of changes since the last cursor.
- `find`   — run the pack's SQL against its warehouse; every row is a case to look at.
- `brief`  — let the agent look into one case and, if the numbers justify it, draft an action.
- `report` — write down what the run found, for a person to read on Monday morning.

The model chooses nothing about the order. That is the difference between a workflow and an
agent, and it is a feature: the steps are the same every week, so a run can be compared with
last week's, resumed after a crash, and explained to somebody who was not there. Inside a
step, where the work is genuinely open-ended, the agent chooses freely.

Two rules hold the whole thing up.

**The run decides whether it is due, not the clock.** Whatever fires it — launchd, cron, a
CI schedule, a person — calls Vardex often, and Vardex works out whether this week's run has
happened yet. Every scheduler misses ticks: a laptop is shut, a hosted runner is queued for
twenty minutes at the top of the hour, a container restarts. A job that runs when it is told
runs the wrong number of times. A job that knows when it is due runs once.

**Nothing outside Vardex changes without a person.** A scheduled run has nobody to ask, so
its actions can only be drafted. They land in the inbox, where somebody approves them, and
the approval — a separate command, run by a human being — makes the change.
"""

import json
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vardex.agent import MAX_STEPS, run_agent
from vardex.agent import Run as AgentRun
from vardex.approvals import Inbox, Item
from vardex.events import Batch, FeedSpec, read_feed
from vardex.llm import ModelClient, ModelError, ToolCall
from vardex.packs import PackError, read_yaml
from vardex.permissions import Budgets, Permissions
from vardex.prompts import WATCH
from vardex.runs import Run, RunStore, Step, call_key, new_run_id, now_text
from vardex.semantic import Knowledge
from vardex.sql import QueryError, Warehouse, check_sql, format_value, transpile
from vardex.tools import Toolbox, ToolResult

if TYPE_CHECKING:  # only for the type of the client the live reader borrows
    import httpx2

WORKFLOW_DIR = "workflows"
REPORT_DIR = "reports"
STATE_DIR = "state"
NAME = r"^[a-z0-9]+(-[a-z0-9]+)*$"
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MAX_CASES = 50  # one run looks into this many cases at most, however many the SQL found
MAX_WORKERS = 4  # cases looked into at the same time; beyond this the provider is the limit


class Schedule(BaseModel):
    """When a workflow is due, in the pack's own words and the pack's own time zone."""

    model_config = ConfigDict(extra="forbid")

    every: Literal["hour", "day", "week"] = "week"
    at: str = Field(default="06:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")  # local time
    weekday: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ] = "monday"
    # A real place, not an offset: offsets do not know about summer time, and a business
    # that says "half past six" means half past six in both halves of the year.
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as err:
            raise ValueError(f"{value} is not a time zone name, such as Europe/Oslo") from err
        return value

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def hour_minute(self) -> tuple[int, int]:
        hour, minute = self.at.split(":")
        return int(hour), int(minute)

    def due_before(self, when: datetime) -> datetime:
        """The last moment this workflow should have run, at or before `when`.

        Everything else is built on this. "Is it due?" becomes "has the run that this
        moment asked for happened yet?", which is a question about work done rather than
        about the clock, and so it survives a scheduler that fires late, twice, or not at
        all. It also makes the twice-yearly clock change a non-event: the hour that happens
        twice asks for the same run both times, and the run has already happened.
        """
        local = when.astimezone(self.zone)
        hour, minute = self.hour_minute
        if self.every == "hour":
            slot = local.replace(minute=0, second=0, microsecond=0)
            return slot if slot <= local else slot - timedelta(hours=1)
        slot = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if self.every == "day":
            return slot if slot <= local else slot - timedelta(days=1)
        back = (slot.weekday() - WEEKDAYS.index(self.weekday)) % 7
        slot = slot - timedelta(days=back)
        return slot if slot <= local else slot - timedelta(days=7)

    def next_after(self, when: datetime) -> datetime:
        """The next moment it is due, for telling a person when to expect the run."""
        step = {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(days=7)}
        return self.due_before(when) + step[self.every]

    def period(self, when: datetime) -> str:
        """A name for the slot a moment belongs to, such as 2026-W44.

        It names the piece of work. Two attempts at the same week's watch propose the same
        change under the same key, so the inbox holds one proposal; next week's attempt is a
        different period, so it may raise the same concern again. Without this, a monitor
        either fills the inbox with duplicates or goes quiet about a problem for ever.
        """
        local = self.due_before(when)
        if self.every == "hour":
            return local.strftime("%Y-%m-%dT%H")
        if self.every == "day":
            return local.strftime("%Y-%m-%d")
        year, week, _ = local.isocalendar()
        return f"{year}-W{week:02d}"

    def describe(self) -> str:
        if self.every == "hour":
            return f"every hour, on the hour ({self.timezone})"
        if self.every == "day":
            return f"every day at {self.at} ({self.timezone})"
        return f"every {self.weekday} at {self.at} ({self.timezone})"


class WatchStep(BaseModel):
    """Read the source's feed of changes, and keep the ones about things the pack follows."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NAME)
    uses: Literal["watch"]
    feed: FeedSpec
    # SQL giving one column of keys the pack cares about. Everything else in the feed is
    # dropped, which is what turns a register's whole firehose into a handful of changes.
    follow: str | None = None


class FindStep(BaseModel):
    """Run the pack's own SQL. Every row is a case for the run to look into."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NAME)
    uses: Literal["find"]
    about: str  # the ontology object each row is about, such as Company
    sql: str  # first column the object's key, a `reason` column, and any figures besides
    changed_only: bool = False  # only cases a `watch` step saw change
    limit: int = Field(default=MAX_CASES, ge=1, le=500)


class BriefStep(BaseModel):
    """Let the agent look into each case, write a note, and draft an action if it is right."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NAME)
    uses: Literal["brief"]
    question: str  # asked once per case; {key}, {label}, {reason} and the figures fill in
    propose: str | None = None  # the action it may draft, by its name in the ontology
    max_steps: int = Field(default=MAX_STEPS, ge=1, le=40)
    cases: int = Field(default=MAX_CASES, ge=1, le=500)  # at most this many per run
    # Cases looked into at the same time. They are independent — one company each — so this
    # is wall-clock for free, up to the point where the rate limit answers instead.
    workers: int = Field(default=1, ge=1, le=MAX_WORKERS)


class TidyStep(BaseModel):
    """Close what nobody answered, and say what is late.

    A monitor that only ever adds to the inbox teaches people to stop reading it. This step
    is the other direction: proposals older than the pack's `expire_after_days` are closed,
    and anything past its deadline is put at the top of the report with the name of whoever
    is supposed to be chasing it. Escalation is not a notification system; it is making the
    overdue thing impossible to miss in the place people already look.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NAME)
    uses: Literal["tidy"]


class ReportStep(BaseModel):
    """Write down what the run found, as something a person reads over coffee."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=NAME)
    uses: Literal["report"]
    title: str


AnyStep = Annotated[
    WatchStep | FindStep | BriefStep | TidyStep | ReportStep, Field(discriminator="uses")
]


class WorkflowSpec(BaseModel):
    """The contents of a pack's workflows/<name>.yaml."""

    model_config = ConfigDict(extra="forbid")

    description: str
    steps: list[AnyStep] = Field(min_length=1)
    schedule: Schedule | None = None  # without one, it runs only when somebody asks
    budget: Budgets = Budgets()  # what one run of this workflow may spend
    name: str = ""  # filled in from the file name

    @model_validator(mode="after")
    def ids_are_unique(self) -> "WorkflowSpec":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("every step needs its own id")
        return self

    def period(self, when: datetime) -> str:
        """The piece of work this moment belongs to. A workflow with no schedule is its own
        period every time, because nobody can say what a run by hand is one of."""
        return self.schedule.period(when) if self.schedule else when.strftime("%Y-%m-%dT%H%M%S")


# Cases may be looked into in threads, and the journal is one dictionary shared between them.
JOURNAL = threading.Lock()


class WorkflowError(Exception):
    """Raised when a step cannot run at all: bad SQL, or a feed that has changed shape."""


def live_feed(http: Callable[[], "httpx2.Client"]) -> Callable[[FeedSpec, str | None], Batch]:
    """Reading a feed for real: one client per read, closed afterwards."""

    def read(spec: FeedSpec, cursor: str | None) -> Batch:
        with http() as client:
            return read_feed(client, spec, cursor)

    return read


def workflows_dir(pack: Path) -> Path:
    return pack / WORKFLOW_DIR


def workflow_names(pack: Path) -> list[str]:
    """Every workflow the pack declares, by the name of its file."""
    folder = workflows_dir(pack)
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def load_workflow(pack: Path, name: str) -> WorkflowSpec:
    """Read and validate one workflows/<name>.yaml. The file name is the workflow's name."""
    path = workflows_dir(pack) / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(workflow_names(pack)) or "none"
        raise PackError(f"No workflow {name} in {pack}. This pack has: {known}.")
    return read_yaml(path, WorkflowSpec, "workflow file").model_copy(update={"name": name})


def load_workflows(pack: Path) -> list[WorkflowSpec]:
    return [load_workflow(pack, name) for name in workflow_names(pack)]


def check_workflow(spec: WorkflowSpec, knowledge: Knowledge, permissions: Permissions) -> list[str]:
    """Everything wrong with a workflow that can be found without running it.

    This is the same bargain as the rest of the pack format: a mistake in YAML is found by
    `vardex check` on a Friday afternoon, not by a run that fails at half past six on a
    Monday morning with nobody watching.
    """
    problems = []
    described = set(knowledge.tables)
    objects = {o.name for o in knowledge.ontology.objects}
    actions = {a.name for a in knowledge.ontology.actions}
    watching = [step for step in spec.steps if isinstance(step, WatchStep)]
    for step in spec.steps:
        where = f"{spec.name} step {step.id}"
        if isinstance(step, WatchStep) and step.follow:
            problems += [
                f"{where} follow: {problem}"
                for problem in check_sql(step.follow, knowledge.questions.dialect, described)
            ]
        if isinstance(step, FindStep):
            if step.about not in objects:
                problems.append(f"{where}: there is no object {step.about} in the ontology")
            if step.changed_only and not watching:
                problems.append(f"{where}: changed_only needs a watch step before it")
            problems += [
                f"{where}: {problem}"
                for problem in check_sql(step.sql, knowledge.questions.dialect, described)
            ]
        if isinstance(step, BriefStep) and step.propose:
            if step.propose not in actions:
                problems.append(f"{where}: there is no action {step.propose} in the ontology")
            elif not permissions.allows(step.propose, "draft"):
                level = permissions.level(step.propose) or "no level"
                problems.append(
                    f"{where}: {step.propose} has {level} in permissions.yaml, so this "
                    "workflow could never propose anything"
                )
    if not any(isinstance(step, ReportStep) for step in spec.steps):
        problems.append(f"{spec.name}: no report step, so a run leaves nothing to read")
    return problems


@dataclass
class Watermarks:
    """Where each feed was read up to. One small file per workflow, beside its runs."""

    path: Path

    def read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, name: str) -> str | None:
        value = self.read().get(name)
        return str(value) if value is not None else None

    def set(self, name: str, cursor: str | None) -> None:
        """Save a cursor, whole, through a temporary file.

        The cursor moves only after the changes it covers are in the run's own file. Moving
        it first would lose changes the moment anything downstream failed; never moving it
        would read them again, which is why work that writes carries an idempotency key.
        """
        if cursor is None:
            return
        state = self.read() | {name: cursor}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".writing")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.path)


@dataclass(frozen=True)
class Case:
    """One thing the run is looking into: a row from a `find` step."""

    key: str
    label: str
    reason: str
    facts: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "reason": self.reason, "facts": self.facts}

    def filled(self, template: str) -> str:
        """The step's question with this case's own numbers in it."""
        values = {"key": self.key, "label": self.label, "reason": self.reason, **self.facts}
        try:
            return template.format(**values)
        except KeyError as err:
            raise WorkflowError(
                f"the question asks for {err}, which this case does not have. It has: "
                f"{', '.join(values)}."
            ) from err


@dataclass
class Connections:
    """Everything a run can reach. Built by whatever starts the run, never by the runner."""

    pack: str  # the pack's name, as pack.yaml gives it
    folder: str  # the pack's folder, as it was typed, so the report can print a command
    knowledge: Knowledge
    permissions: Permissions
    warehouse: Warehouse
    client: ModelClient  # already wrapped in the pack's budget
    inbox: Inbox
    store: RunStore
    marks: Watermarks
    reports: Path
    toolbox: Callable[[dict[str, str]], Toolbox]  # a toolbox for one case, with its context
    # How this run reads a feed: live, live and recorded, or replayed from a recording. The
    # runner does not know which, which is what makes a recorded run repeatable in CI.
    feed: Callable[[FeedSpec, str | None], Batch]


def is_due(spec: WorkflowSpec, store: RunStore, now: datetime | None = None) -> bool:
    """Whether this workflow's turn has come and gone without a run that finished.

    A workflow with no schedule is never due: somebody has to ask for it.
    """
    if spec.schedule is None:
        return False
    moment = now or datetime.now(UTC)
    last = store.last_done(spec.name)
    if last is None:
        return True
    try:
        since = datetime.fromisoformat(last.started_at)
    except ValueError:
        return True
    return since < spec.schedule.due_before(moment)


def start(spec: WorkflowSpec, connections: Connections, *, trigger: str, dry_run: bool) -> Run:
    """A new run, with a step for each of the workflow's steps, saved before anything runs.

    Saved first, on purpose. A run that dies in its first step has still left a file saying
    what it was trying to do, and that file is the only evidence anybody will have.
    """
    run = Run(
        id=new_run_id(spec.name),
        pack=connections.pack,
        workflow=spec.name,
        trigger=trigger,
        started_at=now_text(),
        dry_run=dry_run,
        steps=[Step(id=step.id, uses=step.uses) for step in spec.steps],
    )
    run.facts["period"] = spec.period(datetime.now(UTC))
    connections.store.save(run)
    return run


class StopRun(Exception):
    """Raised inside a step to stop the whole run without failing it: out of budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def run_workflow(
    spec: WorkflowSpec,
    run: Run,
    connections: Connections,
    *,
    on_step: Callable[[Step], None] | None = None,
    on_call: Callable[[ToolCall, ToolResult], None] | None = None,
) -> Run:
    """Run the workflow's steps in order, saving after every one, and return the run.

    A step that has already finished is skipped, which is all resuming is.

    A run that stops because it is out of budget is `stopped`, not `failed`. Those are
    different words to whoever reads the list on Monday: one needs a decision about money,
    the other needs somebody to fix something. Either way the report is still written, so a
    run that got half way says what it found.
    """
    run.state = "running"
    if _has_run_before(run):
        run.attempts += 1  # this is a resume, and the count belongs in the record
    connections.store.save(run)
    for spec_step in spec.steps:
        record = run.step(spec_step.id)
        if record is None or record.finished:
            continue
        if isinstance(spec_step, ReportStep):
            continue  # written at the end, whatever happened before it
        record.state = "running"
        record.started_at = now_text()
        connections.store.save(run)
        try:
            _run_step(spec, spec_step, run, record, connections, on_call=on_call)
        except StopRun as stop:
            _end(run, record, "stopped", "stopped", stop.reason, connections, spec, on_step)
            return run
        except (WorkflowError, QueryError, PackError, ModelError, OSError) as err:
            _end(run, record, "failed", "failed", str(err), connections, spec, on_step)
            return run
        record.state = "done"
        record.ended_at = now_text()
        connections.store.save(run)
        if on_step is not None:
            on_step(record)
    # Finished before the report is written, so the report can say so. Nothing is left to
    # go wrong but the writing of the file itself, and that is recorded on its own step.
    run.state = "done"
    run.note = ""  # whatever stopped an earlier attempt is history; the steps still say so
    run.ended_at = now_text()
    connections.store.save(run)
    _write_report(spec, run, connections, on_step)
    connections.store.save(run)
    return run


def _has_run_before(run: Run) -> bool:
    return any(step.started_at for step in run.steps)


def _end(
    run: Run,
    record: Step,
    step_state: str,
    run_state: str,
    reason: str,
    connections: Connections,
    spec: WorkflowSpec,
    on_step: Callable[[Step], None] | None,
) -> None:
    """Close a run that is not going to finish: record why, then still write the report."""
    record.state = step_state  # type: ignore[assignment]
    record.note = reason
    record.ended_at = now_text()
    run.state = run_state  # type: ignore[assignment]
    run.note = f"{record.id}: {reason}"
    run.ended_at = now_text()
    connections.store.save(run)
    _write_report(spec, run, connections, on_step)
    connections.store.save(run)


def _write_report(
    spec: WorkflowSpec,
    run: Run,
    connections: Connections,
    on_step: Callable[[Step], None] | None = None,
) -> None:
    """Write the report step, every attempt.

    Unlike every other step, this one is written again on a resume rather than skipped. It
    costs nothing, it is the same file, and a report left over from the attempt that stopped
    would tell a person the run had stopped when it has since finished.
    """
    for spec_step in spec.steps:
        if not isinstance(spec_step, ReportStep):
            continue
        record = run.step(spec_step.id)
        if record is None:
            continue
        record.started_at = record.started_at or now_text()
        try:
            _report(spec, spec_step, run, record, connections)
            record.state = "done"
        except OSError as err:  # a report nobody can write must not hide the run
            record.state = "failed"
            record.note = str(err)
        record.ended_at = now_text()
        connections.store.save(run)
        if on_step is not None:
            on_step(record)


def _run_step(
    spec: WorkflowSpec,
    step: AnyStep,
    run: Run,
    record: Step,
    connections: Connections,
    *,
    on_call: Callable[[ToolCall, ToolResult], None] | None = None,
) -> None:
    if isinstance(step, WatchStep):
        _watch(step, run, record, connections)
    elif isinstance(step, FindStep):
        _find(step, run, record, connections)
    elif isinstance(step, BriefStep):
        _brief(spec, step, run, record, connections, on_call=on_call)
    elif isinstance(step, TidyStep):
        _tidy(step, run, record, connections)


def keys_from_sql(warehouse: Warehouse, knowledge: Knowledge, sql: str) -> set[str]:
    """One column of keys from the pack's own SQL, as strings."""
    result = warehouse.run(
        transpile(sql, knowledge.questions.dialect, warehouse.dialect), max_rows=100_000
    )
    return {str(row[0]) for row in result.rows if row[0] is not None}


def _watch(step: WatchStep, run: Run, record: Step, connections: Connections) -> None:
    """Read the feed from the saved cursor, keep what the pack follows, move the cursor."""
    cursor = connections.marks.get(step.id)
    batch = connections.feed(step.feed, cursor)
    keys = batch.keys
    followed = None
    if step.follow:
        followed = keys_from_sql(connections.warehouse, connections.knowledge, step.follow)
        keys = [key for key in keys if key in followed]
    run.facts["changed"] = sorted(set(run.facts.get("changed", [])) | set(keys))
    run.facts[f"feed:{step.id}"] = {
        "read": batch.read,
        "kept": len(keys),
        "pages": batch.pages,
        "more": batch.more,
        "cursor": batch.cursor,
        "followed": len(followed) if followed is not None else None,
    }
    record.items = len(keys)
    record.note = f"{batch.read} changes read, {len(keys)} about something we follow" + (
        ", and more waiting" if batch.more else ""
    )
    # Saved before the cursor moves: the changes are in the run's file now, so a crash
    # after this line loses nothing that resuming cannot pick up.
    connections.store.save(run)
    if not run.dry_run:
        # A dry run changes nothing, and the cursor is a change. Moving it would mean a
        # rehearsal quietly consumed the week's changes and the real run saw none of them.
        connections.marks.set(step.id, batch.cursor)


def _find(step: FindStep, run: Run, record: Step, connections: Connections) -> None:
    """Run the pack's SQL and turn every row into a case."""
    knowledge = connections.knowledge
    problems = check_sql(step.sql, knowledge.questions.dialect, set(knowledge.tables))
    if problems:
        raise WorkflowError(f"the SQL may not run: {'; '.join(problems)}")
    obj = knowledge.ontology.object(step.about)
    result = connections.warehouse.run(
        transpile(step.sql, knowledge.questions.dialect, connections.warehouse.dialect),
        max_rows=step.limit + 1,
    )
    if "reason" not in result.columns:
        raise WorkflowError(
            "a find step's SQL must return a `reason` column: one sentence saying why this "
            f"row needs a look. It returned: {', '.join(result.columns)}."
        )
    key_column = result.columns[0]
    label_column = obj.label if obj.label in result.columns else key_column
    cases = []
    for row in result.rows[: step.limit]:
        values = dict(zip(result.columns, row, strict=True))
        key = str(values[key_column])
        cases.append(
            Case(
                key=key,
                label=str(values.get(label_column, key)),
                reason=str(values.get("reason", "")),
                facts={
                    name: format_value(value)
                    for name, value in values.items()
                    if name not in (key_column, "reason", label_column)
                },
            )
        )
    if step.changed_only:
        changed = set(run.facts.get("changed", []))
        cases = [case for case in cases if case.key in changed]
    run.facts["cases"] = run.facts.get("cases", []) + [case.as_dict() for case in cases]
    over = len(result.rows) - step.limit
    record.items = len(cases)
    record.note = f"{len(cases)} case(s)" + (
        f", and at least {over} more over the limit of {step.limit}" if over > 0 else ""
    )


def journalled(box: Toolbox, run: Run) -> Toolbox:
    """A toolbox that will not do the same writing call twice, however often it is asked.

    This is the whole of idempotency, and it is a dozen lines. Every call that changed
    something outside Vardex — or that put a proposal in the inbox — is kept in the run's
    journal under a key made from the tool's name and its arguments. A resumed run that asks
    for the same change is handed the first answer back, and nothing moves twice.

    Reading calls are not journalled. They are cheap, they change nothing, and a resumed run
    is better off seeing today's numbers than last night's.
    """
    original = box.run

    def run_once(call: ToolCall) -> ToolResult:
        key = call_key(call.name, call.arguments)
        with JOURNAL:
            remembered = run.journal.get(key)
        if remembered is not None:
            return ToolResult(
                content=remembered["content"] + "\n(Done by an earlier attempt at this run.)",
                structured=remembered.get("structured"),
                wrote=False,  # it was done before, by whichever attempt recorded it
            )
        result = original(call)
        if result.wrote or (result.structured or {}).get("state") == "proposed":
            with JOURNAL:
                run.journal[key] = {"content": result.content, "structured": result.structured}
        return result

    box.run = run_once  # type: ignore[method-assign]
    return box


def _brief(
    spec: WorkflowSpec,
    step: BriefStep,
    run: Run,
    record: Step,
    connections: Connections,
    *,
    on_call: Callable[[ToolCall, ToolResult], None] | None = None,
) -> None:
    """Look into each case with the agent, saving the run after every one of them.

    One case at a time unless the pack asks for more, which it may because the cases are
    independent: one company each.
    """
    cases = [Case(**case) for case in run.facts.get("cases", [])][: step.cases]
    if not cases:
        record.note = "nothing to look into"
        return
    notes: dict[str, str] = run.facts.setdefault("notes", {})
    proposals: list[str] = run.facts.setdefault("proposals", [])
    failures: dict[str, str] = run.facts.setdefault("failures", {})
    period = run.facts.get("period", "")
    hours = connections.permissions.approvals.needed_within_hours
    spent = Decimal(record.cost_usd)
    todo = [case for case in cases if f"{step.id}:{case.key}" not in run.journal]

    def look_into(case: Case) -> tuple[Case, AgentRun]:
        """One case, in whatever thread. It touches the run only through the journal, which
        has a lock of its own; everything else is written back by the thread that started
        this one."""
        context = {
            "run": run.id,
            "step": step.id,
            "case": case.key,
            # The name of this piece of work. The same case in the same period is the same
            # work however often it is attempted, and that is what the inbox dedupes on.
            "work": f"{spec.name}:{step.id}:{case.key}:{period}",
            "why": case.reason,
            "by": run.id,
            "needed_by": (datetime.now(UTC) + timedelta(hours=hours)).isoformat(timespec="seconds"),
        }
        box = journalled(connections.toolbox(context), run)
        return case, run_agent(
            connections.client,
            box,
            case.filled(step.question),
            system=WATCH,
            max_steps=step.max_steps,
            on_call=on_call,
        )

    for case, answer in _look_into_each(todo, look_into, step.workers):
        run.spend(answer.cost_usd)
        spent += answer.cost_usd or Decimal(0)
        record.cost_usd = str(spent)
        if answer.stop == "budget":
            connections.store.save(run)
            raise StopRun(
                f"{answer.note} Looked into {len(notes)} of {len(cases)} cases; "
                f"resume with `vardex run {connections.folder} --resume {run.id}`."
            )
        if answer.answered:
            notes[case.key] = answer.answer
        else:
            failures[case.key] = f"{answer.stop}: {answer.note}"
        for _, result in answer.tool_calls:
            reference = (result.structured or {}).get("reference")
            if reference and not result.is_error and reference not in proposals:
                proposals.append(str(reference))
        run.journal[f"{step.id}:{case.key}"] = {
            "steps": len(answer.steps),
            "cost_usd": str(answer.cost_usd) if answer.cost_usd else None,
            "stop": answer.stop,
        }
        record.items = len(notes) + len(failures)
        connections.store.save(run)
    record.note = f"{len(notes)} note(s), {len(proposals)} proposal(s)" + (
        f", {len(failures)} case(s) failed" if failures else ""
    )


def _look_into_each(
    cases: list[Case],
    look_into: Callable[[Case], tuple[Case, AgentRun]],
    workers: int,
) -> Iterator[tuple[Case, AgentRun]]:
    """Each case's answer, in the order the cases came, however many threads did the work.

    One at a time unless the pack asks for more. With more, the cases run in a pool and the
    answers arrive here, in the thread that started them, in the order the `find` step put
    them in — so the run's file is written from one thread and reads the same as a run that
    was not in a hurry. A case that raises is raised here, after the ones before it have
    been recorded.
    """
    if workers < 2 or len(cases) < 2:
        yield from (look_into(case) for case in cases)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as pool:
        yield from pool.map(look_into, cases)


def _tidy(step: TidyStep, run: Run, record: Step, connections: Connections) -> None:
    """Expire what nobody answered in time, and write down what is late."""
    rules = connections.permissions.approvals
    closed = (
        []
        if run.dry_run  # a rehearsal decides nothing, not even to give up on something
        else connections.inbox.expire(after_days=rules.expire_after_days, by=run.id)
    )
    overdue = connections.inbox.overdue()
    run.facts["expired"] = [item.reference for item in closed]
    run.facts["overdue"] = [item.reference for item in overdue]
    run.facts["escalate_to"] = rules.escalate_to
    record.items = len(closed) + len(overdue)
    record.note = f"{len(closed)} expired, {len(overdue)} overdue" + (
        f", to chase: {rules.escalate_to}" if overdue and rules.escalate_to else ""
    )


def _report(
    spec: WorkflowSpec, step: ReportStep, run: Run, record: Step, connections: Connections
) -> None:
    """Write the run down as markdown, and remember where."""
    text = digest(spec, step, run, connections.inbox, connections.folder)
    connections.reports.mkdir(parents=True, exist_ok=True)
    path = connections.reports / f"{run.id}.md"
    path.write_text(text, encoding="utf-8")
    run.facts["report"] = str(path)
    record.items = 1
    record.note = str(path)


def digest(
    spec: WorkflowSpec, step: ReportStep, run: Run, inbox: Inbox, folder: str = "PACK"
) -> str:
    """What the run found, as something a person can read in two minutes.

    It leads with what needs a decision, because that is the only part with a deadline on
    it. The figures and the notes come after, and what went wrong is never left out: a
    digest that hides its failures teaches people to trust a run that should not be trusted.
    """
    cases = [Case(**case) for case in run.facts.get("cases", [])]
    notes: dict[str, str] = run.facts.get("notes", {})
    failures: dict[str, str] = run.facts.get("failures", {})
    waiting = inbox.items(state="proposed")
    mine = [item for item in waiting if item.reference in run.facts.get("proposals", [])]
    lines = [
        f"# {step.title}",
        "",
        f"- Run `{run.id}`, started by {run.trigger} at {run.started_at}",
        f"- Workflow `{spec.name}`" + (f", {spec.schedule.describe()}" if spec.schedule else ""),
        f"- Finished: **{run.state}**" + (f" — {run.note}" if run.note else ""),
        f"- Cost ${run.spent:.4f}" + (" (dry run, nothing written)" if run.dry_run else ""),
        "",
    ]
    feeds = {name: feed for name, feed in run.facts.items() if name.startswith("feed:")}
    if feeds:
        lines += ["## What changed", ""]
        for name, feed in feeds.items():
            lines.append(
                f"- `{name.split(':', 1)[1]}`: {feed['read']} changes read, "
                f"{feed['kept']} about something we follow"
                + (", and more waiting for the next run" if feed.get("more") else "")
            )
        lines.append("")
    late = [item for item in waiting if item.reference in run.facts.get("overdue", [])]
    if late:
        chase = run.facts.get("escalate_to") or "whoever owns this pack"
        lines += [f"## Late — {chase}, these are yours ({len(late)})", ""]
        lines += [
            f"- `{item.reference}` {item.proposal.label}: proposed {item.made_at}, "
            f"due {item.needed_by}"
            for item in late
        ]
        lines.append("")
    expired = run.facts.get("expired", [])
    if expired:
        lines += [
            f"## Closed, unanswered ({len(expired)})",
            "",
            "Nobody decided these in time, so they are closed. The figures they rested on are "
            "too old to act on now; the watch will raise them again if they still hold.",
            "",
        ]
        lines += [f"- `{reference}`" for reference in expired]
        lines.append("")
    lines += [f"## Needs a decision ({len(mine)})", ""]
    if mine:
        for item in mine:
            lines += [
                f"### {item.proposal.label} — `{item.reference}`",
                "",
                f"> {item.proposal.sentence}",
                "",
                f"- Why: {item.proposal.why or 'see the note below'}",
                f"- Answer by: {item.needed_by or 'no deadline'}",
                f"- Approve: `vardex approve {folder} {item.reference} --by YOUR-NAME`",
                "",
            ]
    else:
        lines += ["Nothing. No action met the bar this run.", ""]
    lines += [f"## What the run looked at ({len(cases)})", ""]
    for case in cases:
        lines += [f"### {case.label} ({case.key})", "", f"- Found because: {case.reason}"]
        if case.facts:
            lines.append(
                "- Figures: " + ", ".join(f"{name} {value}" for name, value in case.facts.items())
            )
        note = notes.get(case.key)
        if note:
            lines += ["", note.strip(), ""]
        elif case.key in failures:
            lines += ["", f"**Not looked into:** {failures[case.key]}", ""]
        else:
            lines += ["", "_This run did not reach it._", ""]
    if failures:
        lines += ["## What went wrong", ""]
        lines += [f"- {key}: {why}" for key, why in failures.items()]
        lines.append("")
    overdue = [item for item in waiting if item.overdue() and item not in late]
    if overdue:
        lines += ["## Overdue, from earlier runs", ""]
        lines += [
            f"- `{item.reference}` {item.proposal.label}: waiting since {item.made_at}, "
            f"due {item.needed_by}"
            for item in overdue
        ]
        lines.append("")
    return "\n".join(lines)


def median(values: list[Decimal]) -> Decimal:
    """The middle value, or the lower of the two middles. Not the mean: one nine-step run
    would drag that up and tell you nothing about a normal week."""
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def seconds_of(run: Run) -> float:
    """How long a run took, from its own stamps. Zero when it has not finished."""
    if not run.started_at or not run.ended_at:
        return 0.0
    try:
        started = datetime.fromisoformat(run.started_at)
        ended = datetime.fromisoformat(run.ended_at)
    except ValueError:
        return 0.0
    return (ended - started).total_seconds()


@dataclass
class Health:
    """How a workflow has been doing, as a few numbers a person can read in one line.

    The first field is the one that matters. A monitor that stopped running produces no
    output, no error and no alert — it looks exactly like a quiet week — so something has to
    ask the question "when did this last work?" out loud, and that something is here.
    """

    workflow: str
    silent_for: float | None  # cadences since the last run that finished; None with no schedule
    runs: int
    states: dict[str, int]
    median_cost: Decimal
    worst_cost: Decimal
    median_seconds: float
    drafted: int
    waiting: int
    done: int
    rejected: int
    expired: int

    @property
    def quiet(self) -> bool:
        """True when more than one cadence has passed since a run finished."""
        return self.silent_for is not None and self.silent_for > 1.0

    @property
    def agreed(self) -> float | None:
        """The share of answered proposals that were agreed to. None when none were answered.

        Worth watching rather than celebrating. Near zero means the `find` step is wrong;
        near one, every week, usually means nobody is really reading them."""
        answered = self.done + self.rejected
        return self.done / answered if answered else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "silent_for": self.silent_for,
            "quiet": self.quiet,
            "runs": self.runs,
            "states": self.states,
            "median_cost_usd": str(self.median_cost),
            "worst_cost_usd": str(self.worst_cost),
            "median_seconds": round(self.median_seconds, 1),
            "proposals": {
                "drafted": self.drafted,
                "waiting": self.waiting,
                "done": self.done,
                "rejected": self.rejected,
                "expired": self.expired,
            },
            "agreed": self.agreed,
        }


def health(
    spec: WorkflowSpec,
    store: RunStore,
    inbox: Inbox,
    *,
    limit: int = 20,
    now: datetime | None = None,
) -> Health:
    """What the last few runs of one workflow say about how it is doing."""
    moment = now or datetime.now(UTC)
    runs = store.recent(limit=limit, workflow=spec.name)
    states: dict[str, int] = {}
    for run in runs:
        states[run.state] = states.get(run.state, 0) + 1
    costs = [run.spent for run in runs]
    silent = None
    last = store.last_done(spec.name)
    if spec.schedule is not None:
        cadence = (
            spec.schedule.next_after(moment) - spec.schedule.due_before(moment)
        ).total_seconds()
        since = (
            (moment - datetime.fromisoformat(last.started_at)).total_seconds()
            if last and last.started_at
            else None
        )
        silent = round(since / cadence, 2) if since is not None else float("inf")
    mine = [item for item in inbox.items() if item.proposal.action]
    by_state: dict[str, int] = {}
    for item in mine:
        by_state[item.state] = by_state.get(item.state, 0) + 1
    return Health(
        workflow=spec.name,
        silent_for=silent,
        runs=len(runs),
        states=states,
        median_cost=median(costs),
        worst_cost=max(costs) if costs else Decimal(0),
        median_seconds=float(median([Decimal(str(seconds_of(run))) for run in runs])),
        drafted=len(mine),
        waiting=by_state.get("proposed", 0),
        done=by_state.get("done", 0) + by_state.get("undone", 0),
        rejected=by_state.get("rejected", 0),
        expired=by_state.get("expired", 0),
    )


def reports_path(data_dir: Path, pack: str) -> Path:
    return data_dir / pack / REPORT_DIR


def marks_path(data_dir: Path, pack: str, workflow: str) -> Path:
    return data_dir / pack / STATE_DIR / f"{workflow}.json"


def waiting_summary(items: list[Item]) -> str:
    """One line about the inbox, for the end of a run."""
    if not items:
        return "nothing waiting for a decision"
    overdue = sum(1 for item in items if item.overdue())
    return f"{len(items)} waiting for a decision" + (f", {overdue} overdue" if overdue else "")
