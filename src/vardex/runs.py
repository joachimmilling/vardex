"""What a run of a workflow knows about itself, written down after every step.

A run that lives longer than a request has to survive the process that started it. The
machine reboots, the model is overloaded, the budget runs out, a person has to approve
something before the work can go on: in every one of those cases the run stops in the
middle, and what matters is whether it can be picked up where it stopped.

So a run is a file. After every step, and after every item inside a step, the whole run is
written to disk, whole, through a temporary file. Resuming reads that file and skips what
is already recorded. This is the simplest form of what the durable-execution engines do:
keep a journal of finished work, and on a second attempt return the journal's answer
instead of doing the work again.

Two things make the journal more than a progress bar:

- A writing tool call is recorded under a key made from the tool and its arguments, so a
  resumed run that asks for the same change again is given the first answer back and
  nothing moves twice. That is an idempotency key, kept by the caller rather than hoped
  for from the system being written to.
- Nothing is ever deleted. A run keeps its failed steps and its stopped state, because the
  run that went wrong is the one somebody has to read.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

RUNS_DIR = "runs"
ID_STAMP = "%Y%m%dT%H%M%SZ"
LOCK_STALE_SECONDS = 6 * 60 * 60  # a lock older than this was left by a process that died

# Why a run is in the state it is in. Only `done` means every step finished.
RunState = Literal["running", "done", "failed", "stopped"]
# `stopped` is not a failure: the budget ran out, or a person has to answer first.
# A step is `skipped` when there was nothing for it to do, and `stopped` when it was cut
# short — the difference matters on a resume, which retries the second and not the first.
StepState = Literal["pending", "running", "done", "failed", "skipped", "stopped"]


def now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(workflow: str, when: datetime | None = None) -> str:
    """A run's name, which is also where its file is. Sortable, and readable out loud."""
    stamp = (when or datetime.now(UTC)).strftime(ID_STAMP)
    return f"{workflow}-{stamp}"


def call_key(name: str, arguments: dict[str, Any]) -> str:
    """The key a tool call is journalled under: the same call asked twice is one key.

    The arguments are serialised with sorted keys so that two dicts that mean the same
    thing hash the same. Eight hex characters is plenty for the few hundred calls one run
    can make, and short enough to read in a transcript.
    """
    material = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False, default=str)
    return f"{name}:{hashlib.sha256(material.encode()).hexdigest()[:8]}"


@dataclass
class Step:
    """One step of a workflow, as the run records it."""

    id: str
    uses: str
    state: StepState = "pending"
    started_at: str = ""
    ended_at: str = ""
    note: str = ""  # why it failed, or what it found
    items: int = 0  # units of work this step dealt with
    cost_usd: str = "0"  # a string, because JSON has no decimals and money is not a float

    @property
    def finished(self) -> bool:
        """Whether a resume should leave this step alone."""
        return self.state in ("done", "skipped")


@dataclass
class Run:
    """One run of one workflow: what it has done, what it has spent, and where it stopped."""

    id: str
    pack: str
    workflow: str
    trigger: str  # schedule, manual or event — who asked for this run
    started_at: str
    state: RunState = "running"
    ended_at: str = ""
    note: str = ""
    dry_run: bool = False
    attempts: int = 1  # how many times this run has been picked up
    steps: list[Step] = field(default_factory=list)
    journal: dict[str, Any] = field(default_factory=dict)  # finished work, by key
    facts: dict[str, Any] = field(default_factory=dict)  # what one step leaves for the next
    cost_usd: str = "0"

    def step(self, id: str) -> Step | None:
        return next((s for s in self.steps if s.id == id), None)

    def spend(self, amount: Decimal | None) -> None:
        """Add to what the run has cost. An unpriced call adds nothing and is not hidden:
        the step's own note says so."""
        if amount is not None:
            self.cost_usd = str(Decimal(self.cost_usd) + amount)

    @property
    def spent(self) -> Decimal:
        return Decimal(self.cost_usd)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_from_dict(data: dict[str, Any]) -> Run:
    """A run read back from its file. Unknown fields are refused rather than ignored: a
    run written by a newer Vardex must not be half-understood by an older one."""
    steps = [Step(**step) for step in data.pop("steps", [])]
    return Run(steps=steps, **data)


@dataclass
class RunStore:
    """Where a pack's runs are kept, one JSON file each."""

    folder: Path

    def path(self, run_id: str) -> Path:
        return self.folder / f"{run_id}.json"

    def save(self, run: Run) -> None:
        """Write the whole run, through a temporary file, so a crash cannot leave half of
        one. This is called after every step and every item, so it has to be cheap: a run
        file is a few kilobytes, which `os.replace` moves into place atomically."""
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path(run.id).with_suffix(".writing")
        temporary.write_text(
            json.dumps(run.as_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        os.replace(temporary, self.path(run.id))

    def load(self, run_id: str) -> Run:
        path = self.path(run_id)
        if not path.is_file():
            raise RunError(f"There is no run {run_id} in {self.folder}")
        try:
            return run_from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError) as err:
            raise RunError(f"{path} is not a run this version can read: {err}") from err

    def ids(self) -> list[str]:
        """Every run, newest first. The id starts with the workflow and ends with a UTC
        stamp, so sorting the names sorts the runs."""
        if not self.folder.is_dir():
            return []
        return sorted((p.stem for p in self.folder.glob("*.json")), reverse=True)

    def recent(self, limit: int = 20, workflow: str | None = None) -> list[Run]:
        runs = []
        for run_id in self.ids():
            if workflow and not run_id.startswith(f"{workflow}-"):
                continue
            try:
                runs.append(self.load(run_id))
            except RunError:
                continue  # a half-written or future-version file must not hide the rest
            if len(runs) >= limit:
                break
        return runs

    def last(self, workflow: str) -> Run | None:
        """The newest run of one workflow, whatever happened to it."""
        runs = self.recent(limit=1, workflow=workflow)
        return runs[0] if runs else None

    def last_done(self, workflow: str) -> Run | None:
        """The newest run of one workflow that finished every step. This is what a schedule
        is measured from: a run that failed does not count as having happened."""
        for run in self.recent(limit=50, workflow=workflow):
            if run.state == "done" and not run.dry_run:
                return run
        return None


def runs_path(data_dir: Path, pack: str) -> Path:
    """Where a pack's runs live."""
    return data_dir / pack / RUNS_DIR


class RunError(Exception):
    """Raised when a run cannot be read, or two runs of one workflow would overlap."""


@dataclass
class Lock:
    """One workflow, one run at a time.

    Schedulers fire twice. A person runs a workflow by hand while the timer fires. Both
    runs then read the same feed from the same cursor and do the same work twice, and
    because each writes the cursor at the end, one of them loses its place as well.

    The lock is a file created with O_EXCL, which is the one filesystem operation that can
    say "I made this, nobody else did". It holds the process id and the time, so a lock
    left behind by a machine that was switched off can be taken over once it is old enough.
    """

    path: Path
    stale_seconds: float = LOCK_STALE_SECONDS
    taken: bool = False

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create()
        except FileExistsError:
            held = self._read()
            age = time.time() - float(held.get("at", 0))
            if age < self.stale_seconds:
                raise RunError(
                    f"{self.path.stem} is already running (process {held.get('pid', '?')}, "
                    f"started {int(age)} s ago). Wait for it, or delete {self.path} if that "
                    "process is gone."
                ) from None
            # Old enough that whoever held it is not coming back.
            self.path.unlink(missing_ok=True)
            self._create()
        self.taken = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self.taken:
            self.path.unlink(missing_ok=True)
            self.taken = False

    def _create(self) -> None:
        handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump({"pid": os.getpid(), "at": time.time(), "since": now_text()}, file)

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
