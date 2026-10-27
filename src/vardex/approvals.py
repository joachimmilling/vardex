"""The approval inbox: where a proposal waits for a person, and what happens after.

A tool that may act "once a person has said yes" has to find a person to ask. Asking at the
keyboard works for an agent somebody is watching. It is useless for work that runs at half
past six on a Monday morning: there is nobody there, so the tool refuses, and the run is
reduced to telling you what it would have done.

So the question and the answer are separated in time. An unattended run may only *draft*:
it writes the proposal, with the sentence a person has to agree to and the reason it was
made, and stops. Later, a person reads the inbox and decides. The change is then carried
out by the approval, not by the agent — a different process, started by a human being,
which is the whole point.

The inbox is an append-only log, one line per thing that happened, and the state of a
proposal is what you get by replaying its lines. That is more code than keeping a `state`
column, and worth it three times over:

- Appending one line is the safest write there is. A crash leaves a whole log with one
  fewer line in it, never a half-changed record.
- "Who approved this, when, and what did it say at the time" is the question auditors ask,
  and the log is the answer. A status column throws that away on every update.
- Undo is just another entry. Nothing is ever rewritten, so the history of a change
  survives being reversed.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from vardex.actions import (
    Change,
    Connector,
    ConnectorError,
    Proposal,
    change_from_dict,
    proposal_from_dict,
)

INBOX_FILE = "inbox.jsonl"

# What has happened to a proposal. Only `done` means something changed outside Vardex.
State = Literal["proposed", "approved", "rejected", "done", "failed", "undone", "expired"]
OPEN: tuple[State, ...] = ("proposed",)  # waiting for a person
FINAL: tuple[State, ...] = ("rejected", "undone", "expired")


class ApprovalError(Exception):
    """Raised when a decision cannot be made: no such proposal, or already decided."""


def now() -> datetime:
    return datetime.now(UTC)


def stamp(when: datetime | None = None) -> str:
    return (when or now()).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Entry:
    """One line of the log: something that happened to one proposal."""

    at: str
    reference: str
    state: State
    by: str = ""  # who did it: a person's name, or the run that proposed it
    note: str = ""
    proposal: dict[str, Any] | None = None  # only on the first entry
    change: dict[str, Any] | None = None  # only when something was carried out
    needed_by: str = ""

    def as_dict(self) -> dict[str, Any]:
        body = {
            "at": self.at,
            "reference": self.reference,
            "state": self.state,
            "by": self.by,
            "note": self.note,
            "needed_by": self.needed_by,
        }
        if self.proposal is not None:
            body["proposal"] = self.proposal
        if self.change is not None:
            body["change"] = self.change
        return body


@dataclass
class Item:
    """One proposal, and everything that has happened to it."""

    reference: str
    proposal: Proposal
    state: State
    history: list[Entry] = field(default_factory=list)
    needed_by: str = ""  # when somebody should have answered by
    change: Change | None = None
    decided_by: str = ""
    note: str = ""

    @property
    def open(self) -> bool:
        return self.state in OPEN

    @property
    def made_at(self) -> str:
        return self.proposal.made_at

    def overdue(self, at: datetime | None = None) -> bool:
        """True when this is still waiting and the deadline has passed."""
        if not self.open or not self.needed_by:
            return False
        try:
            return datetime.fromisoformat(self.needed_by) < (at or now())
        except ValueError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "state": self.state,
            "needed_by": self.needed_by,
            "decided_by": self.decided_by,
            "note": self.note,
            "proposal": self.proposal.as_dict(),
            "change": self.change.as_dict() if self.change else None,
            "history": [entry.as_dict() for entry in self.history],
        }


@dataclass
class Inbox:
    """A pack's proposals, and the decisions made about them."""

    path: Path

    def append(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry.as_dict(), ensure_ascii=False, default=str) + "\n")

    def entries(self) -> list[Entry]:
        """Every line of the log, in the order it was written. A line that cannot be read
        is skipped rather than fatal: one bad line must not hide the whole inbox."""
        if not self.path.is_file():
            return []
        found = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                found.append(
                    Entry(
                        at=data["at"],
                        reference=data["reference"],
                        state=data["state"],
                        by=data.get("by", ""),
                        note=data.get("note", ""),
                        needed_by=data.get("needed_by", ""),
                        proposal=data.get("proposal"),
                        change=data.get("change"),
                    )
                )
            except (ValueError, KeyError):
                continue
        return found

    def items(self, state: State | None = None) -> list[Item]:
        """Every proposal, newest first, built by replaying the log."""
        by_reference: dict[str, Item] = {}
        for entry in self.entries():
            item = by_reference.get(entry.reference)
            if item is None:
                if entry.proposal is None:
                    continue  # a decision about a proposal this log never saw
                item = Item(
                    reference=entry.reference,
                    proposal=proposal_from_dict(entry.proposal),
                    state=entry.state,
                    needed_by=entry.needed_by,
                )
                by_reference[entry.reference] = item
            item.state = entry.state
            item.history.append(entry)
            if entry.needed_by:
                item.needed_by = entry.needed_by
            if entry.change is not None:
                item.change = change_from_dict(entry.change)
            if entry.state in ("approved", "rejected", "undone"):
                item.decided_by = entry.by
            if entry.note:
                item.note = entry.note
        found = sorted(by_reference.values(), key=lambda i: i.made_at, reverse=True)
        return [item for item in found if state is None or item.state == state]

    def get(self, reference: str) -> Item:
        """One proposal, by its reference or by enough of the start of it to be unique.

        Typing a full reference is a chore and mistyping one is worse, so a prefix will do —
        but only if it matches one proposal. Several matches is an error with the list in
        it, never a guess at which one was meant.
        """
        items = self.items()
        exact = [item for item in items if item.reference == reference]
        if exact:
            return exact[0]
        matches = [item for item in items if item.reference.startswith(reference)]
        if not matches:
            raise ApprovalError(f"There is no proposal {reference} in {self.path}.")
        if len(matches) > 1:
            names = ", ".join(item.reference for item in matches[:5])
            raise ApprovalError(f"{reference} matches several proposals: {names}.")
        return matches[0]

    def add(self, proposal: Proposal, *, by: str = "", needed_by: str = "") -> Item:
        """Put a proposal in the inbox, unless its reference is already there.

        The second half of that sentence is the idempotency. A scheduled run that is picked
        up again proposes the same thing again, with the same key, and this returns the
        proposal that is already waiting instead of adding a twin.
        """
        existing = {item.reference: item for item in self.items()}
        if proposal.reference in existing:
            return existing[proposal.reference]
        self.append(
            Entry(
                at=stamp(),
                reference=proposal.reference,
                state="proposed",
                by=by,
                needed_by=needed_by,
                proposal=proposal.as_dict(),
            )
        )
        return self.get(proposal.reference)

    def draft(self, proposal: Proposal, *, by: str = "", needed_by: str = "") -> tuple[str, bool]:
        """What a drafting tool calls: put this where people look, once.

        Returns the reference and whether it is new. False means an identical proposal is
        already waiting, which is the answer a retried run should get.
        """
        before = {item.reference for item in self.items()}
        self.add(proposal, by=by, needed_by=needed_by)
        return proposal.reference, proposal.reference not in before

    def record_done(self, proposal: Proposal, change: Change, *, by: str = "") -> None:
        """Write down a change that was made without waiting: approved at the keyboard, or
        by a pack whose level is `act`. The inbox is the audit trail either way, so a change
        that never waited for anybody is still in the same log as one that did."""
        if proposal.reference not in {item.reference for item in self.items()}:
            self.append(
                Entry(
                    at=stamp(),
                    reference=proposal.reference,
                    state="proposed",
                    by=by,
                    proposal=proposal.as_dict(),
                    note="approved as it was asked for, not through the inbox",
                )
            )
            self.append(Entry(at=stamp(), reference=proposal.reference, state="approved", by=by))
        self.append(
            Entry(
                at=stamp(),
                reference=proposal.reference,
                state="done",
                by=by,
                change=change.as_dict(),
            )
        )

    def decide(self, reference: str, *, approved: bool, by: str, note: str = "") -> Item:
        """Record a person's yes or no. This changes nothing outside Vardex by itself."""
        item = self.get(reference)
        if not item.open:
            raise ApprovalError(
                f"{item.reference} is {item.state}, not waiting for a decision. "
                "Nothing was changed."
            )
        if not by.strip():
            raise ApprovalError("An approval needs a name: who is saying yes to this?")
        self.append(
            Entry(
                at=stamp(),
                reference=item.reference,
                state="approved" if approved else "rejected",
                by=by,
                note=note,
            )
        )
        return self.get(item.reference)

    def carry_out(self, reference: str, connector: Connector, *, by: str = "") -> Item:
        """Do what an approved proposal says, through the connector, and record the result.

        A failure is recorded as `failed` and the proposal can be tried again, because the
        usual reason is that the other system was down for a minute. The failure is kept in
        the log either way: a change that failed is a thing that happened.
        """
        item = self.get(reference)
        if item.state not in ("approved", "failed"):
            raise ApprovalError(
                f"{item.reference} is {item.state}, not approved. Nothing was done."
            )
        try:
            change = connector.carry_out(item.proposal)
        except (ConnectorError, OSError) as err:
            self.append(
                Entry(
                    at=stamp(),
                    reference=item.reference,
                    state="failed",
                    by=by,
                    note=str(err),
                )
            )
            raise ApprovalError(f"{item.reference} could not be carried out: {err}") from err
        self.append(
            Entry(
                at=stamp(),
                reference=item.reference,
                state="done",
                by=by,
                note=f"carried out in {connector.name}",
                change=change.as_dict(),
            )
        )
        return self.get(item.reference)

    def undo(self, reference: str, connector: Connector, *, by: str) -> Item:
        """Put a carried-out change back, and record that too."""
        item = self.get(reference)
        if item.state != "done" or item.change is None:
            raise ApprovalError(f"{item.reference} is {item.state}: there is nothing to undo.")
        if not getattr(connector, "undoable", False):
            raise ApprovalError(
                f"{connector.name} cannot undo a change. Decide what should happen instead "
                "and make that change on purpose — do not leave it half done."
            )
        try:
            connector.undo(item.change)
        except (ConnectorError, OSError) as err:
            raise ApprovalError(f"{item.reference} could not be undone: {err}") from err
        self.append(
            Entry(
                at=stamp(),
                reference=item.reference,
                state="undone",
                by=by,
                note=f"undone in {connector.name}",
                change=item.change.as_dict(),
            )
        )
        return self.get(item.reference)

    def expire(
        self, *, after_days: int, at: datetime | None = None, by: str = "vardex"
    ) -> list[Item]:
        """Close proposals nobody answered in time.

        A proposal left open for a fortnight is not a decision waiting to be made, it is a
        decision nobody is going to make, and the numbers it rests on are stale. Expiring it
        is honest; leaving it to be approved later is not.
        """
        cutoff = (at or now()) - timedelta(days=after_days)
        closed = []
        for item in self.items(state="proposed"):
            try:
                made = datetime.fromisoformat(item.made_at)
            except ValueError:
                continue
            if made < cutoff:
                self.append(
                    Entry(
                        at=stamp(at),
                        reference=item.reference,
                        state="expired",
                        by=by,
                        note=f"nobody answered within {after_days} days",
                    )
                )
                closed.append(self.get(item.reference))
        return closed

    def overdue(self, at: datetime | None = None) -> list[Item]:
        """Open proposals whose deadline has passed: what to escalate."""
        return [item for item in self.items(state="proposed") if item.overdue(at)]


def inbox_path(data_dir: Path, pack: str) -> Path:
    """Where a pack's inbox lives."""
    return data_dir / pack / INBOX_FILE
