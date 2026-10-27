"""Tools that change something, built from the pack's ontology.

A read-only agent is a reporting tool. The moment it may change something, three questions
appear that reading never asked: what exactly is about to change, who agreed to it, and how
do we undo it?

An action is declared in the pack's ontology, on an object the pack already describes, with
typed fields and one sentence a person can say yes or no to. Vardex turns it into a tool.
What that tool is allowed to do comes from `permissions.yaml`, so the same declaration is a
proposal in one pack and a write in another, with no change to the code.

The outside system sits behind a Connector, like every other outside system in Vardex: one
small class per system, and the engine speaks only its own vocabulary. A connector owes
Vardex two things, and the second is the one people forget:

- `carry_out`, which does it and returns a `Change` — including the values as they were,
  because an undo that does not know what it is putting back is a guess;
- `undo`, which puts it back. A system that cannot undo says so with `undoable = False`,
  and Vardex refuses the undo rather than pretending.

Note what is *not* on that list: drafting. A proposal is Vardex's own paperwork and goes in
Vardex's own inbox, so nothing reaches the other system until a person has said yes. The
connector is only ever asked to move.

Every proposal carries an idempotency key, chosen by whatever asked for it. A scheduled run
that is picked up twice proposes the same change twice, and the key is what makes the second
one the same proposal rather than a second one.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vardex.packs import PackError
from vardex.semantic import Action, Knowledge, ObjectType, quoted
from vardex.sql import QueryError, Warehouse
from vardex.tools import Invocation, Tool, ToolFailure, ToolResult, arguments_schema

CONNECTOR_DIR = "connectors"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def reference_for(action: str, key: str, idempotency_key: str) -> str:
    """The name a proposal is known by, to Vardex and to the person approving it.

    It is built from the idempotency key, so asking for the same change twice gives the
    same reference, and the second ask can be recognised instead of carried out again.
    """
    digest = hashlib.sha256(idempotency_key.encode()).hexdigest()[:8]
    return f"{action}-{key}-{digest}"


@dataclass(frozen=True)
class Proposal:
    """One action, filled in, aimed at one object, ready for a person to read."""

    action: str
    key: str  # the object's key, such as an organisation number
    label: str  # what people call it, such as MOWI ASA
    fields: dict[str, Any]
    sentence: str  # the pack's confirm line, filled in
    made_at: str
    # What this proposal is one of. Two calls with the same key are the same proposal, so a
    # retried run cannot fill the inbox with duplicates. Empty means "this call only".
    idempotency_key: str = ""
    why: str = ""  # the reason, in the words of whoever asked

    @property
    def reference(self) -> str:
        return reference_for(self.action, self.key, self.idempotency_key or self.made_at)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "key": self.key,
            "label": self.label,
            "fields": self.fields,
            "sentence": self.sentence,
            "made_at": self.made_at,
            "idempotency_key": self.idempotency_key,
            "why": self.why,
            "reference": self.reference,
        }


def proposal_from_dict(data: dict[str, Any]) -> Proposal:
    """A proposal read back from a file, ignoring whatever the log keeps beside it."""
    known = {"action", "key", "label", "fields", "sentence", "made_at", "idempotency_key", "why"}
    return Proposal(**{name: value for name, value in data.items() if name in known})


@dataclass(frozen=True)
class Change:
    """What a connector did, and what it would take to put it back.

    `reference` is the outside system's own name for the change, not Vardex's: a case
    number, a row id, a URL. Vardex keeps it, prints it and hands it back on an undo, and
    never looks inside it. `before` is what makes an undo more than a guess — a connector
    that cannot say what the values were leaves it empty, and the undo then restores
    nothing rather than restoring something invented.
    """

    reference: str  # what the outside system calls it, for a person looking it up there
    before: dict[str, Any]  # the fields as they were; empty when the system cannot say
    after: dict[str, Any]
    at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "before": self.before,
            "after": self.after,
            "at": self.at,
        }


def change_from_dict(data: dict[str, Any]) -> Change:
    return Change(
        reference=data["reference"],
        before=data.get("before", {}),
        after=data.get("after", {}),
        at=data.get("at", ""),
    )


class ConnectorError(Exception):
    """Raised when an outside system refuses, or cannot do what was asked of it."""


class Drafts(Protocol):
    """Where a proposal waits for a person, and where every change is written down.

    Declared here as the two things a tool needs, so that the module which turns an action
    into a tool does not have to know how the inbox keeps its records — the same bargain as
    Warehouse, Index and Connector.
    """

    def draft(self, proposal: Proposal, *, by: str, needed_by: str) -> tuple[str, bool]:
        """Put a proposal where people look. Returns its reference, and whether it is new:
        the same proposal asked for twice is one proposal and the second answer is False."""
        ...

    def record_done(self, proposal: Proposal, change: Change, *, by: str) -> None:
        """Write down that this change was made, for the audit trail."""
        ...


class Connector(Protocol):
    """What Vardex needs from an outside system that can carry out an action."""

    name: str  # what and where it is, for people; never a password
    undoable: bool  # whether a change made here can be put back

    def carry_out(self, proposal: Proposal) -> Change:
        """Do it, and say what changed and what it was before."""
        ...

    def undo(self, change: Change) -> str:
        """Put it back. Raises ConnectorError when that is not possible."""
        ...


def check_fields(action: Action, given: dict[str, Any]) -> dict[str, Any]:
    """The action's fields, checked and converted. Raises ToolFailure with what is wrong, so
    the model can correct itself; a field the action does not declare is refused outright."""
    unknown = sorted(set(given) - {f.name for f in action.fields} - {"key"})
    if unknown:
        raise ToolFailure(f"{action.name} has no field {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for field_spec in action.fields:
        if field_spec.name not in given or given[field_spec.name] is None:
            if field_spec.required:
                raise ToolFailure(
                    f"{action.name} needs {field_spec.name}: {field_spec.description}"
                )
            continue
        value = given[field_spec.name]
        if field_spec.type == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError) as err:
                raise ToolFailure(
                    f"{field_spec.name} must be a whole number, not {value!r}"
                ) from err
        elif field_spec.type == "amount":
            try:
                value = float(value)
            except (TypeError, ValueError) as err:
                raise ToolFailure(f"{field_spec.name} must be a number, not {value!r}") from err
        elif field_spec.type == "choice" and str(value) not in field_spec.choices:
            raise ToolFailure(
                f"{field_spec.name} must be one of {', '.join(field_spec.choices)}, not {value!r}"
            )
        else:
            value = str(value)
        values[field_spec.name] = value
    return values


def in_sentence(values: dict[str, Any]) -> dict[str, Any]:
    """The fields as they read inside the action's sentence. A model writes a reason as a
    sentence of its own, full stop included, and the sentence it goes into has one already:
    without this, a person is asked to approve "… because equity was negative in 2025..".
    """
    return {
        name: value.strip().rstrip(".").rstrip() if isinstance(value, str) else value
        for name, value in values.items()
    }


def find_object(warehouse: Warehouse, obj: ObjectType, key: str) -> str:
    """The label of the object with this key. Raises ToolFailure when there is no such row:
    an action aimed at nothing must fail before anything is written."""
    column = obj.label or obj.key
    # The key comes from the model, so it is quoted as a literal, never pasted in raw.
    sql = f"select {column} from {obj.table} where {obj.key} = {quoted(key)}"
    try:
        rows = warehouse.run(sql, max_rows=2).rows
    except QueryError as err:
        raise ToolFailure(f"could not look up {obj.name} {key}: {err}") from err
    if not rows:
        raise ToolFailure(f"there is no {obj.name} with {obj.key} {key}")
    return str(rows[0][0])


def action_tool(
    action: Action,
    knowledge: Knowledge,
    warehouse: Warehouse,
    connector: Connector,
    drafts: Drafts,
) -> Tool:
    """One of the pack's actions, as a tool. What it may do is decided per call, by the level
    the pack gave it: draft writes a proposal, approve and act carry it out."""
    obj = knowledge.ontology.object(action.object)
    fields = {
        "key": {
            "type": "string",
            "description": f"The {obj.key} of the {obj.name} this is about.",
        },
        **{
            f.name: (
                {"type": f.json_type, "description": f.description, "enum": f.choices}
                if f.choices
                else {"type": f.json_type, "description": f.description}
            )
            for f in action.fields
        },
    }
    required = ["key", *[f.name for f in action.fields if f.required]]

    def run(call: Invocation) -> ToolResult:
        key = str(call.argument("key", "")).strip()
        if not key:
            raise ToolFailure(f"{action.name} needs the {obj.key} of the {obj.name}")
        label = find_object(warehouse, obj, key)
        values = check_fields(action, call.arguments)
        work = call.context.get("work", "")
        proposal = Proposal(
            action=action.name,
            key=key,
            label=label,
            fields=values,
            sentence=action.confirm.format(label=label, key=key, **in_sentence(values)),
            made_at=now(),
            idempotency_key=f"{work}:{action.name}:{key}" if work else "",
            why=call.context.get("why", ""),
        )
        who = call.context.get("by") or call.context.get("run") or "whoever is at the keyboard"
        if call.dry_run:
            return ToolResult(
                f"Dry run, nothing written. It would have: {proposal.sentence}",
                structured={**proposal.as_dict(), "state": "dry_run"},
            )
        if not call.may_write:
            # draft: the proposal goes in the inbox for a person, and nothing else happens.
            # Nothing reaches the other system on this path, not even as a draft record.
            reference, is_new = drafts.draft(
                proposal, by=who, needed_by=call.context.get("needed_by", "")
            )
            again = "" if is_new else " It was already proposed, so this did not add a second."
            return ToolResult(
                f"Proposed, not done: {proposal.sentence}\nA person must approve it, as "
                f"{reference}.{again}",
                structured={**proposal.as_dict(), "reference": reference, "state": "proposed"},
            )
        change = connector.carry_out(proposal)
        drafts.record_done(proposal, change, by=who)
        return ToolResult(
            f"Done: {proposal.sentence}\nReference {change.reference} in {connector.name}.",
            structured={**proposal.as_dict(), "change": change.as_dict(), "state": "done"},
            wrote=True,
        )

    return Tool(
        name=action.name,
        summary=f"{action.description} Changes a {obj.name}; what it may do depends on this "
        "pack's permissions.",
        description=(
            f"{action.description}\n\n"
            f"It acts on one {obj.name}, named by its {obj.key}. Look the object up first and "
            "be sure you have the right one.\n\n"
            "Depending on what this pack allows, the call either writes a proposal for a "
            "person to approve, or carries the change out. Either way you are told which "
            "happened, in those words — report that to the person rather than claiming the "
            "change is made.\n\n"
            "Asking for the same change twice in one piece of work gives you the same "
            "proposal back, so a retry cannot double it.\n\n"
            f'A person will read this sentence and decide: "{action.confirm}".'
        ),
        schema=arguments_schema(fields, required),
        run=run,
        writes=True,
    )


def action_tools(
    knowledge: Knowledge,
    warehouse: Warehouse,
    connectors: dict[str, Connector],
    drafts: Drafts,
) -> list[Tool]:
    """Every action in the pack's ontology whose connector is available."""
    tools = []
    for action in knowledge.ontology.actions:
        connector = connectors.get(action.connector)
        if connector is None:
            raise PackError(
                f"action {action.name} needs the connector {action.connector}, which is not "
                "configured. Set it up, or take the action out of the ontology."
            )
        tools.append(action_tool(action, knowledge, warehouse, connector, drafts))
    return tools


@dataclass
class FileConnector:
    """A stand-in for a real system: it keeps the changes made to it in files.

    It exists so a pack can declare its actions, and a team can try the whole path from a
    question to a proposal to an approved change and back out again, before anyone has
    wired up a CRM. Replacing it with a real connector means writing `carry_out` and `undo`
    against that system's API, and changing nothing else.

    It keeps two files: a log of everything that has happened, which is never rewritten,
    and the current value of each field it has set, which is what makes a real `before`
    possible. A CRM would read that from the CRM.
    """

    folder: Path
    name: str = "a file standing in for the real system"
    undoable: bool = True
    state: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.state = self._read_state()

    @property
    def log_path(self) -> Path:
        return self.folder / "changes.jsonl"

    @property
    def state_path(self) -> Path:
        return self.folder / "state.json"

    def _read_state(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_state(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".writing")
        temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.state_path)

    def _append(self, entry: dict[str, Any]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def carry_out(self, proposal: Proposal) -> Change:
        # A real connector would return the id the system gave the change. A file has no
        # ids, so the thing being changed is the handle: one action on one object.
        handle = f"{proposal.action}:{proposal.key}"
        change = Change(
            reference=handle,
            before=dict(self.state.get(handle, {})),
            after=dict(proposal.fields),
            at=now(),
        )
        self.state[handle] = dict(proposal.fields)
        self._write_state()
        self._append(
            {"at": change.at, "what": "done", **proposal.as_dict(), "change": change.as_dict()}
        )
        return change

    def undo(self, change: Change) -> str:
        """Put the fields back as they were.

        It refuses if the values are no longer the ones this change left behind. Somebody
        else has been here since, and an undo that overwrites their work is worse than no
        undo: it is a second unapproved change, wearing the first one's reference.
        """
        current = self.state.get(change.reference)
        if current is None or current != change.after:
            raise ConnectorError(
                f"{change.reference} is not as this change left it in {self.name}, so it "
                "cannot be undone: somebody has changed it since. Look at the current "
                "value and decide what it should be."
            )
        if change.before:
            self.state[change.reference] = dict(change.before)
        else:
            self.state.pop(change.reference, None)
        self._write_state()
        self._append(
            {
                "at": now(),
                "what": "undone",
                "reference": change.reference,
                "restored": change.before,
            }
        )
        return change.reference


def connector_path(data_dir: Path, pack: str, connector: str) -> Path:
    """Where a stand-in connector keeps what it has been told to do."""
    return data_dir / pack / CONNECTOR_DIR / connector
