"""Tools that change something, built from the pack's ontology.

A read-only agent is a reporting tool. The moment it may change something, three questions
appear that reading never asked: what exactly is about to change, who agreed to it, and how
do we undo it? Vardex answers the first two here. The third is not solved yet.

An action is declared in the pack's ontology, on an object the pack already describes, with
typed fields and one sentence a person can say yes or no to. Vardex turns it into a tool. What
that tool is allowed to do comes from `permissions.yaml`, so the same declaration is a
proposal in one pack and a write in another, with no change to the code.

The outside system is behind a Connector, like every other outside system in Vardex: one
small class per system, and the engine speaks only its own vocabulary. The connector here
writes to a file, so a pack can be built and tested without a CRM behind it.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vardex.packs import PackError
from vardex.semantic import Action, Knowledge, ObjectType, quoted
from vardex.sql import QueryError, Warehouse
from vardex.tools import Invocation, Tool, ToolFailure, ToolResult, arguments_schema

PROPOSAL_DIR = "proposals"


@dataclass(frozen=True)
class Proposal:
    """One action, filled in, aimed at one object, ready for a person to read."""

    action: str
    key: str  # the object's key, such as an organisation number
    label: str  # what people call it, such as MOWI ASA
    fields: dict[str, Any]
    sentence: str  # the pack's confirm line, filled in
    made_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "key": self.key,
            "label": self.label,
            "fields": self.fields,
            "sentence": self.sentence,
            "made_at": self.made_at,
        }


class Connector(Protocol):
    """What Vardex needs from an outside system that can carry out an action."""

    name: str  # what and where it is, for people; never a password

    def record(self, proposal: Proposal) -> str:
        """Write the proposal down for a person to decide on. Returns a reference."""
        ...

    def carry_out(self, proposal: Proposal) -> str:
        """Do it. Returns a reference a person can look the change up by."""
        ...


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def check_fields(action: Action, given: dict[str, Any]) -> dict[str, Any]:
    """The action's fields, checked and converted. Raises ToolFailure with what is wrong, so
    the model can correct itself; a field the action does not declare is refused outright."""
    unknown = sorted(set(given) - {f.name for f in action.fields} - {"key"})
    if unknown:
        raise ToolFailure(f"{action.name} has no field {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for field in action.fields:
        if field.name not in given or given[field.name] is None:
            if field.required:
                raise ToolFailure(f"{action.name} needs {field.name}: {field.description}")
            continue
        value = given[field.name]
        if field.type == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError) as err:
                raise ToolFailure(f"{field.name} must be a whole number, not {value!r}") from err
        elif field.type == "amount":
            try:
                value = float(value)
            except (TypeError, ValueError) as err:
                raise ToolFailure(f"{field.name} must be a number, not {value!r}") from err
        elif field.type == "choice" and str(value) not in field.choices:
            raise ToolFailure(
                f"{field.name} must be one of {', '.join(field.choices)}, not {value!r}"
            )
        else:
            value = str(value)
        values[field.name] = value
    return values


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
    action: Action, knowledge: Knowledge, warehouse: Warehouse, connector: Connector
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
        proposal = Proposal(
            action=action.name,
            key=key,
            label=label,
            fields=values,
            sentence=action.confirm.format(label=label, key=key, **values),
            made_at=now(),
        )
        if call.dry_run:
            return ToolResult(
                f"Dry run, nothing written. It would have: {proposal.sentence}",
                structured=proposal.as_dict(),
            )
        if not call.may_write:
            # draft: the proposal is written down for a person, and nothing else happens.
            reference = connector.record(proposal)
            return ToolResult(
                f"Proposed, not done: {proposal.sentence}\nA person must approve it. "
                f"Reference {reference} in {connector.name}.",
                structured={**proposal.as_dict(), "reference": reference, "state": "proposed"},
            )
        reference = connector.carry_out(proposal)
        return ToolResult(
            f"Done: {proposal.sentence}\nReference {reference} in {connector.name}.",
            structured={**proposal.as_dict(), "reference": reference, "state": "done"},
            wrote=True,
        )

    return Tool(
        name=action.name,
        summary=f"{action.description} Changes a {obj.name}; what it may do depends on this "
        "pack's permissions.",
        description=(
            f"{action.description}\n\n"
            f"It acts on one {obj.name}, named by its {obj.key}. Look the object up first and "
            "be sure you have the right one: there is no undo in this version.\n\n"
            "Depending on what this pack allows, the call either writes a proposal for a "
            "person to approve, or carries the change out. Either way you are told which "
            "happened, in those words — report that to the person rather than claiming the "
            "change is made.\n\n"
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
        tools.append(action_tool(action, knowledge, warehouse, connector))
    return tools


@dataclass
class FileConnector:
    """A stand-in for a real system: it keeps proposals and changes in a JSON-lines file.

    It exists so a pack can declare its actions, and a team can try the whole path from a
    question to a proposal, before anyone has wired up a CRM. Replacing it with a real
    connector means writing `record` and `carry_out` against that system's API, and changing
    nothing else.
    """

    folder: Path
    name: str = "the proposals file"

    def path(self, action: str) -> Path:
        return self.folder / f"{action}.jsonl"

    def append(self, proposal: Proposal, state: str) -> str:
        path = self.path(proposal.action)
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = f"{proposal.action}-{proposal.key}-{proposal.made_at}"
        line = json.dumps(
            {**proposal.as_dict(), "state": state, "reference": reference}, ensure_ascii=False
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
        return reference

    def record(self, proposal: Proposal) -> str:
        return self.append(proposal, "proposed")

    def carry_out(self, proposal: Proposal) -> str:
        return self.append(proposal, "done")


def proposals_path(data_dir: Path, pack: str) -> Path:
    """Where a pack's proposals are kept."""
    return data_dir / pack / PROPOSAL_DIR
