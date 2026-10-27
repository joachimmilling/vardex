"""What a pack lets its agent do, and how much it may spend doing it.

Every tool an agent can call has a permission level, and the pack sets it in
`permissions.yaml`. The engine never decides: a tool the pack has not listed refuses to run,
whatever the model asks and whoever is asking. The four levels are a ladder, and each rung
allows strictly more than the one below:

- `read`     — looks something up and changes nothing.
- `draft`    — writes a proposal for a person, and nothing else. Nothing outside Vardex moves.
- `approve`  — does the thing, but only after a person has said yes to this exact call.
- `act`      — does the thing.

The same file holds the spending caps and the approval rules, because all three answer the
same question: what may this pack do on its own? A cap is a number of dollars, enforced in
the model layer, so no prompt, no tool and no agent can spend past it.

`approve` has two faces, and the pack does not choose between them. When somebody is at the
keyboard, they are asked there and then. When nothing is watching — a scheduled run at half
past six — there is nobody to ask, so the tool may only draft, and the proposal waits in the
inbox for a person. Same level, same declaration, two ways of answering.
"""

from decimal import Decimal
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from vardex.packs import PackError, read_yaml

Permission = Literal["read", "draft", "approve", "act"]
LADDER: tuple[Permission, ...] = get_args(Permission)  # weakest first
WRITING: tuple[Permission, ...] = ("approve", "act")  # these may change something outside


class Budgets(BaseModel):
    """What a pack may spend on models. A cap left out is no cap at all."""

    model_config = ConfigDict(extra="forbid")

    per_run: Decimal | None = Field(default=None, gt=0)  # one question, however many steps
    per_day: Decimal | None = Field(default=None, gt=0)  # the pack's whole day, all channels


class Approvals(BaseModel):
    """Who may decide on this pack's proposals, and how long they have to do it."""

    model_config = ConfigDict(extra="forbid")

    # Who may say yes. Empty means anyone who can run the command, which is honest on one
    # person's machine and not enough for a company, where this belongs behind real identity.
    approvers: list[str] = []
    needed_within_hours: int = Field(default=72, ge=1)  # the deadline put on a new proposal
    expire_after_days: int = Field(default=14, ge=1)  # after this, nobody is going to answer
    escalate_to: str = ""  # who hears about a proposal nobody has answered in time


class Permissions(BaseModel):
    """The contents of a pack's permissions.yaml."""

    model_config = ConfigDict(extra="forbid")

    budgets: Budgets = Budgets()
    approvals: Approvals = Approvals()
    tools: dict[str, Permission] = {}

    def level(self, tool: str) -> Permission | None:
        """The level the pack gave this tool, or None if it gave it none."""
        return self.tools.get(tool)

    def allows(self, tool: str, needed: Permission) -> bool:
        """True when the pack lets this tool do something that needs `needed`."""
        given = self.level(tool)
        return given is not None and LADDER.index(given) >= LADDER.index(needed)

    def may_approve(self, who: str) -> bool:
        """Whether this person is on the pack's list of approvers."""
        return not self.approvals.approvers or who in self.approvals.approvers


def load_permissions(pack: Path) -> Permissions:
    """Read permissions.yaml in a pack folder. A pack without one may only read nothing: the
    file is how a pack says what its agent is for, so its absence is an error, not a default."""
    path = pack / "permissions.yaml"
    if not path.is_file():
        raise PackError(
            f"No permissions.yaml in {pack}. An agent needs one: it says which tools it may "
            "call and what it may spend."
        )
    return read_yaml(path, Permissions, "permissions file")


def unlisted(permissions: Permissions, tools: list[str]) -> list[str]:
    """Every tool that exists but has no level, and every level for a tool that does not."""
    problems = [
        f"tool {name} has no permission level" for name in tools if name not in permissions.tools
    ]
    problems += [
        f"permissions.yaml gives a level to {name}, which is not a tool here"
        for name in permissions.tools
        if name not in tools
    ]
    return problems
