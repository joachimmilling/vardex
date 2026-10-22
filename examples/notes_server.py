"""An MCP server over a folder of notes, so Vardex's agent can read what the team wrote.

Run it by hand to try it:

    VARDEX_NOTES_DIR=notes uv run python examples/notes_server.py

It is listed in packs/norwegian-companies/mcp.yaml, so `vardex agent` starts it itself. It
stands in for the servers a company already runs — Slack, Drive, GitHub, a wiki. Those need
credentials and a network; this one needs a folder, so the path can be tried end to end
before anybody asks for an access token.
"""

import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

NOTES = Path(os.getenv("VARDEX_NOTES_DIR") or "notes")
MAX_HITS = 5
AROUND = 300  # characters of a note shown around a match

mcp = MCPServer(
    "team-notes",
    version="1.0.0",
    instructions="Search the team's own notes. They are opinions and working notes, not facts.",
)


@mcp.tool(
    description=(
        "Search the team's own notes and working documents for a word or phrase.\n\n"
        "Use it for what colleagues have written about a company: a view, a worry, a decision "
        "taken in a meeting. It does not hold figures, and nothing in it is an official "
        "source — quote it as a colleague's note, never as a fact from a report."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def search_notes(query: str) -> str:
    """Find notes containing the query."""
    if not NOTES.is_dir():
        return f"There is no notes folder at {NOTES}."
    wanted = query.casefold()
    found = []
    for path in sorted(NOTES.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        at = text.casefold().find(wanted)
        if at == -1:
            continue
        start = max(0, at - AROUND // 2)
        found.append(f"{path.relative_to(NOTES)}\n…{text[start : start + AROUND].strip()}…")
        if len(found) == MAX_HITS:
            break
    if not found:
        return f"No note mentions {query!r}."
    return f"{len(found)} notes mention {query!r}:\n\n" + "\n\n".join(found)


if __name__ == "__main__":
    mcp.run("stdio")
