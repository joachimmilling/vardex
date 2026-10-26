"""The tools an agent may call, and the one gate they all pass through.

A tool is a contract between code that is certain and a caller that is not. The caller is a
model, so the contract is written in prose: the name, the arguments and a description that
says when to use it and when not to. Nothing else reaches the model — never the code, never
the database, never the key.

Every call goes through `Toolbox.call`, which is the only place that decides whether a tool
may run. It checks the pack's permission level before the arguments, so a tool the pack has
not listed cannot run even if its arguments are perfect. The CLI, the MCP server and the
agent loop all call through here, which is what makes "no channel holds logic of its own"
true rather than aspirational.

Descriptions are disclosed progressively. The model starts with a one-line summary of each
tool and calls `describe_tool` for the whole manual, the same way it calls `list_tables` and
then `describe_table`. A catalogue of a thousand tables, or fifty tools from four MCP
servers, then costs a few hundred tokens instead of filling the window.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from vardex.embeddings import Embedder, Reranker
from vardex.llm import ToolCall, ToolSpec
from vardex.permissions import WRITING, Permission, Permissions
from vardex.retrieval import Filters, Index, Passage, SearchError, search
from vardex.semantic import Knowledge
from vardex.sql import QueryError, Warehouse, check_sql, format_table, transpile

SEARCH_LIMIT = 8  # passages one search gives back
PREVIEW_WORDS = 60  # of a passage, before the agent asks to read the whole of it


@dataclass(frozen=True)
class ToolResult:
    """What one tool call gave back, on its way to the model.

    `is_error` is for a failure the model can do something about: a typo in SQL, a filter that
    matched nothing, a refused permission. It is not an exception. The model reads it, learns
    and tries something else, which is how an agent recovers without a person.
    """

    content: str
    is_error: bool = False
    seconds: float = 0.0  # how long the tool itself took, so a slow one can be found
    structured: dict[str, Any] | None = None  # the same answer as data, for a channel to render
    wrote: bool = False  # True when something outside Vardex actually changed


@dataclass(frozen=True)
class Invocation:
    """One call as the tool sees it: its arguments, and what it is allowed to do."""

    arguments: dict[str, Any]
    permission: Permission  # the level the pack gave this tool
    dry_run: bool = False  # say what would happen, change nothing

    @property
    def may_write(self) -> bool:
        """True when this call may change something outside Vardex."""
        return self.permission in WRITING and not self.dry_run

    def argument(self, name: str, default: Any = None) -> Any:
        return self.arguments.get(name, default)


@dataclass(frozen=True)
class Tool:
    """One tool: what the model is told, and the code behind it."""

    name: str
    summary: str  # one line, always in the window
    description: str  # the whole manual, fetched with describe_tool
    schema: dict[str, Any]  # JSON Schema for the arguments
    run: Callable[[Invocation], ToolResult]
    writes: bool = False  # True when it can change something outside Vardex

    @property
    def spec(self) -> ToolSpec:
        """How it goes into a request: the summary, not the manual."""
        return ToolSpec(self.name, self.summary, self.schema)


Approver = Callable[[ToolCall, Tool], bool]


def arguments_schema(
    properties: dict[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    """A JSON Schema for a tool's arguments: an object, and nothing but the named properties."""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


@dataclass
class Toolbox:
    """The tools of one run, and the gate every call passes through."""

    tools: dict[str, Tool]
    permissions: Permissions
    approve: Approver | None = None  # asks a person; None means nobody is there to ask
    dry_run: bool = False
    calls: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """The tools as the model first sees them, in a fixed order so the cache holds."""
        return tuple(tool.spec for tool in self.tools.values())

    def level(self, name: str) -> Permission | None:
        return self.permissions.level(name)

    def call(self, call: ToolCall) -> ToolResult:
        """Run one call, or refuse it, and keep it. Every refusal is a result the model can
        read, never an exception: a model that asked for something it may not have should learn
        that and go on, not crash the run."""
        result = self.run(call)
        self.record(call, result)
        return result

    def run(self, call: ToolCall) -> ToolResult:
        """Run one call and time it, without keeping it.

        Separate from `call` so several calls from one turn can run in threads while the record
        is still written in the order the model asked, which keeps transcripts and recordings
        the same from run to run.
        """
        started = time.monotonic()
        result = self._call(call)
        return replace(result, seconds=time.monotonic() - started)

    def record(self, call: ToolCall, result: ToolResult) -> None:
        self.calls.append((call, result))

    def _call(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self.tools)) or "none"
            return ToolResult(
                f"There is no tool called {call.name!r}. The tools are: {known}.", is_error=True
            )
        permission = self.level(call.name)
        if permission is None:
            # The pack never said what this tool may do, so it may do nothing. Adding a tool
            # to the engine must not quietly grant it to every pack that upgrades.
            return ToolResult(
                f"{tool.name} has no permission level in this pack's permissions.yaml, so it "
                "cannot be called. Tell the person who asked; do not try again.",
                is_error=True,
            )
        if tool.writes and permission == "read":
            return ToolResult(
                f"{tool.name} would change something outside Vardex, and this pack allows it "
                "only to read. Tell the person what you would have done instead.",
                is_error=True,
            )
        if tool.writes and permission == "approve" and not self.dry_run:
            if self.approve is None:
                return ToolResult(
                    f"{tool.name} needs a person to approve this call, and nobody is there to "
                    "ask. Say what you would do and what approval it needs.",
                    is_error=True,
                )
            if not self.approve(call, tool):
                return ToolResult(
                    f"The person refused this {tool.name} call. Do not try it again; ask what "
                    "they would like instead.",
                    is_error=True,
                )
        try:
            return tool.run(Invocation(call.arguments, permission, self.dry_run))
        except (QueryError, SearchError, ToolFailure) as err:
            # Something the model may be able to work around: bad SQL, a search that could
            # not run, a connector that said no. Anything else is a bug and must not be hidden.
            return ToolResult(f"{tool.name} failed: {err}", is_error=True)


class ToolFailure(Exception):
    """Raised inside a tool when the call cannot be completed for a reason the model could
    work around: a missing record, a value out of range, a connector that refused."""


def describe_tool(toolbox: Toolbox) -> Tool:
    """The tool that hands out the other tools' manuals."""

    def run(call: Invocation) -> ToolResult:
        name = str(call.argument("tool", ""))
        tool = toolbox.tools.get(name)
        if tool is None:
            known = ", ".join(sorted(toolbox.tools))
            return ToolResult(
                f"There is no tool called {name!r}. The tools are: {known}.", is_error=True
            )
        level = toolbox.level(name)
        allowed = {
            "read": "It may only read.",
            "draft": "It may write a proposal for a person, and nothing more.",
            "approve": "It may act, once a person has approved the call.",
            "act": "It may act.",
            None: "This pack gives it no permission level, so it cannot be called.",
        }[level]
        return ToolResult(f"{tool.name}\n\n{tool.description}\n\n{allowed}")

    return Tool(
        name="describe_tool",
        summary="Read the full instructions for one of the other tools, by name.",
        description=(
            "Give the name of a tool and get its whole manual: what it does, what each "
            "argument means, what it returns, and what this pack allows it to do.\n\n"
            "Call this before using a tool whose one-line summary leaves you guessing, and "
            "never guess at arguments instead. One call costs a few hundred tokens; a wrong "
            "call costs a step and may mislead you."
        ),
        schema=arguments_schema(
            {"tool": {"type": "string", "description": "The tool's name."}}, ["tool"]
        ),
        run=run,
    )


def warehouse_tools(warehouse: Warehouse, knowledge: Knowledge) -> list[Tool]:
    """The tools that answer from the pack's tables: the catalogue, then the query."""
    described = set(knowledge.tables)

    def list_tables(call: Invocation) -> ToolResult:
        lines = [
            f"{name}: {doc.description.strip().splitlines()[0] if doc.description else ''}"
            for name, doc in sorted(knowledge.tables.items())
        ]
        metrics = [f"{m.name} ({m.unit}): {m.description}" for m in knowledge.metrics]
        text = (
            f"The warehouse is {warehouse.name}, dialect {warehouse.dialect}.\n\n"
            f"Tables ({len(lines)}), with describe_table for the columns:\n"
            + "\n".join(f"  {line}" for line in lines)
            + "\n\nMetrics, which are the numbers this pack answers with:\n"
            + "\n".join(f"  {line}" for line in metrics)
        )
        if knowledge.questions.glossary:
            words = ", ".join(sorted(knowledge.questions.glossary))
            text += f"\n\nThe pack defines these words: {words}. describe_table explains them."
        return ToolResult(text, structured={"tables": sorted(described)})

    def describe_table(call: Invocation) -> ToolResult:
        name = str(call.argument("table", ""))
        doc = knowledge.tables.get(name)
        if doc is None:
            return ToolResult(
                f"There is no described table {name!r}. Call list_tables for the ones there are.",
                is_error=True,
            )
        columns = "\n".join(f"  {c.name}: {c.description}" for c in doc.columns)
        glossary = "\n".join(
            f"  {word}: {meaning}" for word, meaning in sorted(knowledge.questions.glossary.items())
        )
        text = f"{name}\n\n{doc.description}\n\nColumns:\n{columns}"
        if glossary:
            text += f"\n\nThe pack's words:\n{glossary}"
        return ToolResult(text)

    def run_sql(call: Invocation) -> ToolResult:
        sql = str(call.argument("sql", ""))
        max_rows = int(call.argument("max_rows", 100) or 100)
        problems = check_sql(sql, knowledge.questions.dialect, described)
        if problems:
            # Checked before it runs, in the pack's own dialect, so the model is told what is
            # wrong with its SQL rather than whatever the database would have said.
            return ToolResult(
                "That SQL may not run:\n" + "\n".join(f"  {p}" for p in problems), is_error=True
            )
        result = warehouse.run(
            transpile(sql, knowledge.questions.dialect, warehouse.dialect), max_rows=max_rows
        )
        return ToolResult(
            format_table(result),
            structured={"columns": result.columns, "rows": [list(r) for r in result.rows]},
        )

    return [
        Tool(
            name="list_tables",
            summary="List the warehouse's tables and metrics, one line each. Start here for "
            "any question about numbers.",
            description=(
                "Returns every table you may query, with one line on what it holds, and every "
                "metric this pack defines, with its unit.\n\n"
                "Call it once before writing SQL. It costs a few hundred tokens and tells you "
                "the dialect, so you do not guess at either a table name or a function. Then "
                "call describe_table for the two or three tables you actually need: their "
                "columns are not listed here, because a whole catalogue would crowd out the "
                "question."
            ),
            schema=arguments_schema({}),
            run=list_tables,
        ),
        Tool(
            name="describe_table",
            summary="Columns and meanings of one table, by name.",
            description=(
                "Returns one table's description, every column with what it means, and the "
                "pack's own words for things.\n\n"
                "Call it for each table you are about to query, and read the column "
                "descriptions before you write SQL: this pack's columns carry units and "
                "conventions that their names do not show. A column that is not described "
                "here cannot be queried."
            ),
            schema=arguments_schema(
                {"table": {"type": "string", "description": "A table name from list_tables."}},
                ["table"],
            ),
            run=describe_table,
        ),
        Tool(
            name="run_sql",
            summary="Run one read-only SELECT on the warehouse and get the rows back.",
            description=(
                "Runs one query and returns the rows as a table. Write it in the dialect "
                "list_tables names, using only described tables, by their names alone.\n\n"
                "One statement, and a query: anything that writes is refused before it runs, "
                "as is a name you may not read. Ask for the rows you need and no more — "
                "max_rows defaults to 100, and every row you fetch you also pay to read.\n\n"
                "Prefer the pack's metrics to arithmetic of your own: they are the definitions "
                "the business agreed on. If a query fails, read the reason: it names the table, "
                "column or function that was wrong."
            ),
            schema=arguments_schema(
                {
                    "sql": {"type": "string", "description": "One SELECT statement."},
                    "max_rows": {
                        "type": "integer",
                        "description": "Rows to fetch, at most 1000. Default 100.",
                    },
                },
                ["sql"],
            ),
            run=run_sql,
        ),
    ]


def document_tools(
    index: Index,
    embedder: Embedder,
    reranker: Reranker | None = None,
    *,
    limit: int = SEARCH_LIMIT,
) -> list[Tool]:
    """The tools that answer from the pack's documents: search, then read."""
    found: dict[int, Passage] = {}  # what this run has seen, so a passage can be read in full

    def preview(text: str) -> str:
        words = text.split()
        if len(words) <= PREVIEW_WORDS:
            return text
        return " ".join(words[:PREVIEW_WORDS]) + " …"

    def search_documents(call: Invocation) -> ToolResult:
        query = str(call.argument("query", ""))
        filters = Filters(
            about=call.argument("about"),
            kind=call.argument("kind"),
            period=call.argument("period"),
            language=call.argument("language"),
            document=call.argument("document"),
        )
        result = search(index, embedder, query, filters, limit=limit, reranker=reranker)
        if not result.hits:
            return ToolResult(
                "No passage matched. Try other words, or drop a filter: a filter that no "
                "document matches returns nothing rather than something near it.",
                is_error=True,
            )
        lines = []
        for hit in result.hits:
            passage = hit.passage
            found[passage.id] = passage
            lines.append(
                f"[{passage.id}] {passage.document.title}, page {passage.page} "
                f"({passage.document.language})\n{preview(passage.text)}"
            )
        return ToolResult(
            f"{len(result.hits)} passages, the best first. read_passage gives the whole of "
            "one.\n\n" + "\n\n".join(lines),
            structured={"passages": [p.id for p in result.passages]},
        )

    def read_passage(call: Invocation) -> ToolResult:
        try:
            number = int(call.argument("id"))
        except (TypeError, ValueError):
            return ToolResult("Give the number in square brackets from a search.", is_error=True)
        passage = found.get(number)
        if passage is None:
            return ToolResult(
                f"Passage {number} is not among the ones this run found. Search first, then "
                "read a passage it returned.",
                is_error=True,
            )
        return ToolResult(
            f"{passage.document.title}, page {passage.page}\n{passage.link}\n\n{passage.text}"
        )

    return [
        Tool(
            name="search_documents",
            summary="Find passages in the pack's documents by meaning and by words. Start here "
            "for anything a report says in prose.",
            description=(
                "Searches every indexed document, by meaning and by words at once, and returns "
                "the best passages with a number, the document, the page and the first lines "
                "of the text.\n\n"
                "Ask it what you want to know, in a sentence, in either language: the search "
                "finds a Norwegian passage for an English question. Filter only on what the "
                "question actually says — about (an object's key, such as an organisation "
                "number), kind, period, language, document — because a filter nothing matches "
                "returns nothing at all.\n\n"
                "The text is cut short. Call read_passage with a number before you quote or "
                "take a figure from it: a number half-read is worse than no number."
            ),
            schema=arguments_schema(
                {
                    "query": {"type": "string", "description": "What you want to know."},
                    "about": {
                        "type": "string",
                        "description": "The key of the object it is "
                        "about, such as an organisation number.",
                    },
                    "kind": {"type": "string", "description": "Such as annual-report."},
                    "period": {"type": "string", "description": "Such as 2025 or 2026-Q2."},
                    "language": {"type": "string", "description": "en, nb or nn."},
                    "document": {"type": "string", "description": "One document, by its id."},
                },
                ["query"],
            ),
            run=search_documents,
        ),
        Tool(
            name="read_passage",
            summary="Read the whole of one passage a search returned, by its number.",
            description=(
                "Returns one passage in full, with its document, page and a link a person can "
                "open at that page.\n\n"
                "Read a passage before you quote it or take a number from it. Use the link in "
                "your answer so the person can check the page themselves. Only passages this "
                "run has already found can be read."
            ),
            schema=arguments_schema(
                {"id": {"type": "integer", "description": "The number in square brackets."}},
                ["id"],
            ),
            run=read_passage,
        ),
    ]


def build_toolbox(
    permissions: Permissions,
    *,
    warehouse: Warehouse | None = None,
    knowledge: Knowledge | None = None,
    index: Index | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    extra: list[Tool] | None = None,
    approve: Approver | None = None,
    dry_run: bool = False,
) -> Toolbox:
    """Every tool this run has, in one box. A part that was not connected adds no tools, so
    an agent with no index simply has no way to search documents, and is told so."""
    tools: list[Tool] = []
    if warehouse is not None and knowledge is not None:
        tools += warehouse_tools(warehouse, knowledge)
    if index is not None and embedder is not None:
        tools += document_tools(index, embedder, reranker)
    tools += extra or []
    box = Toolbox(
        {tool.name: tool for tool in tools}, permissions, approve=approve, dry_run=dry_run
    )
    helper = describe_tool(box)
    box.tools = {helper.name: helper, **box.tools}
    return box


def as_json(result: ToolResult) -> str:
    """A tool result as JSON, for a channel that talks to a program."""
    body: dict[str, Any] = {"content": result.content, "is_error": result.is_error}
    if result.structured is not None:
        body["data"] = result.structured
    return json.dumps(body, ensure_ascii=False, default=str)
