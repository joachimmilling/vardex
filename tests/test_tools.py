"""The toolbox: the gate every call passes through, and the tools behind it."""

import duckdb
import pytest
from fakes import FakeEmbedder, MemoryIndex

from vardex.documents import DocumentSpec
from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.llm import ToolCall
from vardex.permissions import Permissions
from vardex.semantic import ColumnDoc, Knowledge, Metric, ModelDoc, Ontology, Questions
from vardex.tools import (
    Invocation,
    Tool,
    Toolbox,
    ToolFailure,
    ToolResult,
    arguments_schema,
    build_toolbox,
)

TABLES = """
create table companies (orgnr varchar, name varchar);
insert into companies values ('100000008', 'DEMO LAKS AS'), ('100000016', 'DEMO DATA AS');
create table secrets (token varchar);
insert into secrets values ('do-not-read-me');
"""


def knowledge() -> Knowledge:
    return Knowledge(
        tables={
            "companies": ModelDoc(
                name="companies",
                description="One row per company.",
                columns=[
                    ColumnDoc(name="orgnr", description="Organisation number."),
                    ColumnDoc(name="name", description="Registered name."),
                ],
            )
        },
        metrics=[
            Metric(
                name="companies",
                description="How many companies.",
                table="companies",
                sql="count(*)",
                unit="companies",
            )
        ],
        ontology=Ontology(),
        questions=Questions(glossary={"company": "A limited company."}),
    )


def warehouse(tmp_path) -> DuckDBWarehouse:
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute(TABLES)
    return DuckDBWarehouse(path)


def box(tmp_path, **levels) -> Toolbox:
    return build_toolbox(
        Permissions(tools=levels), warehouse=warehouse(tmp_path), knowledge=knowledge()
    )


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


def test_list_tables_names_the_tables_the_metrics_and_the_dialect(tmp_path):
    result = box(tmp_path, list_tables="read").call(call("list_tables"))
    assert not result.is_error
    assert "companies: One row per company." in result.content
    assert "companies (companies): How many companies." in result.content
    assert "dialect duckdb" in result.content
    assert "secrets" not in result.content  # a table nobody described is not a table


def test_describe_table_gives_the_columns_and_the_packs_words(tmp_path):
    result = box(tmp_path, describe_table="read").call(call("describe_table", table="companies"))
    assert "orgnr: Organisation number." in result.content
    assert "company: A limited company." in result.content


def test_describe_table_refuses_a_table_nobody_described(tmp_path):
    result = box(tmp_path, describe_table="read").call(call("describe_table", table="secrets"))
    assert result.is_error
    assert "no described table 'secrets'" in result.content


def test_run_sql_returns_rows(tmp_path):
    result = box(tmp_path, run_sql="read").call(
        call("run_sql", sql="select name from companies order by orgnr")
    )
    assert not result.is_error
    assert "DEMO LAKS AS" in result.content
    assert result.structured == {"columns": ["name"], "rows": [["DEMO LAKS AS"], ["DEMO DATA AS"]]}


def test_run_sql_refuses_an_undescribed_table_before_it_runs(tmp_path):
    result = box(tmp_path, run_sql="read").call(call("run_sql", sql="select * from secrets"))
    assert result.is_error
    assert "secrets is not one of the tables you may use" in result.content


def test_run_sql_refuses_a_write(tmp_path):
    result = box(tmp_path, run_sql="read").call(call("run_sql", sql="delete from companies"))
    assert result.is_error
    assert "only a query may run" in result.content


def test_a_tool_with_no_permission_level_refuses_to_run(tmp_path):
    # The pack never mentioned run_sql, so it cannot be called, however good the SQL is.
    result = box(tmp_path, list_tables="read").call(call("run_sql", sql="select 1"))
    assert result.is_error
    assert "no permission level" in result.content


def test_an_unknown_tool_is_an_error_the_model_can_recover_from(tmp_path):
    result = box(tmp_path, list_tables="read").call(call("delete_everything"))
    assert result.is_error
    assert "no tool called 'delete_everything'" in result.content
    assert "list_tables" in result.content  # it is told what there is


def test_describe_tool_gives_the_manual_and_what_the_pack_allows(tmp_path):
    result = box(tmp_path, describe_tool="read", run_sql="read").call(
        call("describe_tool", tool="run_sql")
    )
    assert "One statement, and a query" in result.content
    assert "It may only read." in result.content


def test_the_model_first_sees_only_summaries(tmp_path):
    specs = box(tmp_path, list_tables="read", run_sql="read").specs
    summaries = {spec.name: spec.description for spec in specs}
    assert summaries["run_sql"].startswith("Run one read-only SELECT")
    assert "max_rows defaults to 100" not in summaries["run_sql"]  # that is in the manual


def test_the_tool_list_keeps_its_order_so_the_prompt_cache_holds(tmp_path):
    one = warehouse(tmp_path)

    def names() -> list[str]:
        built = build_toolbox(
            Permissions(tools={"run_sql": "read"}), warehouse=one, knowledge=knowledge()
        )
        return [spec.name for spec in built.specs]

    assert names() == names() == ["describe_tool", "list_tables", "describe_table", "run_sql"]


# Tools that write


def proposing_tool(record: list[str]) -> Tool:
    def run(invocation: Invocation) -> ToolResult:
        if invocation.may_write:
            record.append("wrote")
            return ToolResult("Done.", wrote=True)
        return ToolResult("Proposed, not done.")

    return Tool(
        name="change_something",
        summary="Change something outside Vardex.",
        description="It changes something.",
        schema=arguments_schema({}),
        run=run,
        writes=True,
    )


def writing_box(level: str | None, **kwargs) -> tuple[Toolbox, list[str]]:
    record: list[str] = []
    levels = {"change_something": level} if level else {}
    return build_toolbox(
        Permissions(tools=levels), extra=[proposing_tool(record)], **kwargs
    ), record


def test_a_writing_tool_a_pack_allows_only_to_read_refuses(tmp_path):
    toolbox, record = writing_box("read")
    result = toolbox.call(call("change_something"))
    assert result.is_error
    assert "allows it only to read" in result.content
    assert record == []


def test_at_draft_it_proposes_and_writes_nothing(tmp_path):
    toolbox, record = writing_box("draft")
    result = toolbox.call(call("change_something"))
    assert not result.is_error
    assert result.content == "Proposed, not done."
    assert record == []


def test_at_approve_it_asks_first(tmp_path):
    asked = []
    toolbox, record = writing_box(
        "approve", approve=lambda c, t: asked.append(c.name) is None or True
    )
    result = toolbox.call(call("change_something"))
    assert result.wrote and record == ["wrote"]
    assert asked == ["change_something"]


def test_at_approve_a_refusal_stops_it(tmp_path):
    toolbox, record = writing_box("approve", approve=lambda c, t: False)
    result = toolbox.call(call("change_something"))
    assert result.is_error
    assert "refused" in result.content
    assert record == []


def test_at_approve_with_nobody_to_ask_it_drafts_instead(tmp_path):
    toolbox, record = writing_box("approve")
    result = toolbox.call(call("change_something"))
    assert not result.is_error and not result.wrote
    assert result.content == "Proposed, not done."  # it stepped down a rung
    assert record == []


def test_at_act_it_just_does_it(tmp_path):
    toolbox, record = writing_box("act")
    assert toolbox.call(call("change_something")).wrote
    assert record == ["wrote"]


def test_a_dry_run_never_writes_however_high_the_level(tmp_path):
    toolbox, record = writing_box("act", dry_run=True)
    result = toolbox.call(call("change_something"))
    assert not result.wrote
    assert record == []


def test_a_failure_inside_a_tool_becomes_something_the_model_can_read():
    def run(invocation: Invocation) -> ToolResult:
        raise ToolFailure("there is no Company 999")

    tool = Tool("look_up", "Look something up.", "…", arguments_schema({}), run)
    toolbox = build_toolbox(Permissions(tools={"look_up": "read"}), extra=[tool])
    result = toolbox.call(call("look_up"))
    assert result.is_error
    assert "look_up failed: there is no Company 999" in result.content


def test_a_bug_in_a_tool_is_not_swallowed():
    def run(invocation: Invocation) -> ToolResult:
        raise ZeroDivisionError("a real bug")

    tool = Tool("broken", "Broken.", "…", arguments_schema({}), run)
    toolbox = build_toolbox(Permissions(tools={"broken": "read"}), extra=[tool])
    with pytest.raises(ZeroDivisionError):
        toolbox.call(call("broken"))


# Document tools


def passages(index: MemoryIndex) -> None:
    document = DocumentSpec(
        id="demo-2025",
        title="Demo Laks AS, annual report 2025",
        about="100000008",
        kind="annual-report",
        period="2025",
        language="en",
        url="https://example.test/demo.pdf",
    )
    index.put(
        document,
        "fingerprint",
        [(12, "The harvest of salmon rose in 2025. " + "more words " * 40), (13, "Oil and gas.")],
        [FakeEmbedder().vector("salmon harvest"), FakeEmbedder().vector("oil gas")],
    )


def document_box(**levels) -> tuple[Toolbox, MemoryIndex]:
    index = MemoryIndex()
    passages(index)
    embedder = FakeEmbedder()
    return build_toolbox(Permissions(tools=levels), index=index, embedder=embedder), index


def test_search_documents_returns_numbered_previews():
    toolbox, _ = document_box(search_documents="read")
    result = toolbox.call(call("search_documents", query="salmon harvest"))
    assert not result.is_error
    assert "Demo Laks AS, annual report 2025, page 12" in result.content
    assert "more words more …" in result.content  # cut short on purpose
    assert result.structured == {"passages": [1, 2]}


def test_read_passage_gives_the_whole_passage_and_a_link():
    toolbox, _ = document_box(search_documents="read", read_passage="read")
    toolbox.call(call("search_documents", query="salmon harvest"))
    result = toolbox.call(call("read_passage", id=1))
    assert "https://example.test/demo.pdf#page=12" in result.content
    assert "…" not in result.content


def test_a_passage_this_run_never_found_cannot_be_read():
    toolbox, _ = document_box(read_passage="read")
    result = toolbox.call(call("read_passage", id=99))
    assert result.is_error
    assert "Search first" in result.content


def test_a_filter_nothing_matches_says_so_rather_than_answering_from_nothing():
    toolbox, _ = document_box(search_documents="read")
    result = toolbox.call(call("search_documents", query="salmon", period="1999"))
    assert result.is_error
    assert "drop a filter" in result.content


def test_an_agent_without_an_index_has_no_way_to_search(tmp_path):
    toolbox = box(tmp_path, run_sql="read")
    assert "search_documents" not in toolbox.tools
    result = toolbox.call(call("search_documents", query="anything"))
    assert result.is_error  # and it is told so, rather than being given an empty answer
