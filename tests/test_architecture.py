"""The engine must never depend on a specific pack, a model provider, a database or a library
that reads documents or runs embedding models."""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENGINE = ROOT / "src" / "vardex"
PACKS = ROOT / "packs"


def test_engine_knows_no_pack_by_name():
    folders = [folder.name for folder in PACKS.iterdir() if folder.is_dir()]
    # A pack's name as it is written, and as it is written in code: norwegian_companies too
    names = {spelling for name in folders for spelling in (name, name.replace("-", "_"))}
    for file in ENGINE.rglob("*.py"):
        text = file.read_text(encoding="utf-8")
        for name in names:
            assert name not in text, f"{file.name} mentions the pack {name!r}"


def imported_modules(tree: ast.AST) -> list[str]:
    """Every absolute module name an import statement in the tree names."""
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def test_engine_imports_no_pack():
    for file in ENGINE.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for module in imported_modules(tree):
            assert module.split(".")[0] != "packs", f"{file.name} imports {module!r}"


def test_only_the_anthropic_client_imports_anthropic():
    for file in ENGINE.rglob("*.py"):
        if file.name == "anthropic_client.py":
            continue
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for module in imported_modules(tree):
            assert module.split(".")[0] != "anthropic", f"{file.name} imports {module!r}"


# Each database driver or library, and the only engine modules that may import it.
# warehouse.py builds a pack's DuckDB file; every query goes through a Warehouse, every search
# through an Index, every embedding through an Embedder, and only documents.py reads a PDF.
DRIVERS = {
    "duckdb": {"duckdb_warehouse.py", "warehouse.py"},
    "psycopg": {"postgres_warehouse.py", "pgvector_index.py"},
    "fastembed": {"fastembed_models.py"},
    "pypdfium2": {"documents.py"},
    # MCP is a protocol like any other outside system: one module offers Vardex's tools over
    # it, one uses other people's, and the engine itself knows nothing about it.
    "mcp": {"mcp_server.py", "mcp_client.py"},
}


def test_only_the_warehouses_import_a_database_driver():
    for file in ENGINE.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for module in imported_modules(tree):
            allowed = DRIVERS.get(module.split(".")[0])
            assert allowed is None or file.name in allowed, f"{file.name} imports {module!r}"


def test_a_channel_holds_no_logic_of_its_own():
    """The CLI and the MCP server may build a toolbox and print; they may not decide what a
    tool is allowed to do. That decision lives in tools.py, once, for every channel."""
    gate = {"WRITING", "LADDER"}  # the names a second gate would have to use
    for name in ("cli.py", "mcp_server.py"):
        text = (ENGINE / name).read_text(encoding="utf-8")
        for word in gate:
            assert word not in text, f"{name} looks like it decides permissions itself"


def test_only_two_places_carry_a_change_out():
    """A change to the world outside Vardex happens in exactly two places: the action tool,
    when somebody is at the keyboard to say yes, and the approval, when they said yes later.
    A third caller of a connector would be a way round the inbox, and the inbox is the record
    anybody asking "who agreed to this?" is going to be shown."""
    for file in ENGINE.rglob("*.py"):
        if file.name in ("actions.py", "approvals.py"):
            continue
        text = file.read_text(encoding="utf-8")
        for way in ("connector.carry_out(", "connector.undo("):
            assert way not in text, f"{file.name} changes the outside world itself"
