"""The engine must never depend on a specific pack, a model provider, a database or a library
that reads documents or runs embedding models."""

import ast
import subprocess
import sys
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


# Each provider's SDK, and the one module that may import it
SDKS = {
    "anthropic": "anthropic_client.py",
    "openai": "openai_client.py",
    "mlx_lm": "mlx_trainer.py",
}


def test_only_a_providers_own_client_imports_its_sdk():
    for file in ENGINE.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for module in imported_modules(tree):
            owner = SDKS.get(module.split(".")[0])
            assert owner is None or file.name == owner, f"{file.name} imports {module!r}"


def test_starting_vardex_imports_no_sdk_and_no_database_driver():
    """Every connector is an extra, found by its entry point when a setting asks for it. So
    the command-line tool must start without importing any of them: Vardex installed with no
    extras at all still starts, and says which extra a setting needs."""
    connectors = ["anthropic", "openai", "duckdb", "psycopg", "fastembed", "mlx_lm"]
    script = (
        f"import sys, vardex.cli; print(','.join(m for m in {connectors!r} if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "", f"starting vardex imported {loaded}"


def test_the_engine_names_no_connector_where_it_chooses_one():
    """connect and connect_warehouse look connectors up by name; neither names one in its
    code. Their docstrings may give examples."""
    tree = ast.parse((ENGINE / "cli.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("connect", "connect_warehouse"):
            code = node.body[1:] if ast.get_docstring(node) else node.body
            text = "\n".join(ast.unparse(statement) for statement in code).lower()
            for name in ("anthropic", "openai", "ollama", "duckdb", "postgres"):
                assert name not in text, f"{node.name} names {name}"


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
    # Tracing is a standard, not a library choice spread through the engine: one module
    # speaks OpenTelemetry, and everything else asks it for a span.
    "opentelemetry": {"tracing.py"},
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
