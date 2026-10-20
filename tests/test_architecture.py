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
}


def test_only_the_warehouses_import_a_database_driver():
    for file in ENGINE.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for module in imported_modules(tree):
            allowed = DRIVERS.get(module.split(".")[0])
            assert allowed is None or file.name in allowed, f"{file.name} imports {module!r}"
