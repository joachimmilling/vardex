"""The engine must never depend on a specific pack."""

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENGINE = ROOT / "src" / "vardex"
PACKS = ROOT / "packs"


def test_engine_knows_no_pack_by_name():
    pack_names = [folder.name for folder in PACKS.iterdir() if folder.is_dir()]
    for file in ENGINE.rglob("*.py"):
        text = file.read_text(encoding="utf-8")
        for name in pack_names:
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
