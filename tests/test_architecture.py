"""The engine must never depend on a specific pack."""

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
