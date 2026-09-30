from pathlib import Path

import pytest

from vardex.packs import PackError, load_pack

PACKS = Path(__file__).parent.parent / "packs"


def write_pack(folder: Path, text: str) -> Path:
    (folder / "pack.yaml").write_text(text, encoding="utf-8")
    return folder


def test_first_pack_is_valid():
    pack = load_pack(PACKS / "norwegian-companies")
    assert pack.name == "norwegian-companies"
    assert "analyst" in pack.recipes


def test_missing_pack_yaml(tmp_path):
    with pytest.raises(PackError):
        load_pack(tmp_path)


def test_unknown_recipe_is_rejected(tmp_path):
    folder = write_pack(tmp_path, "name: demo\ndescription: x\nrecipes: [fortune-teller]\n")
    with pytest.raises(PackError):
        load_pack(folder)


def test_misspelt_field_is_rejected(tmp_path):
    text = "name: demo\ndescription: x\nrecipes: [analyst]\nrecipies: [monitor]\n"
    with pytest.raises(PackError):
        load_pack(write_pack(tmp_path, text))


def test_the_norway_problem(tmp_path):
    # In YAML 1.1, an unquoted `no` means False. The schema must catch it.
    text = "name: demo\ndescription: x\nlanguages: [en, no]\nrecipes: [analyst]\n"
    with pytest.raises(PackError):
        load_pack(write_pack(tmp_path, text))
