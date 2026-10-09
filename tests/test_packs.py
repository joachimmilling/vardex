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
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, text))
    assert "  recipies: is not a known field; check the spelling" in str(caught.value)


def test_the_norway_problem(tmp_path):
    # In YAML 1.1, an unquoted `no` means False. The schema must catch it.
    text = "name: demo\ndescription: x\nlanguages: [en, no]\nrecipes: [analyst]\n"
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, text))
    assert "languages → item 2: must be one of en, nb, nn (got False)" in str(caught.value)


def test_every_problem_gets_one_line(tmp_path):
    text = "name: Demo\nrecipes: []\n"
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, text))
    lines = str(caught.value).splitlines()[1:]
    assert lines == [
        "  name: must be lowercase letters and digits, joined by hyphens (got 'Demo')",
        "  description: is required",
        "  recipes: must have at least 1 item(s)",
    ]


def test_a_list_instead_of_a_mapping(tmp_path):
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, "- analyst\n"))
    assert "pack.yaml: must be a mapping of field: value" in str(caught.value)
