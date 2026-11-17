from pathlib import Path

import pytest
from pydantic import ValidationError

from vardex.packs import PackError, PackManifest, describe_problem, load_pack

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
    text = "name: demo\ndescription: x\nlanguages:\n  en: {}\n  no: {}\nrecipes: [analyst]\n"
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, text))
    assert "a language code read as false" in str(caught.value)
    assert "'no':" in str(caught.value)


def test_a_scale_word_means_one_thing_in_every_language(tmp_path):
    text = (
        "name: demo\ndescription: x\nrecipes: [analyst]\nlanguages:\n"
        "  en: {scale_words: {billion: 9}}\n  nb: {scale_words: {billion: 12}}\n"
    )
    with pytest.raises(PackError) as caught:
        load_pack(write_pack(tmp_path, text))
    assert "'billion' is 9 in en and 12 in nb" in str(caught.value)


def test_the_pack_names_its_stemmers_and_scale_words(tmp_path):
    text = (
        "name: demo\ndescription: x\nrecipes: [analyst]\nlanguages:\n"
        "  sv: {stemmer: swedish, scale_words: {miljoner: 6, Procent: null}}\n"
    )
    manifest = load_pack(write_pack(tmp_path, text))
    assert manifest.stemmers == {"sv": "swedish"}
    assert manifest.scale_words == {"miljoner": 6, "procent": None}


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


def test_a_problem_with_the_whole_reply_names_the_reply_not_a_file():
    try:
        PackManifest.model_validate_json("```sql\nselect 1\n```")
    except ValidationError as err:
        assert describe_problem(err.errors()[0]).startswith("the reply: invalid JSON")
    else:
        raise AssertionError("a reply that is not JSON parsed")
