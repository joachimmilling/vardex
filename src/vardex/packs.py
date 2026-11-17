"""The pack format, version 0: what a pack.yaml may contain, and how to load one."""

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Recipe = Literal[
    "analyst",
    "knowledge-assistant",
    "support-desk",
    "document-processor",
    "back-office-agent",
    "monitor",
]
# A language, by its ISO 639 code: en, nb, nn, sv. Which ones a pack reads is the pack's to
# say, in pack.yaml; the engine knows none of them by name.
Language = str
LANGUAGE_CODE = r"^[a-z]{2,3}$"


class LanguageSpec(BaseModel):
    """What the engine needs to know about one language a pack reads."""

    model_config = ConfigDict(extra="forbid")

    # PostgreSQL's text search configuration for its words, so a search for "rapporten"
    # finds "rapporterte": english, norwegian, swedish… simple stems nothing.
    stemmer: str = "simple"
    # The words after a number that change what it means, each with its power of ten, or
    # null for a percent: million is 6. A number followed by one is traced at that scale.
    scale_words: dict[str, int | None] = {}


class PackManifest(BaseModel):
    """The contents of pack.yaml."""

    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not ignored

    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    version: str = "0.1.0"
    description: str
    languages: dict[str, LanguageSpec] = Field(default={"en": LanguageSpec()}, min_length=1)
    recipes: list[Recipe] = Field(min_length=1)

    @field_validator("languages", mode="before")
    @classmethod
    def codes_are_text(cls, value: Any) -> Any:
        # YAML reads a bare no, on, yes or off as true or false, so Norwegian, written no:,
        # arrives as False. Say so, rather than "must be a string".
        if isinstance(value, dict):
            for code in value:
                if isinstance(code, bool):
                    raise ValueError(
                        f"a language code read as {str(code).lower()}: YAML reads a bare no, "
                        "yes, on or off as true or false, so put the code in quotes: 'no':"
                    )
        return value

    @model_validator(mode="after")
    def words_mean_one_thing(self) -> "PackManifest":
        """A pack reads all its languages' scale words in every text, so a word may not mean
        two things: billion is 10⁹ in English and 10¹² in Norwegian, and cannot be both."""
        for code in self.languages:
            if not re.match(LANGUAGE_CODE, code):
                raise ValueError(f"{code!r} is not a language code such as en or nb")
        seen: dict[str, tuple[str, int | None]] = {}
        for code, spec in self.languages.items():
            for word, power in spec.scale_words.items():
                word = word.lower()
                if word in seen and seen[word][1] != power:
                    other, before = seen[word]
                    raise ValueError(
                        f"the scale word {word!r} is {before} in {other} and {power} in {code}: "
                        "leave it out of one of them"
                    )
                seen[word] = (code, power)
        return self

    @property
    def scale_words(self) -> dict[str, int | None]:
        """Every scale word in every language the pack reads."""
        return {
            word.lower(): power
            for spec in self.languages.values()
            for word, power in spec.scale_words.items()
        }

    @property
    def stemmers(self) -> dict[str, str]:
        return {code: spec.stemmer for code, spec in self.languages.items()}


class PackError(Exception):
    """Raised when a pack is missing or invalid."""


def describe_problem(error: Any) -> str:
    """Turn one Pydantic error into a line such as `languages → item 2: must be ...`."""
    where = " → ".join(
        f"item {part + 1}" if isinstance(part, int) else str(part) for part in error["loc"]
    )
    ctx = error.get("ctx", {})
    got = f" (got {error['input']!r})"
    match error["type"]:
        case "missing":
            problem = "is required"
        case "extra_forbidden":
            problem = "is not a known field; check the spelling"
        case "literal_error":
            choices = ctx["expected"].replace("'", "").replace(" or ", ", ")
            problem = f"must be one of {choices}{got}"
        case "string_pattern_mismatch":
            problem = f"must be lowercase letters and digits, joined by hyphens{got}"
        case "too_short":
            problem = f"must have at least {ctx['min_length']} item(s)"
        case "model_type" | "dict_type":
            problem = "must be a mapping of field: value"
        case _:
            problem = error["msg"][0].lower() + error["msg"][1:] + got
    return f"{where or 'pack.yaml'}: {problem}"


def read_yaml[T: BaseModel](path: Path, model: type[T], kind: str) -> T:
    """Read a YAML file from a pack and validate it, or raise a PackError that lists every
    problem, one per line."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return model.model_validate(data)
    except yaml.YAMLError as err:
        raise PackError(f"{path} is not valid YAML:\n{err}") from err
    except ValidationError as err:
        problems = "\n".join(f"  {describe_problem(e)}" for e in err.errors())
        raise PackError(f"{path} is not a valid {kind}:\n{problems}") from err


def load_pack(folder: Path) -> PackManifest:
    """Read and validate the pack.yaml in a folder."""
    manifest = folder / "pack.yaml"
    if not manifest.is_file():
        raise PackError(f"No pack.yaml found in {folder}")
    return read_yaml(manifest, PackManifest, "pack")


# What a pack is made of, for its fingerprint: its declarations, SQL and prompts. Not its golden
# sets, which change no answer, and not what dbt builds from it.
PACK_FILES = (".yaml", ".yml", ".sql", ".md", ".txt")
NOT_THE_PACK = ("evals", "target", "logs", "dbt_packages", "__pycache__")


def fingerprint(folder: Path) -> str:
    """A short name for the exact contents of a pack, which changes when any file that can
    change an answer changes. Two traces with the same fingerprint ran the same pack."""
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if not path.is_file() or path.suffix not in PACK_FILES:
            continue
        if any(part in NOT_THE_PACK or part.startswith(".") for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:12]
