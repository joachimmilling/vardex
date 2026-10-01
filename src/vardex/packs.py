"""The pack format, version 0: what a pack.yaml may contain, and how to load one."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Recipe = Literal[
    "analyst",
    "knowledge-assistant",
    "support-desk",
    "document-processor",
    "back-office-agent",
    "monitor",
]
Language = Literal["en", "nb", "nn"]  # English, Norwegian Bokmål, Norwegian Nynorsk


class PackManifest(BaseModel):
    """The contents of pack.yaml."""

    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not ignored

    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    version: str = "0.1.0"
    description: str
    languages: list[Language] = ["en"]
    recipes: list[Recipe] = Field(min_length=1)


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


def load_pack(folder: Path) -> PackManifest:
    """Read and validate the pack.yaml in a folder."""
    manifest = folder / "pack.yaml"
    if not manifest.is_file():
        raise PackError(f"No pack.yaml found in {folder}")
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        return PackManifest.model_validate(data)
    except yaml.YAMLError as err:
        raise PackError(f"{manifest} is not valid YAML:\n{err}") from err
    except ValidationError as err:
        problems = "\n".join(f"  {describe_problem(e)}" for e in err.errors())
        raise PackError(f"{manifest} is not a valid pack:\n{problems}") from err
