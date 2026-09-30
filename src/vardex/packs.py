"""The pack format, version 0: what a pack.yaml may contain, and how to load one."""

from pathlib import Path
from typing import Literal

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


def load_pack(folder: Path) -> PackManifest:
    """Read and validate the pack.yaml in a folder."""
    manifest = folder / "pack.yaml"
    if not manifest.is_file():
        raise PackError(f"No pack.yaml found in {folder}")
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        return PackManifest.model_validate(data)
    except (yaml.YAMLError, ValidationError) as err:
        raise PackError(f"{manifest} is not a valid pack:\n{err}") from err
