"""Settings, read from environment variables and a local .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str


def load_settings() -> Settings:
    """Read settings from the environment, after loading .env if there is one."""
    load_dotenv()  # reads .env in the current folder; never overrides real environment variables
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("VARDEX_MODEL") or DEFAULT_MODEL,  # an empty value means "use the default"
    )
