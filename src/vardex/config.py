"""Settings, read from environment variables and a local .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    timeout_seconds: float = 120.0  # per attempt; the SDK's own default is ten minutes
    max_retries: int = 2  # extra attempts after a rate limit, an overload or a dropped connection
    data_dir: Path = Path("data")  # fetched files and the warehouse, one folder per pack


def load_settings() -> Settings:
    """Read settings from the environment, after loading .env if there is one."""
    load_dotenv()  # reads .env in the current folder; never overrides real environment variables
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("VARDEX_MODEL") or DEFAULT_MODEL,  # an empty value means "use the default"
        data_dir=Path(os.getenv("VARDEX_DATA_DIR") or "data"),
    )
