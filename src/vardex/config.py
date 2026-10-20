"""Settings, read from environment variables and a local .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    timeout_seconds: float = 120.0  # per attempt; the SDK's own default is ten minutes
    max_retries: int = 2  # extra attempts after a rate limit, an overload or a dropped connection
    data_dir: Path = Path("data")  # fetched files and the warehouse, one folder per pack
    warehouse_url: str | None = None  # a warehouse someone else runs; None means the pack's own
    query_timeout_seconds: float = 30.0  # a query that runs longer is stopped
    index_url: str | None = None  # PostgreSQL with pgvector, for the index of documents
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rerank_model: str | None = DEFAULT_RERANK_MODEL  # None: search without reranking


def load_settings() -> Settings:
    """Read settings from the environment, after loading .env if there is one."""
    load_dotenv()  # reads .env in the current folder; never overrides real environment variables
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("VARDEX_MODEL") or DEFAULT_MODEL,  # an empty value means "use the default"
        data_dir=Path(os.getenv("VARDEX_DATA_DIR") or "data"),
        warehouse_url=os.getenv("VARDEX_WAREHOUSE_URL") or None,
        index_url=os.getenv("VARDEX_INDEX_URL") or None,
        embedding_model=os.getenv("VARDEX_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        rerank_model=rerank_model(os.getenv("VARDEX_RERANK_MODEL")),
    )


def rerank_model(value: str | None) -> str | None:
    """The reranker to use: the default when unset or empty, none when set to none."""
    if not value:
        return DEFAULT_RERANK_MODEL
    return None if value.lower() == "none" else value
