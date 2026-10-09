"""Ingestion: fetch a pack's sources and land each fetch as a file, untouched.

A pack lists its sources in warehouse/ingest.yaml. Every fetch lands one new file in
<data>/<pack>/landing/<source>/, named by the time it was fetched. Landed files are never
edited, so the warehouse can always be thrown away and rebuilt from them.
"""

import gzip
import json
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vardex import __version__
from vardex.packs import PackError, read_yaml

STAMP = "%Y-%m-%dT%H%M%SZ"  # landed files are named by when they were fetched, in UTC
RETRY_STATUSES = {429, 502, 503, 504}  # worth another try: rate limits and overloads
RETRY_WAITS = (1.0, 2.0, 4.0)  # seconds to wait before each extra attempt


class Source(BaseModel):
    """One source in ingest.yaml: where its data comes from and how often to fetch it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")  # also the name of its raw table
    description: str
    url: str
    format: Literal["csv", "json"] = "json"
    keys: str | None = None  # SQL giving one column of keys; then the url is fetched per key
    refresh_days: int = Field(default=1, ge=1)  # fetch again when the newest file is this old
    history: bool = False  # keep every fetch, or only the newest
    workers: int = Field(default=4, ge=1, le=16)  # requests at once, for a source with keys

    @model_validator(mode="after")
    def keys_need_a_placeholder(self) -> "Source":
        if bool(self.keys) != ("{key}" in self.url):
            raise ValueError("a source with keys needs {key} in its url, and no other source may")
        if self.keys and self.format != "json":
            raise ValueError("a source with keys must be json")
        return self


class IngestSpec(BaseModel):
    """The contents of warehouse/ingest.yaml in a pack: its sources, in the order to load them."""

    model_config = ConfigDict(extra="forbid")

    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def names_are_unique(self) -> "IngestSpec":
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("every source needs its own name")
        return self


class IngestError(Exception):
    """Raised when a source cannot be fetched and there is no earlier copy to fall back on."""


def load_ingest(pack: Path) -> IngestSpec:
    """Read and validate warehouse/ingest.yaml in a pack folder."""
    path = pack / "warehouse" / "ingest.yaml"
    if not path.is_file():
        raise PackError(f"No warehouse/ingest.yaml in {pack}")
    return read_yaml(path, IngestSpec, "ingest file")


def http_client() -> httpx2.Client:
    """The client Vardex fetches with. It says who is asking, as public APIs request."""
    return httpx2.Client(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": f"vardex/{__version__} (+https://github.com/joachimmilling/vardex)"},
    )


def landed(folder: Path) -> list[Path]:
    """The files landed for one source, oldest first."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


def fetched_at(path: Path) -> datetime:
    """When a landed file was fetched, read from its name."""
    return datetime.strptime(path.name.split(".")[0], STAMP).replace(tzinfo=UTC)


def is_due(source: Source, folder: Path, now: datetime) -> bool:
    """Whether a source should be fetched again. Counted in calendar days, so a daily job
    that runs a minute earlier than yesterday still fetches."""
    files = landed(folder)
    return not files or (now.date() - fetched_at(files[-1]).date()).days >= source.refresh_days


def get(client: httpx2.Client, url: str, into: Path | None = None) -> httpx2.Response:
    """GET a URL, and try again after a rate limit, an overload or a dropped connection.

    With `into`, a successful body is streamed to that file instead of held in memory.
    """
    for wait in (*RETRY_WAITS, None):  # None marks the last attempt: whatever happens, stop
        try:
            with client.stream("GET", url) as response:
                if wait is None or response.status_code not in RETRY_STATUSES:
                    if into is not None and response.status_code == 200:
                        with into.open("wb") as file:
                            for chunk in response.iter_bytes():
                                file.write(chunk)
                    else:
                        response.read()
                    return response
        except httpx2.TransportError:
            if wait is None:
                raise
        time.sleep(wait)
    raise AssertionError("unreachable: the last attempt returns or raises")


def fetch_file(client: httpx2.Client, source: Source, folder: Path, now: datetime) -> Path:
    """Download a source's url into a new landed file, and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / ".download.part"  # renamed only when complete, so no half file ever lands
    try:
        response = get(client, source.url, into=part)
    except httpx2.TransportError as err:
        raise IngestError(f"{source.name}: could not reach {source.url}: {err}") from err
    if response.status_code != 200:
        part.unlink(missing_ok=True)
        raise IngestError(f"{source.name}: HTTP {response.status_code} from {source.url}")
    with part.open("rb") as file:
        gzipped = file.read(2) == b"\x1f\x8b"  # the magic bytes at the start of every gzip file
    path = folder / f"{now.strftime(STAMP)}.{source.format}{'.gz' if gzipped else ''}"
    part.rename(path)
    return path


def fetch_one(client: httpx2.Client, url: str, key: str) -> dict[str, Any]:
    """Fetch one key. Every outcome becomes a row, so one bad key never stops the rest."""
    try:
        response = get(client, url.replace("{key}", key))
    except httpx2.TransportError as err:
        return {"key": key, "status": None, "body": None, "error": str(err)}
    if response.status_code != 200:
        return {"key": key, "status": response.status_code, "body": None, "error": response.text}
    try:
        return {"key": key, "status": 200, "body": response.json(), "error": None}
    except json.JSONDecodeError:
        return {"key": key, "status": 200, "body": None, "error": "the body is not JSON"}


def fetch_keys(
    client: httpx2.Client, source: Source, keys: Iterable[str], folder: Path, now: datetime
) -> Counter:
    """Fetch a source's url once per key into one landed file of JSON lines, and count the
    HTTP statuses."""
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(source.workers) as pool:
        rows = list(pool.map(lambda key: fetch_one(client, source.url, key), keys))
    part = folder / ".download.part"
    with gzip.open(part, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    part.rename(folder / f"{now.strftime(STAMP)}.jsonl.gz")
    return Counter(row["status"] for row in rows)


def prune(source: Source, folder: Path) -> None:
    """Delete all but the newest file, unless the source keeps its history."""
    if not source.history:
        for old in landed(folder)[:-1]:
            old.unlink()


def describe_statuses(statuses: Counter) -> str:
    """A count of statuses such as `5,703 × 200, 43 × 500, 22 × 404`."""
    return ", ".join(f"{n:,} × {status or 'no answer'}" for status, n in statuses.most_common())
