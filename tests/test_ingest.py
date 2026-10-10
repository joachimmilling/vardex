import gzip
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest

from vardex.ingest import (
    IngestError,
    Source,
    fetch_file,
    fetch_keys,
    fetched_at,
    is_due,
    landed,
    load_ingest,
    prune,
)
from vardex.packs import PackError

PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"
NOW = datetime(2026, 10, 5, 6, 15, 2, tzinfo=UTC)
FILE = Source(name="units", description="x", url="https://example.no/units.csv", format="csv")
KEYED = Source(name="accounts", description="x", url="https://example.no/a/{key}", keys="select 1")


def client(handler) -> httpx2.Client:
    """A client whose requests never leave the test: handler answers them."""
    return httpx2.Client(transport=httpx2.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr("vardex.ingest.time.sleep", lambda seconds: None)


def test_the_first_pack_declares_its_sources():
    spec = load_ingest(PACK)
    names = [source.name for source in spec.sources]
    assert names[:2] == ["units", "accounts"]  # accounts takes its keys from units
    assert spec.sources[1].history


def write_ingest(tmp_path: Path, text: str) -> Path:
    (tmp_path / "warehouse").mkdir()
    (tmp_path / "warehouse" / "ingest.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def test_a_keyed_source_needs_a_placeholder(tmp_path):
    text = "sources:\n  - {name: a, description: x, url: 'https://x.no/a', keys: select 1}\n"
    with pytest.raises(PackError) as caught:
        load_ingest(write_ingest(tmp_path, text))
    assert "needs {key} in its url" in str(caught.value)


def test_source_names_are_unique(tmp_path):
    line = "  - {name: a, description: x, url: 'https://x.no/a'}\n"
    with pytest.raises(PackError) as caught:
        load_ingest(write_ingest(tmp_path, "sources:\n" + line + line))
    assert "every source needs its own name" in str(caught.value)


def test_a_missing_ingest_file(tmp_path):
    with pytest.raises(PackError):
        load_ingest(tmp_path)


def test_a_file_lands_with_the_time_in_its_name(tmp_path):
    body = gzip.compress(b"a,b\n1,2\n")
    path, changed = fetch_file(
        client(lambda r: httpx2.Response(200, content=body)), FILE, tmp_path, NOW
    )
    assert changed
    assert path.name == "2026-10-05T061502Z.csv.gz"  # gzip found from the bytes, not the url
    assert gzip.decompress(path.read_bytes()) == b"a,b\n1,2\n"
    assert fetched_at(path) == NOW
    assert landed(tmp_path) == [path]


def test_a_rate_limit_is_tried_again(tmp_path):
    replies = iter(
        [httpx2.Response(429), httpx2.Response(503), httpx2.Response(200, text="a\n1\n")]
    )
    calls = []

    def handler(request):
        calls.append(request)
        return next(replies)

    path, _ = fetch_file(client(handler), FILE, tmp_path, NOW)
    assert len(calls) == 3
    assert path.read_text() == "a\n1\n"


def test_a_failed_download_lands_nothing(tmp_path):
    with pytest.raises(IngestError) as caught:
        fetch_file(client(lambda r: httpx2.Response(404)), FILE, tmp_path, NOW)
    assert "HTTP 404" in str(caught.value)
    assert landed(tmp_path) == []
    assert not (tmp_path / ".download.part").exists()


def test_a_dropped_connection_gives_up_after_four_attempts(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx2.ConnectError("no route to host")

    with pytest.raises(IngestError) as caught:
        fetch_file(client(handler), FILE, tmp_path, NOW)
    assert len(calls) == 4
    assert "could not reach" in str(caught.value)


def test_every_key_becomes_a_row(tmp_path):
    def handler(request):
        key = request.url.path.rsplit("/", 1)[1]
        if key == "1":
            return httpx2.Response(200, json=[{"year": 2025}])
        if key == "2":
            return httpx2.Response(404)
        return httpx2.Response(500, text="layout not supported (BANK)")

    statuses = fetch_keys(client(handler), KEYED, ["1", "2", "3"], tmp_path, NOW)
    assert statuses == Counter({200: 1, 404: 1, 500: 1})
    [path] = landed(tmp_path)
    assert path.name == "2026-10-05T061502Z.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    assert rows[0] == {"key": "1", "status": 200, "body": [{"year": 2025}], "error": None}
    assert rows[2]["error"] == "layout not supported (BANK)"


def test_due_counts_calendar_days(tmp_path):
    (tmp_path / "2026-10-04T061502Z.csv").write_text("a\n")
    a_minute_earlier = datetime(2026, 10, 5, 6, 14, tzinfo=UTC)  # 23 hours 59 minutes later
    assert is_due(FILE, tmp_path, a_minute_earlier)
    assert not is_due(FILE, tmp_path, datetime(2026, 10, 4, 23, 59, tzinfo=UTC))
    assert is_due(FILE, tmp_path / "never-fetched", NOW)


def test_prune_keeps_only_the_newest_without_history(tmp_path):
    for day in ("03", "04", "05"):
        (tmp_path / f"2026-10-{day}T061502Z.csv").write_text("a\n")
    prune(KEYED.model_copy(update={"history": True}), tmp_path)
    assert len(landed(tmp_path)) == 3
    prune(FILE, tmp_path)
    assert [p.name for p in landed(tmp_path)] == ["2026-10-05T061502Z.csv"]


def test_an_unchanged_file_is_not_downloaded_again(tmp_path):
    def first(request):
        return httpx2.Response(200, text="a\n1\n", headers={"ETag": '"v1"'})

    fetch_file(client(first), FILE, tmp_path, NOW)
    seen = []

    def second(request):
        seen.append(request.headers.get("If-None-Match"))
        return httpx2.Response(304)

    tomorrow = datetime(2026, 10, 6, 6, 15, 2, tzinfo=UTC)
    path, changed = fetch_file(client(second), FILE, tmp_path, tomorrow)
    assert seen == ['"v1"']
    assert not changed
    assert path.name == "2026-10-06T061502Z.csv"
    assert path.read_text() == "a\n1\n"
