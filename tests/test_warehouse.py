"""Builds of a small pack made here, with every request answered by the test itself."""

from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pytest

from vardex.packs import PackError
from vardex.warehouse import WarehouseError, build, load_metrics, query

NOW = datetime(2026, 10, 5, 6, 0, tzinfo=UTC)
PEOPLE = "id,name,team\n1,Kari,sales\n2,Ola,sales\n3,Ingrid,support\n"
FILES = {
    "pack.yaml": "name: demo\ndescription: A test pack.\nrecipes: [analyst]\n",
    "warehouse/ingest.yaml": """
sources:
  - {name: people, description: Staff., url: 'https://example.no/people.csv', format: csv}
  - name: badges
    description: One request per person.
    url: https://example.no/badges/{key}
    keys: select id from raw.people
    history: true
""",
    "warehouse/metrics.yaml": """
metrics:
  - {name: headcount, description: People., table: people, sql: count(*), unit: persons}
""",
    "warehouse/dbt_project.yml": "name: demo\nprofile: vardex\nconfig-version: 2\n",
    "warehouse/profiles.yml": """
vardex:
  target: local
  outputs:
    local: {type: duckdb, path: "{{ env_var('VARDEX_WAREHOUSE') }}"}
""",
    "warehouse/models/sources.yml": """
sources:
  - name: raw
    schema: raw
    loaded_at_field: _fetched_at
    freshness: {error_after: {count: 2, period: day}}
    tables: [{name: people}, {name: badges}]
""",
    "warehouse/models/people.sql": """
select cast(id as integer) as id, name, team
from {{ source('raw', 'people') }}
""",
    "warehouse/models/people.yml": """
models:
  - name: people
    columns:
      - {name: id, data_tests: [unique, not_null]}
""",
}


def make_pack(folder: Path, **changes: str) -> Path:
    for name, text in {**FILES, **changes}.items():
        path = folder / "pack" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder / "pack"


def answer(people: str = PEOPLE):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/people.csv":
            return httpx2.Response(200, text=people)
        return httpx2.Response(200, json={"badge": f"B-{request.url.path[-1]}"})

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def offline() -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline")

    return httpx2.Client(transport=httpx2.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr("vardex.ingest.time.sleep", lambda seconds: None)


def test_a_build_fetches_loads_models_and_tests(tmp_path):
    lines = []
    result = build(make_pack(tmp_path), tmp_path / "data", answer(), now=NOW, say=lines.append)

    assert result.path == tmp_path / "data" / "demo" / "warehouse.duckdb"
    assert "people: 3 rows in raw.people" in lines
    assert "badges: fetched 3 keys: 3 × 200" in lines
    rows = query(result.path, "select name from people where team = 'sales' order by id")
    assert rows.fetchall() == [("Kari",), ("Ola",)]
    badges = query(result.path, "select body->>'badge' from raw.badges order by key").fetchall()
    assert badges == [("B-1",), ("B-2",), ("B-3",)]


def test_a_relative_data_folder(tmp_path, monkeypatch):
    pack = make_pack(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = build(pack, Path("data"), answer(), now=NOW)
    assert result.path == Path("data/demo/warehouse.duckdb")
    assert query(result.path, "select count(*) from people").fetchone() == (3,)


def test_a_second_build_the_same_day_fetches_nothing(tmp_path):
    pack = make_pack(tmp_path)
    build(pack, tmp_path / "data", answer(), now=NOW)
    lines = []
    build(pack, tmp_path / "data", offline(), now=NOW, say=lines.append)
    assert not any("fetched" in line for line in lines)


def test_a_failed_fetch_falls_back_on_the_copy_from_before(tmp_path):
    pack = make_pack(tmp_path)
    build(pack, tmp_path / "data", answer(), now=NOW)
    a_week_later = datetime(2026, 10, 12, tzinfo=UTC)
    result = build(pack, tmp_path / "data", offline(), now=a_week_later)
    assert any("using the copy from 2026-10-05" in w for w in result.warnings)
    assert query(result.path, "select count(*) from people").fetchone() == (3,)


def test_old_data_is_reported_but_still_used(tmp_path):
    # dbt compares with the real clock, so data fetched in January is months old by now
    january = datetime(2026, 1, 5, tzinfo=UTC)
    result = build(make_pack(tmp_path), tmp_path / "data", answer(), now=january)
    assert "source people (error): last fetched 2026-01-05 00:00" in result.stale
    assert result.path.is_file()


def test_a_failing_test_leaves_the_old_warehouse_in_place(tmp_path):
    pack = make_pack(tmp_path)
    build(pack, tmp_path / "data", answer(), now=NOW)
    duplicate = PEOPLE + "3,Ingrid again,support\n"
    with pytest.raises(WarehouseError) as caught:
        build(pack, tmp_path / "data", answer(duplicate), refresh=True, now=NOW)
    assert "test unique_people_id: 1 failing row" in str(caught.value)
    path = tmp_path / "data" / "demo" / "warehouse.duckdb"
    assert query(path, "select count(*) from people").fetchone() == (3,)


def test_a_metric_that_does_not_run_fails_the_build(tmp_path):
    metrics = (
        "metrics:\n  - {name: x, description: x, table: people, sql: sum(salary), unit: NOK}\n"
    )
    pack = make_pack(tmp_path, **{"warehouse/metrics.yaml": metrics})
    with pytest.raises(WarehouseError) as caught:
        build(pack, tmp_path / "data", answer(), now=NOW)
    assert "x: Binder Error" in str(caught.value)
    assert not (tmp_path / "data" / "demo" / "warehouse.duckdb").exists()


def test_a_typo_in_the_pack_fails_before_any_fetch(tmp_path):
    pack = make_pack(tmp_path, **{"warehouse/metrics.yaml": "metrics: []\n"})
    with pytest.raises(PackError):
        build(pack, tmp_path / "data", offline(), now=NOW)
    with pytest.raises(PackError):
        load_metrics(tmp_path)


def test_query_without_a_warehouse(tmp_path):
    with pytest.raises(WarehouseError) as caught:
        query(tmp_path / "warehouse.duckdb", "select 1")
    assert "Run vardex ingest first" in str(caught.value)
