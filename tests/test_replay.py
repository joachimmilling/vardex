"""Record a run with a fake model and a real DuckDB file, then replay it with neither."""

from array import array
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal

import duckdb
import pytest
from fakes import FakeModel
from test_text_to_sql import KNOWLEDGE, QUESTION, SQL, TABLES, TODAY, plan

from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.replay import (
    NotRecorded,
    Recording,
    RecordingClient,
    RecordingEmbedder,
    RecordingIndex,
    RecordingReranker,
    RecordingWarehouse,
    ReplayClient,
    ReplayEmbedder,
    ReplayIndex,
    ReplayReranker,
    ReplayWarehouse,
    digest,
    from_json,
    pack_vector,
    request_key,
    to_json,
    unpack_vector,
)
from vardex.retrieval import search
from vardex.sql import QueryError
from vardex.text_to_sql import answer, build_request

ANSWER = "Demo Laks AS had the highest operating margin in 2025: 23.6%."


@pytest.fixture
def recorded(tmp_path):
    """One question answered live and recorded, then saved and loaded again."""
    with duckdb.connect(tmp_path / "warehouse.duckdb") as con:
        con.execute(TABLES)
    warehouse = DuckDBWarehouse(tmp_path / "warehouse.duckdb")
    recording = Recording("claude-sonnet-5-5", warehouse.name, "duckdb", "2026-10-13")
    client = RecordingClient(FakeModel(plan(sql=SQL), ANSWER), recording)
    connection = RecordingWarehouse(warehouse, recording)
    live = answer(client, connection, KNOWLEDGE, QUESTION, today=TODAY)
    warehouse.close()
    recording.save(tmp_path / "recording.jsonl")
    return live, Recording.load(tmp_path / "recording.jsonl")


def test_a_replay_gives_the_recorded_answer_without_a_model_or_a_warehouse(recorded):
    live, recording = recorded
    replayed = answer(
        ReplayClient(recording), ReplayWarehouse(recording), KNOWLEDGE, QUESTION, today=TODAY
    )
    assert replayed.text == live.text == ANSWER
    assert replayed.result == live.result
    assert [a.cost_usd for a in replayed.answers] == [Decimal("0.004"), Decimal("0.004")]
    assert replayed.figures == live.figures


def test_a_changed_prompt_is_not_in_the_recording(recorded):
    _, recording = recorded
    client = ReplayClient(recording)
    request = build_request(KNOWLEDGE, ReplayWarehouse(recording), QUESTION, today=TODAY)
    client.send(request)  # the recorded plan
    with pytest.raises(NotRecorded, match="a model request that was not recorded"):
        client.send(replace(request, system=request.system + " Be brief."))


def test_each_recorded_answer_is_given_once(recorded):
    _, recording = recorded
    client = ReplayClient(recording)
    request = build_request(KNOWLEDGE, ReplayWarehouse(recording), QUESTION, today=TODAY)
    client.send(request)
    with pytest.raises(NotRecorded):
        client.send(request)


def test_the_key_covers_the_model_and_every_part_of_the_request(recorded):
    _, recording = recorded
    request = build_request(KNOWLEDGE, ReplayWarehouse(recording), QUESTION, today=TODAY)
    key = request_key("claude-sonnet-5-5", request)
    assert request_key("claude-opus-5-5", request) != key
    assert request_key("claude-sonnet-5-5", replace(request, effort="high")) != key
    assert request_key("claude-sonnet-5-5", replace(request, output_schema=None)) != key
    assert request_key("claude-sonnet-5-5", request) == key


def test_a_query_that_failed_fails_again_and_an_unknown_one_is_not_recorded(tmp_path):
    with duckdb.connect(tmp_path / "warehouse.duckdb") as con:
        con.execute(TABLES)
    warehouse = DuckDBWarehouse(tmp_path / "warehouse.duckdb")
    recording = Recording("m", warehouse.name, "duckdb", "2026-10-13")
    with pytest.raises(QueryError):
        RecordingWarehouse(warehouse, recording).run("select profit from accounts")
    warehouse.close()
    replayed = ReplayWarehouse(recording)
    with pytest.raises(QueryError, match="profit"):
        replayed.run("select profit from accounts")
    with pytest.raises(NotRecorded, match="a query that was not recorded: select 1"):
        replayed.run("select 1")


@pytest.mark.parametrize(
    "value",
    [Decimal("1480312000"), Decimal("0.0523"), date(2025, 12, 31), datetime(2026, 10, 5, 6, 15)],
)
def test_values_keep_their_type(value):
    assert from_json(to_json(value)) == value
    assert type(from_json(to_json(value))) is type(value)


def test_plain_values_stay_plain():
    for value in [None, True, 7, 0.5, "Vestland"]:
        assert to_json(value) == value == from_json(value)


def test_a_recording_holds_no_key_and_no_provider_format(recorded, tmp_path):
    text = (tmp_path / "recording.jsonl").read_text(encoding="utf-8")
    assert "api_key" not in text and "x-api-key" not in text
    assert '"stop_reason": "end"' in text  # Vardex's words, not end_turn


def test_a_picture_changes_the_key_but_a_message_without_one_keeps_its_key():
    from vardex.llm import Image, Message, Request

    plain = Request(system="s", messages=[Message("user", "Write out this page.")])
    with_picture = Request(
        system="s", messages=[Message("user", "Write out this page.", (Image(b"\x89PNG1"),))]
    )
    other_picture = Request(
        system="s", messages=[Message("user", "Write out this page.", (Image(b"\x89PNG2"),))]
    )
    keys = {request_key("m", r) for r in (plain, with_picture, other_picture)}
    assert len(keys) == 3
    assert request_key("m", plain) == digest(
        {
            "model": "m",
            "system": "s",
            "messages": [["user", "Write out this page."]],
            "max_tokens": 4096,
            "effort": None,
            "schema": None,
            "cache": [False, False],
        }
    )


def test_a_vector_is_kept_exactly():
    vector = [0.1, -2.5, 1 / 3]
    assert unpack_vector(pack_vector(vector)) == array("f", vector).tolist()


def test_a_search_is_recorded_and_replayed_with_no_index_and_no_model(tmp_path):
    from fakes import FakeEmbedder, FakeReranker, MemoryIndex
    from test_retrieval import document

    index, embedder = MemoryIndex(), FakeEmbedder()
    texts = [(1, "Salmon harvest rose"), (2, "Gas fell"), (3, "Salmon prices fell")]
    index.put(document(), "f", texts, embedder.embed_documents([t for _, t in texts]))
    recording = Recording("", "", "", "2026-10-16")
    live = search(
        RecordingIndex(index, recording),
        RecordingEmbedder(embedder, recording),
        "salmon harvest",
        reranker=RecordingReranker(FakeReranker(), recording),
    )
    recording.save(tmp_path / "search.jsonl")
    loaded = Recording.load(tmp_path / "search.jsonl")
    assert (loaded.embedder, loaded.reranker) == ("fake-embedder", "fake-reranker")
    replayed = search(
        ReplayIndex(loaded),
        ReplayEmbedder(loaded),
        "salmon harvest",
        reranker=ReplayReranker(loaded),
    )
    assert replayed == live
    lines = (tmp_path / "search.jsonl").read_text().splitlines()
    assert sum('"passage":' in line for line in lines) == 3  # each passage once
    with pytest.raises(NotRecorded, match="not embedded"):
        search(ReplayIndex(loaded), ReplayEmbedder(loaded), "gas")
