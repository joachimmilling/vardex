"""Reading a change feed: the cursor, the kinds we keep, and the end of the feed."""

import json

import httpx2
import pytest

from vardex.events import FeedSpec, read_feed

SPEC = FeedSpec(
    url="https://register.example/api/updates",
    items="_embedded.updates",
    key="organisasjonsnummer",
    position="oppdateringsid",
    cursor="oppdateringsid",
    since="dato",
    kind="endringstype",
    stamp="dato",
    keep=["Endring", "Sletting"],
    inclusive=True,
    page_size=3,
    max_pages=2,
    lookback_days=7,
)


def change(number: int, key: str, kind: str = "Endring") -> dict:
    return {
        "oppdateringsid": number,
        "organisasjonsnummer": key,
        "endringstype": kind,
        "dato": f"2026-10-{number:02d}T01:00:00.000Z",
    }


def feed(pages: dict[str | None, list[dict]]) -> tuple[httpx2.Client, list[httpx2.URL]]:
    """A client answering with one page per cursor, and the list of URLs it was asked for."""
    asked: list[httpx2.URL] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        asked.append(request.url)
        cursor = request.url.params.get("oppdateringsid")
        items = pages.get(cursor, [])
        body = {"_embedded": {"updates": items}} if items else {}
        return httpx2.Response(200, content=json.dumps(body))

    return httpx2.Client(transport=httpx2.MockTransport(handler)), asked


def test_a_first_read_starts_a_few_days_back_and_keeps_the_cursor():
    client, asked = feed({None: [change(10, "100000008")]})
    batch = read_feed(client, SPEC, None)
    assert "dato=2026-" in str(asked[0]) and "oppdateringsid" not in str(asked[0])
    assert [event.key for event in batch.events] == ["100000008"]
    assert batch.cursor == "11"  # one past the last change seen, because the feed is inclusive


def test_an_inclusive_cursor_is_stepped_past_so_nothing_is_read_twice():
    client, asked = feed({None: [change(10, "100000008")], "11": []})
    first = read_feed(client, SPEC, None)
    again = read_feed(client, SPEC, first.cursor)
    assert "oppdateringsid=11" in str(asked[1])
    assert again.events == [] and again.cursor == "11"


def test_only_the_kinds_the_pack_asked_for_are_kept():
    client, _ = feed(
        {
            None: [
                change(1, "100000008", "Ny"),
                change(2, "200000006", "Endring"),
                change(3, "300000004", "Sletting"),
            ]
        }
    )
    batch = read_feed(client, SPEC, None)
    assert [event.key for event in batch.events] == ["200000006", "300000004"]
    assert batch.read == 3  # what was there is counted, not only what we kept


def test_the_same_company_changing_twice_is_one_key_to_look_at():
    client, _ = feed({None: [change(1, "100000008"), change(2, "100000008")]})
    assert read_feed(client, SPEC, None).keys == ["100000008"]


def test_a_full_page_is_followed_by_the_next_one():
    client, asked = feed(
        {
            None: [change(n, f"{n}00000008") for n in (1, 2, 3)],
            "4": [change(4, "400000008")],
        }
    )
    batch = read_feed(client, SPEC, None)
    assert len(asked) == 2
    assert len(batch.events) == 4 and batch.pages == 2 and not batch.more


def test_a_backlog_longer_than_the_cap_stops_and_says_so():
    client, _ = feed(
        {
            None: [change(n, f"{n}00000008") for n in (1, 2, 3)],
            "4": [change(n, f"{n}00000008") for n in (4, 5, 6)],
            "7": [change(7, "700000008")],
        }
    )
    batch = read_feed(client, SPEC, None)
    assert batch.pages == 2 and batch.more  # max_pages is 2
    assert batch.cursor == "7"  # the next run carries on from there, not from the beginning
    assert len(batch.events) == 6


def test_the_end_of_the_feed_is_no_list_at_all_and_that_is_not_an_error():
    client, _ = feed({None: []})  # the handler answers {} when there is nothing
    batch = read_feed(client, SPEC, None)
    assert batch.events == [] and batch.read == 0 and not batch.more


def test_a_feed_that_has_changed_shape_stops_the_run_with_the_field_name():
    from vardex.events import FeedError

    client, _ = feed({None: [{"oppdateringsid": 1, "orgnr": "100000008"}]})
    with pytest.raises(FeedError, match="has no organisasjonsnummer"):
        read_feed(client, SPEC, None)


def test_a_feed_with_no_date_parameter_simply_starts_at_the_beginning():
    spec = SPEC.model_copy(update={"since": None})
    client, asked = feed({None: [change(1, "100000008")]})
    read_feed(client, spec, None)
    assert "dato" not in str(asked[0])


def test_a_page_size_the_feed_would_refuse_is_refused_here_first():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FeedSpec(**(SPEC.model_dump() | {"page_size": 5000}))


def test_a_cursor_that_cannot_be_stepped_past_drops_the_row_it_names():
    spec = SPEC.model_copy(update={"position": "dato", "cursor": "dato"})
    first = change(5, "100000008")
    client, _ = feed({None: [first]})
    batch = read_feed(client, spec, None)
    assert batch.cursor == first["dato"]  # a timestamp cannot be stepped past

    def handler(request):
        return httpx2.Response(200, content=json.dumps({"_embedded": {"updates": [first]}}))

    again = read_feed(httpx2.Client(transport=httpx2.MockTransport(handler)), spec, batch.cursor)
    assert again.events == []
