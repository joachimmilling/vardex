"""Reading a source's own feed of changes, from where we left off.

A watch that asks "what changed?" by re-reading everything does not scale and cannot tell
you what moved. Most registers and systems of record publish a feed instead: an
append-only list of changes, each with a position, which you read forward from a cursor
you keep yourself.

This module is the provider-neutral half of that. A pack declares where its feed is and
what the fields are called, in its workflow file; Vardex keeps the cursor and reads
forward. Nothing here knows about any particular register, which is what lets the same
code read a Salesforce change feed, a Kafka offset or a database's change table later.

Three things make a cursor feed harder than it looks, and all three are handled here:

- **The cursor may be inclusive.** Several feeds, the Brønnøysund register's among them,
  return the row you named as well as the ones after it. Read from the cursor you last
  saw and you process that change for ever.
- **The backlog may be enormous.** A feed nobody has read for a month holds a month of
  changes. Reading it all in one run is a run that never ends, so a read is capped and
  says plainly that there is more; the next run carries on from the new cursor.
- **The end of a feed is not an empty list.** It is often no list at all, which is why the
  items are looked up by path and a missing path means "nothing", not a crash.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
from pydantic import BaseModel, ConfigDict, Field

from vardex.ingest import get

MAX_PAGE = 1000  # most feeds cap a page; ask for more and the request is refused
BRREG_STAMP = "%Y-%m-%dT%H:%M:%S.000Z"  # the exact shape this kind of API asks for


class FeedSpec(BaseModel):
    """Where a pack's change feed is, and what its fields are called."""

    model_config = ConfigDict(extra="forbid")

    url: str
    items: str  # dotted path to the list of changes, such as _embedded.oppdaterteEnheter
    key: str  # the field naming what changed, such as organisasjonsnummer
    position: str  # the field holding the cursor value, such as oppdateringsid
    cursor: str  # the query parameter that resumes from a position
    since: str | None = None  # the query parameter that starts from a time, for a first read
    kind: str | None = None  # the field saying what kind of change it was
    stamp: str | None = None  # the field holding when it happened
    keep: list[str] = []  # the kinds worth reading; empty keeps every kind
    inclusive: bool = True  # the cursor's own row comes back again, so step past it
    page_size: int = Field(default=500, ge=1, le=MAX_PAGE)
    size: str = "size"  # the query parameter for the page size
    max_pages: int = Field(default=20, ge=1)  # one read's cap; the rest waits for next time
    lookback_days: int = Field(default=7, ge=1)  # how far back a first read starts


@dataclass(frozen=True)
class Event:
    """One change, in Vardex's words rather than the feed's."""

    key: str  # what changed, by the key the pack's ontology uses
    kind: str  # what kind of change, in the feed's own words
    at: str  # when, as the feed said it
    position: str  # where this change sits in the feed


@dataclass(frozen=True)
class Batch:
    """What one read of a feed gave back."""

    events: list[Event]
    cursor: str | None  # where to read from next time; None when nothing was read
    pages: int
    more: bool  # the cap was reached and there is still a backlog
    read: int  # changes seen, before the kinds we do not care about were dropped

    @property
    def keys(self) -> list[str]:
        """What changed, each key once, in the order first seen."""
        return list(dict.fromkeys(event.key for event in self.events))


class FeedError(Exception):
    """Raised when a feed cannot be read, or its answer is not the shape the pack declared."""


def dig(data: Any, path: str) -> Any:
    """Follow a dotted path through nested dictionaries. A missing step gives None.

    The end of a feed is usually no list rather than an empty one, so "not there" has to
    be an ordinary answer here, not an exception.
    """
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def to_event(spec: FeedSpec, item: dict[str, Any]) -> Event:
    """One item from the feed as an Event. A missing key or position is an error: a feed
    whose shape has changed must stop the run rather than quietly read nothing."""
    key = dig(item, spec.key)
    position = dig(item, spec.position)
    if key is None or position is None:
        raise FeedError(
            f"a change in the feed has no {spec.key} or {spec.position}: {item}. "
            "Check the field names in the workflow's feed block against the feed itself."
        )
    return Event(
        key=str(key),
        kind=str(dig(item, spec.kind) or "") if spec.kind else "",
        at=str(dig(item, spec.stamp) or "") if spec.stamp else "",
        position=str(position),
    )


def next_cursor(spec: FeedSpec, position: str) -> str:
    """Where to read from after seeing this position.

    An inclusive feed returns the row you name, so the cursor has to step past it. That
    only works on positions that count; a feed whose positions are timestamps or opaque
    strings is read again from the last one, and the step that reads it must cope with
    seeing the last change twice. Which is why writing work is journalled.
    """
    if spec.inclusive and position.isdigit():
        return str(int(position) + 1)
    return position


def first_cursor(spec: FeedSpec, now: datetime | None = None) -> dict[str, str]:
    """The query parameters for a feed nobody has read yet: start a few days back.

    Starting from the beginning of a register's history would read millions of changes to
    learn nothing. Starting from now would miss whatever happened while the pack was being
    built. A short look back is the useful answer, and it is the pack's to choose.
    """
    if not spec.since:
        return {}
    started = (now or datetime.now(UTC)) - timedelta(days=spec.lookback_days)
    return {spec.since: started.strftime(BRREG_STAMP)}


def read_feed(
    client: httpx2.Client,
    spec: FeedSpec,
    cursor: str | None = None,
    *,
    now: datetime | None = None,
) -> Batch:
    """Read a feed forward from a cursor, a page at a time, up to the page cap.

    Paging is done by cursor, not by page number. Page numbers are what the APIs offer and
    they are the wrong tool twice over: they shift under you as new changes arrive, and
    they are usually capped — this kind of register refuses any request where
    size × (page + 1) goes over 10,000, so page 21 of a 500-row page size does not exist.
    A cursor has neither problem.
    """
    events: list[Event] = []
    seen = 0
    at = cursor
    pages = 0
    keep = set(spec.keep)
    # A cursor that counts was stepped past when it was saved, so nothing comes back twice.
    # A cursor that is a timestamp or an opaque string could not be stepped, so the row it
    # names arrives again and has to be dropped here.
    repeats = bool(cursor) and spec.inclusive and not str(cursor).isdigit()
    while pages < spec.max_pages:
        parameters = {spec.size: str(spec.page_size)}
        parameters |= {spec.cursor: at} if at else first_cursor(spec, now)
        response = get(client, str(httpx2.URL(spec.url).copy_merge_params(parameters)))
        try:
            body = response.json()
        except ValueError as err:
            raise FeedError(f"{spec.url} did not answer with JSON: {err}") from err
        items = dig(body, spec.items) or []
        if not isinstance(items, list):
            raise FeedError(f"{spec.items} in the feed's answer is not a list of changes")
        pages += 1
        if not items:
            return Batch(events, at, pages, more=False, read=seen)
        for item in items:
            event = to_event(spec, item)
            if repeats and event.position == cursor:
                continue  # the cursor's own row, already dealt with last time
            seen += 1
            if not keep or event.kind in keep:
                events.append(event)
        at = next_cursor(spec, to_event(spec, items[-1]).position)
        if len(items) < spec.page_size:
            return Batch(events, at, pages, more=False, read=seen)
    # The cap was reached with changes still waiting. The cursor is saved all the same, so
    # the next run starts where this one stopped instead of reading the backlog again.
    return Batch(events, at, pages, more=True, read=seen)
