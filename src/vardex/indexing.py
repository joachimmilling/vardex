"""Indexing: fetch a pack's documents, read them, and put their passages in the index.

A document is read again only when its file has changed: the index keeps a fingerprint of
each file. A document that has left the pack's list leaves the index too.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx2

from vardex.documents import (
    Ocr,
    document_path,
    fetch,
    fingerprint,
    load_documents,
    passages,
    read_document,
)
from vardex.embeddings import Embedder
from vardex.llm import Answer, ModelClient
from vardex.packs import load_pack
from vardex.retrieval import Index, in_context

Say = Callable[[str], None]  # prints one line of progress


@dataclass
class Indexed:
    """What an indexing run did, and what the user should hear about it."""

    documents: int = 0  # read and indexed in this run
    unchanged: int = 0
    removed: int = 0
    pages: int = 0
    passages: int = 0
    scanned: int = 0  # pages read by a model, from a picture
    warnings: list[str] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)  # the model calls for scanned pages


def index_pack(
    pack: Path,
    data_dir: Path,
    http: httpx2.Client,
    index: Index,
    embedder: Embedder,
    client: ModelClient | None,
    ocr: Ocr,
    *,
    rebuild: bool = False,
    say: Say = lambda line: None,
) -> Indexed:
    """Fetch every document in the pack's list, and index each one that is new or changed,
    or every one with `rebuild`."""
    spec = load_documents(pack)
    name = load_pack(pack).name
    before = index.documents()
    done = Indexed()
    for document in spec.documents:
        path = document_path(data_dir, name, document)
        if fetch(http, document, path):
            say(f"{document.id}: fetched {path.stat().st_size / 1e6:,.1f} MB")
        print_ = fingerprint(path)
        if not rebuild and before.get(document.id) == print_:
            done.unchanged += 1
            continue
        reading = read_document(path, client, ocr)
        done.answers += reading.answers
        pieces = [(page.number, text) for page in reading.pages for text in passages(page)]
        vectors = embedder.embed_documents(
            [in_context(document.title, page, text) for page, text in pieces]
        )
        index.put(document, print_, pieces, vectors)
        scanned = [page for page in reading.pages if page.read_by == "model"]
        unread = [page.number for page in reading.pages if page.read_by == "unread"]
        for page in scanned:
            if page.unconfirmed:
                done.warnings.append(
                    f"{document.id}, page {page.number}: the OCR program did not find "
                    f"{', '.join(page.unconfirmed)}"
                )
        if unread:
            done.warnings.append(
                f"{document.id}: {len(unread)} scanned pages were not read: that needs a model"
            )
        done.documents += 1
        done.pages += len(reading.pages)
        done.passages += len(pieces)
        done.scanned += len(scanned)
        read = f", {len(scanned)} read from a picture" if scanned else ""
        say(f"{document.id}: {len(reading.pages)} pages{read}, {len(pieces)} passages")
    for gone in sorted(set(before) - {document.id for document in spec.documents}):
        index.remove(gone)
        done.removed += 1
        say(f"{gone}: removed, no longer in the pack's list")
    return done
