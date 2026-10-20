"""Fixtures shared by the tests."""

import os

import psycopg
import pytest
from fakes import FakeEmbedder

from vardex import cli

SCHEMA = "vardex_test"


@pytest.fixture
def postgres_url() -> str:
    """The URL of an empty schema in a PostgreSQL database the tests may change, from
    VARDEX_TEST_POSTGRES_URL. Tests that need it are skipped when it is not set; CI sets it."""
    url = os.getenv("VARDEX_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("VARDEX_TEST_POSTGRES_URL is not set")
    with psycopg.connect(url, autocommit=True) as con:
        con.execute(f"drop schema if exists {SCHEMA} cascade")
        con.execute(f"create schema {SCHEMA}")
    return url + ("&" if "?" in url else "?") + f"options=-csearch_path%3D{SCHEMA}"


def create_tables(url: str, sql: str) -> None:
    """Run setup SQL, several statements at once, as the account in the URL."""
    with psycopg.connect(url, autocommit=True) as con:
        con.execute(sql)


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """Every command embeds with the fake embedder and reranks with none, so no test ever
    loads an embedding model. A test that needs a reranker sets its own."""
    monkeypatch.setattr(cli, "connect_embedder", lambda settings: FakeEmbedder())
    monkeypatch.setattr(cli, "connect_reranker", lambda settings, rerank=True: None)
