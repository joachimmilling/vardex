"""The Index for PostgreSQL with pgvector: passages, their vectors and their words in one place.

With postgres_warehouse.py, the only module in Vardex that imports psycopg; a test enforces
it. Unlike a warehouse, the index is Vardex's own: it creates its tables and writes to them, in
a schema of its own for each pack, so it may share a database with anything else.
"""

from collections.abc import Callable
from typing import Any

import psycopg
from psycopg import sql

from vardex.documents import DocumentSpec
from vardex.embeddings import Vector
from vardex.retrieval import Filters, Passage, SearchError

# Each language's stemmer: "rapporterte" and "rapporten" share the stem "rapport" in Norwegian.
STEMMERS = {"en": "english", "nb": "norwegian", "nn": "norwegian"}
# How many neighbours the vector index looks at per search. It must be at least the number of
# passages asked for, or the index returns fewer: pgvector's default is 40.
EF_SEARCH = 200
COLUMNS = "p.id, d.id, d.title, d.about, d.kind, d.period, d.language, d.url, p.page, p.text"


class PostgresIndex:
    """Passages of one pack's documents, with their vectors and stemmed words, in PostgreSQL."""

    def __init__(
        self,
        url: str,
        pack: str,
        model: str,
        dimensions: int,
        *,
        connect: Callable[..., Any] = psycopg.connect,
    ) -> None:
        self.schema = sql.Identifier("pack_" + pack.replace("-", "_"))
        try:
            self.con = connect(url, autocommit=True, connect_timeout=10)
            info = self.con.info
            self.name = f"PostgreSQL {info.user}@{info.host}:{info.port}/{info.dbname}"
            self.create(model, dimensions)
        except psycopg.Error as err:
            raise SearchError(f"could not open the index in PostgreSQL: {err}") from err

    def create(self, model: str, dimensions: int) -> None:
        """Create the tables on first use, and check that the index was built with the same
        embedding model: vectors from two models cannot be compared."""
        s = self.schema
        self.con.execute("create extension if not exists vector")
        self.con.execute(sql.SQL("create schema if not exists {}").format(s))
        self.con.execute(
            sql.SQL(
                "create table if not exists {}.settings (model text not null, dimensions int)"
            ).format(s)
        )
        row = self.con.execute(sql.SQL("select model, dimensions from {}.settings").format(s))
        found = row.fetchone()
        if found is None:
            self.con.execute(
                sql.SQL("insert into {}.settings values (%s, %s)").format(s), (model, dimensions)
            )
        elif found != (model, dimensions):
            raise SearchError(
                f"the index was built with {found[0]}, not {model}: build it again with "
                "vardex index --rebuild"
            )
        self.con.execute(
            sql.SQL(
                """
                create table if not exists {s}.documents (
                    id text primary key, title text not null, about text not null,
                    kind text not null, period text not null, language text not null,
                    url text not null, fingerprint text not null
                );
                create table if not exists {s}.passages (
                    id bigint generated always as identity primary key,
                    document text not null references {s}.documents on delete cascade,
                    page int not null,
                    text text not null,
                    stemmer regconfig not null,
                    embedding vector({d}) not null,
                    words tsvector generated always as (to_tsvector(stemmer, text)) stored
                );
                create index if not exists passages_embedding on {s}.passages
                    using hnsw (embedding vector_cosine_ops);
                create index if not exists passages_words on {s}.passages using gin (words);
                """
            ).format(s=self.schema, d=sql.Literal(dimensions))
        )

    def documents(self) -> dict[str, str]:
        rows = self.con.execute(
            sql.SQL("select id, fingerprint from {}.documents").format(self.schema)
        ).fetchall()
        return dict(rows)

    def put(
        self,
        document: DocumentSpec,
        fingerprint: str,
        pages: list[tuple[int, str]],
        vectors: list[Vector],
    ) -> None:
        """Replace a document's passages in one transaction: a search sees the old ones or the
        new ones, never half of each."""
        s = self.schema
        stemmer = STEMMERS[document.language]
        with self.con.transaction():
            self.con.execute(
                sql.SQL("delete from {}.documents where id = %s").format(s), (document.id,)
            )
            self.con.execute(
                sql.SQL("insert into {}.documents values (%s, %s, %s, %s, %s, %s, %s, %s)").format(
                    s
                ),
                (
                    document.id,
                    document.title,
                    document.about,
                    document.kind,
                    document.period,
                    document.language,
                    document.url,
                    fingerprint,
                ),
            )
            copy = sql.SQL(
                "copy {}.passages (document, page, text, stemmer, embedding) from stdin"
            ).format(s)
            with self.con.cursor() as cursor, cursor.copy(copy) as rows:
                for (page, text), vector in zip(pages, vectors, strict=True):
                    rows.write_row((document.id, page, text, stemmer, as_text(vector)))

    def remove(self, document: str) -> None:
        self.con.execute(
            sql.SQL("delete from {}.documents where id = %s").format(self.schema), (document,)
        )

    def by_meaning(self, vector: Vector, filters: Filters, limit: int) -> list[Passage]:
        where, params = conditions(filters)
        query = sql.SQL(
            "select {columns}, p.embedding <=> %s::vector as distance from {s}.passages as p"
            " join {s}.documents as d on d.id = p.document where {where}"
            " order by distance limit %s"
        ).format(columns=sql.SQL(COLUMNS), s=self.schema, where=where)
        with self.con.transaction():
            # Search the vector index, and keep searching past passages the filters rule out,
            # so a filter does not leave fewer than `limit` results (pgvector 0.8 and later).
            self.con.execute(f"set local hnsw.ef_search = {max(EF_SEARCH, limit)}")
            self.con.execute("set local hnsw.iterative_scan = relaxed_order")
            rows = self.con.execute(query, [as_text(vector), *params, limit]).fetchall()
        # relaxed_order may return two close passages in the wrong order; put them right
        return [passage(row[:-1]) for row in sorted(rows, key=lambda row: (row[-1], row[0]))]

    def by_words(self, query: str, filters: Filters, limit: int) -> list[Passage]:
        where, params = conditions(filters)
        # Any of the query's words, stemmed for each language, rather than all of them: a
        # question has more words than the passage that answers it.
        sql_query = sql.SQL(
            "with q as (select {any_word} as q)"
            " select {columns} from {s}.passages as p join {s}.documents as d"
            " on d.id = p.document, q where p.words @@ q.q and {where}"
            " order by ts_rank_cd(p.words, q.q) desc, p.id limit %s"
        ).format(
            any_word=sql.SQL(" || ").join(
                sql.SQL("replace(plainto_tsquery({}, %s)::text, ' & ', ' | ')::tsquery").format(
                    sql.Literal(stemmer)
                )
                for stemmer in sorted(set(STEMMERS.values()))
            ),
            columns=sql.SQL(COLUMNS),
            s=self.schema,
            where=where,
        )
        stemmers = len(set(STEMMERS.values()))
        rows = self.con.execute(sql_query, [*[query] * stemmers, *params, limit]).fetchall()
        return [passage(row) for row in rows]

    def close(self) -> None:
        self.con.close()


def conditions(filters: Filters) -> tuple[sql.Composable, list[str]]:
    """The WHERE clause for the filters that are set, and its parameters."""
    columns = {
        "about": "d.about", "kind": "d.kind", "period": "d.period",
        "language": "d.language", "document": "d.id",
    }  # fmt: skip
    parts, params = [], []
    for field, column in columns.items():
        value = getattr(filters, field)
        if value is not None:
            parts.append(sql.SQL(column + " = %s"))
            params.append(value)
    return (sql.SQL(" and ").join(parts) if parts else sql.SQL("true")), params


def as_text(vector: Vector) -> str:
    """A vector as pgvector reads it: [0.1,0.2,…]."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def passage(row: tuple[Any, ...]) -> Passage:
    pid, doc_id, title, about, kind, period, language, url, page, text = row
    document = DocumentSpec(
        id=doc_id, title=title, about=about, kind=kind, period=period, language=language, url=url
    )
    return Passage(pid, document, page, text)
