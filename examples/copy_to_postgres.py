"""Copy a pack's described tables from its DuckDB warehouse into PostgreSQL.

A stand-in for a warehouse a company already runs, so the same questions can be asked of both.
It uses DuckDB's postgres extension, which DuckDB downloads the first time. The account in
the URL must be allowed to create tables; give its password in PGPASSWORD, never in the URL.

    uv run python examples/copy_to_postgres.py packs/norwegian-companies \
        postgresql://postgres@localhost:5432/vardex
"""

import sys
from pathlib import Path

import duckdb

from vardex.config import load_settings
from vardex.packs import load_pack
from vardex.semantic import load_tables
from vardex.warehouse import warehouse_path


def main() -> None:
    pack, url = Path(sys.argv[1]), sys.argv[2]
    source = warehouse_path(load_settings().data_dir, load_pack(pack).name)
    con = duckdb.connect()
    con.execute("install postgres")
    con.execute("load postgres")
    con.execute(f"attach '{source}' as source (read_only)")
    con.execute(f"attach '{url}' as target (type postgres)")
    for table in load_tables(pack):
        con.execute(f"drop table if exists target.public.{table}")
        con.execute(f"create table target.public.{table} as select * from source.main.{table}")
        # Statistics, so PostgreSQL plans well from the first query, not after autovacuum
        con.execute(f"call postgres_execute('target', 'analyze public.{table}')")
        rows = con.execute(f"select count(*) from target.public.{table}").fetchone()[0]
        print(f"{table}: {rows:,} rows")


if __name__ == "__main__":
    main()
