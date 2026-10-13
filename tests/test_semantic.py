"""The semantic layer: loading a pack's description of its tables, and checking it."""

from pathlib import Path

import duckdb
import pytest

from vardex.duckdb_warehouse import DuckDBWarehouse
from vardex.packs import PackError
from vardex.semantic import check, load_knowledge

FILES = {
    "warehouse/models/models.yml": """
models:
  - name: companies
    description: One row per company.
    columns:
      - {name: orgnr, description: Organisation number., data_tests: [unique]}
      - {name: name, description: Registered name.}
      - {name: county, description: County.}
  - name: accounts
    description: One row per company and year.
    columns:
      - {name: accounts_key, description: orgnr and year.}
      - {name: orgnr, description: The company.}
      - {name: revenue, description: Revenue in NOK.}
""",
    "warehouse/metrics.yaml": """
metrics:
  - {name: revenue, description: Revenue., table: accounts, sql: sum(revenue), unit: NOK}
""",
    "warehouse/ontology.yaml": """
objects:
  - {name: Company, description: A company., table: companies, key: orgnr, label: name}
  - {name: Accounts, description: A year of accounts., table: accounts, key: accounts_key}
links:
  - {name: filed_by, description: Who filed them., from: Accounts.orgnr, to: Company}
""",
    "warehouse/questions.yaml": """
glossary:
  company: A limited company.
examples:
  - question: Which county has most companies?
    sql: select county, count(*) as companies from companies group by county
""",
}
TABLES = """
create table companies (orgnr varchar, name varchar, county varchar);
insert into companies values ('100000008', 'DEMO LAKS AS', 'Vestland'),
                             ('100000016', 'DEMO DATA AS', 'Oslo');
create table accounts (accounts_key varchar, orgnr varchar, revenue decimal(18, 0));
insert into accounts values ('100000008-2025', '100000008', 1480312000);
"""


def make_pack(folder: Path, **changes: str) -> Path:
    for name, text in {**FILES, **changes}.items():
        path = folder / "pack" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder / "pack"


def make_warehouse(folder: Path, sql: str = TABLES) -> DuckDBWarehouse:
    path = folder / "warehouse.duckdb"
    with duckdb.connect(path) as con:
        con.execute(sql)
    return DuckDBWarehouse(path)


def test_a_pack_that_agrees_with_itself_and_its_warehouse(tmp_path):
    knowledge = load_knowledge(make_pack(tmp_path))
    assert list(knowledge.tables) == ["companies", "accounts"]
    assert knowledge.ontology.object("Company").label == "name"
    assert knowledge.questions.glossary == {"company": "A limited company."}
    assert check(make_warehouse(tmp_path), knowledge) == []


def test_ontology_and_questions_may_be_left_out(tmp_path):
    pack = make_pack(tmp_path)
    (pack / "warehouse" / "ontology.yaml").unlink()
    (pack / "warehouse" / "questions.yaml").unlink()
    knowledge = load_knowledge(pack)
    assert knowledge.ontology.objects == []
    assert knowledge.questions.examples == []


@pytest.mark.parametrize(
    ("name", "text", "problem"),
    [
        (
            "warehouse/ontology.yaml",
            "objects: [{name: Company, description: x, table: firms, key: orgnr}]",
            "object Company: no described model firms",
        ),
        (
            "warehouse/ontology.yaml",
            "objects: [{name: Company, description: x, table: companies, key: id}]",
            "object Company: companies has no described column id",
        ),
        (
            "warehouse/questions.yaml",
            "examples: [{question: x, sql: 'drop table companies'}]",
            "example 1: only a query may run, not DROP",
        ),
        (
            "warehouse/metrics.yaml",
            "metrics: [{name: n, description: x, table: people, sql: count(*), unit: x}]",
            "metric n: no described model people",
        ),
    ],
)
def test_files_that_disagree_are_refused_on_load(tmp_path, name, text, problem):
    with pytest.raises(PackError) as caught:
        load_knowledge(make_pack(tmp_path, **{name: text}))
    assert problem in str(caught.value)


def test_a_link_to_an_unknown_object_is_refused(tmp_path):
    ontology = FILES["warehouse/ontology.yaml"].replace("to: Company", "to: Firm")
    with pytest.raises(PackError) as caught:
        load_knowledge(make_pack(tmp_path, **{"warehouse/ontology.yaml": ontology}))
    assert "link filed_by: there is no object Firm" in str(caught.value)


def test_check_finds_every_claim_the_warehouse_does_not_bear_out(tmp_path):
    knowledge = load_knowledge(make_pack(tmp_path))
    broken = (
        TABLES.replace("county varchar", "region varchar")
        + """
        insert into companies values ('100000008', 'DEMO LAKS AS AGAIN', 'Vestland');
        insert into companies values (null, 'NO NUMBER AS', 'Oslo');
        insert into accounts values ('100000099-2025', '100000099', 1);
    """
    )
    problems = check(make_warehouse(tmp_path, broken), knowledge)
    assert problems[:4] == [
        "model companies has no column county",
        "object Company: orgnr repeats in 1 row",
        "object Company: orgnr is empty in 1 row",
        "link filed_by: no Company for 1 row",
    ]
    assert problems[4].startswith('example 1: Binder Error: Referenced column "county"')
    assert len(problems) == 5
