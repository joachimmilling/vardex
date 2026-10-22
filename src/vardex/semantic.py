"""The semantic layer: what a pack's tables mean, and how a question may use them.

A pack describes its warehouse in four kinds of file in its warehouse/ folder:

- models/**/*.yml, dbt's descriptions of each model and column. A model without a description
  is not shown to a model writing SQL: what is not described cannot be queried.
- metrics.yaml, every number a question may be answered with.
- ontology.yaml, the objects questions are about, their keys and their links.
- questions.yaml, the pack's own words, and example questions with their SQL.

None of them creates anything. They describe tables that already exist, in a warehouse Vardex
built or in one a company runs, and `check` tests every claim they make against it.
"""

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vardex.documents import DocumentList
from vardex.packs import PackError, read_yaml
from vardex.sql import QueryError, Warehouse, check_sql, transpile

NAME = r"^[a-z][a-z0-9_]*$"


class ColumnDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")  # dbt's files also hold tests and settings

    name: str
    description: str = ""


class ModelDoc(BaseModel):
    """One model in a dbt properties file: a table, what it holds, and its columns."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    columns: list[ColumnDoc] = []


class DbtProperties(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[ModelDoc] = []


class Metric(BaseModel):
    """One metric in metrics.yaml: a name, what it means, and the SQL that computes it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME)
    description: str
    table: str  # the model it is computed from
    sql: str  # an aggregate expression over that table, such as sum(revenue)
    unit: str  # what the number is counted in, such as NOK or companies
    notes: list[str] = []  # what to watch for when using it


class Metrics(BaseModel):
    """The contents of warehouse/metrics.yaml: every metric a question may be answered with."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[Metric] = Field(min_length=1)

    @model_validator(mode="after")
    def names_are_unique(self) -> "Metrics":
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("every metric needs its own name")
        return self


class ObjectType(BaseModel):
    """A kind of thing questions are about, such as a company: a table, one row per object."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]*$")
    description: str
    table: str
    key: str  # the column that identifies one object; unique and never empty
    label: str | None = None  # the column that names one for people, such as a name
    values: list[str] = []  # columns whose values a question names, such as a county


class Link(BaseModel):
    """A many-to-one link: each object on the from side points to one object on the to side,
    by a column that holds the other object's key."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME)
    description: str
    from_: str = Field(alias="from", pattern=r"^[A-Z][A-Za-z0-9]*\.[a-z][a-z0-9_]*$")
    to: str

    @property
    def source(self) -> tuple[str, str]:
        """The object and column the link starts from, such as (AnnualAccounts, orgnr)."""
        name, column = self.from_.split(".")
        return name, column


class ActionField(BaseModel):
    """One thing an action needs to be told, with the type it must be given in."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME)
    description: str
    type: Literal["text", "integer", "amount", "choice"] = "text"
    choices: list[str] = []  # for type choice
    required: bool = True

    @model_validator(mode="after")
    def choices_match_type(self) -> "ActionField":
        if (self.type == "choice") != bool(self.choices):
            raise ValueError(f"field {self.name}: choices belong to type choice, and only it")
        return self

    @property
    def json_type(self) -> str:
        return {"text": "string", "integer": "integer", "amount": "number", "choice": "string"}[
            self.type
        ]


class Action(BaseModel):
    """Something a pack can ask to have done to one of its objects.

    An action is part of the ontology, not a tool bolted on beside it, because the thing it
    changes is one of the objects the pack already describes. That is what makes it checkable:
    the key must name a row that exists, and a person reading the proposal can look it up.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME)
    description: str
    # The object it acts on, such as Company. Not called `on`: YAML reads a bare on as true.
    object: str
    fields: list[ActionField] = []
    connector: str  # the outside system that carries it out, by the name Vardex knows it by
    confirm: str  # a sentence a person must be able to read and say yes or no to

    @model_validator(mode="after")
    def field_names_are_unique(self) -> "Action":
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"action {self.name}: every field needs its own name")
        return self


class Ontology(BaseModel):
    """The contents of warehouse/ontology.yaml."""

    model_config = ConfigDict(extra="forbid")

    objects: list[ObjectType] = []
    links: list[Link] = []
    actions: list[Action] = []

    @model_validator(mode="after")
    def links_join_known_objects(self) -> "Ontology":
        names = [o.name for o in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("every object needs its own name")
        for link in self.links:
            for end in (link.source[0], link.to):
                if end not in names:
                    raise ValueError(f"link {link.name}: there is no object {end}")
        if len({a.name for a in self.actions}) != len(self.actions):
            raise ValueError("every action needs its own name")
        for action in self.actions:
            if action.object not in names:
                raise ValueError(f"action {action.name}: there is no object {action.object}")
        return self

    def object(self, name: str) -> ObjectType:
        return next(o for o in self.objects if o.name == name)

    def action(self, name: str) -> Action:
        return next(a for a in self.actions if a.name == name)


class Example(BaseModel):
    """A question and the SQL that answers it, shown to the model as a worked example."""

    model_config = ConfigDict(extra="forbid")

    question: str
    sql: str


class Questions(BaseModel):
    """The contents of warehouse/questions.yaml."""

    model_config = ConfigDict(extra="forbid")

    dialect: str = "duckdb"  # the SQL dialect of the pack's metrics and examples
    glossary: dict[str, str] = {}  # the pack's words, and what each means in the tables
    style: list[str] = []  # how answers are written for this pack, such as amounts and units
    examples: list[Example] = []  # the pool of worked examples
    show_examples: int | None = Field(default=None, ge=1)  # per question; None shows them all
    choose_examples: Literal["similar", "first"] = "similar"  # the most like it, or the first


@dataclass(frozen=True)
class Knowledge:
    """Everything a pack tells a model about its warehouse."""

    tables: dict[str, ModelDoc]  # the described models: the only tables a model may query
    metrics: list[Metric]
    ontology: Ontology
    questions: Questions


def load_metrics(pack: Path) -> Metrics:
    """Read and validate warehouse/metrics.yaml in a pack folder."""
    path = pack / "warehouse" / "metrics.yaml"
    if not path.is_file():
        raise PackError(f"No warehouse/metrics.yaml in {pack}")
    return read_yaml(path, Metrics, "metrics file")


def load_tables(pack: Path) -> dict[str, ModelDoc]:
    """Every model the pack's dbt project describes, by name."""
    tables = {}
    for path in sorted((pack / "warehouse" / "models").rglob("*.yml")):
        for model in read_yaml(path, DbtProperties, "dbt properties file").models:
            tables[model.name] = model
    return tables


def load_knowledge(pack: Path) -> Knowledge:
    """Read the pack's semantic layer, and check that its files agree with each other.
    ontology.yaml and questions.yaml may be left out; metrics.yaml may not."""
    folder = pack / "warehouse"
    ontology, questions = folder / "ontology.yaml", folder / "questions.yaml"
    knowledge = Knowledge(
        tables=load_tables(pack),
        metrics=load_metrics(pack).metrics,
        ontology=read_yaml(ontology, Ontology, "ontology") if ontology.is_file() else Ontology(),
        questions=(
            read_yaml(questions, Questions, "questions file")
            if questions.is_file()
            else Questions()
        ),
    )
    problems = disagreements(knowledge)
    if problems:
        raise PackError(
            f"The semantic layer in {folder} does not agree with itself:\n"
            + "\n".join(f"  {p}" for p in problems)
        )
    return knowledge


def disagreements(knowledge: Knowledge) -> list[str]:
    """Every place where one file of the semantic layer names something another lacks."""
    columns = {name: {c.name for c in doc.columns} for name, doc in knowledge.tables.items()}
    problems = []

    def need(table: str, column: str | None, where: str) -> None:
        if table not in columns:
            problems.append(f"{where}: no described model {table}")
        elif column and column not in columns[table]:
            problems.append(f"{where}: {table} has no described column {column}")

    for metric in knowledge.metrics:
        need(metric.table, None, f"metric {metric.name}")
    for obj in knowledge.ontology.objects:
        need(obj.table, obj.key, f"object {obj.name}")
        for column in [obj.label, *obj.values]:
            if column:
                need(obj.table, column, f"object {obj.name}")
    for link in knowledge.ontology.links:
        name, column = link.source
        need(knowledge.ontology.object(name).table, column, f"link {link.name}")
    for number, example in enumerate(knowledge.questions.examples, start=1):
        for problem in check_sql(example.sql, knowledge.questions.dialect, set(columns)):
            problems.append(f"example {number}: {problem}")
    return problems


def rows(n: int) -> str:
    return f"{n:,} row{'' if n == 1 else 's'}"


def check(warehouse: Warehouse, knowledge: Knowledge) -> list[str]:
    """Everything the semantic layer claims that the warehouse does not bear out: a described
    column that is missing, a metric or example that does not run, a key that is not unique,
    a link that does not find its object. Empty means every claim holds."""
    actual = {t.name: {c.name for c in t.columns} for t in warehouse.tables()}
    dialect = knowledge.questions.dialect  # the pack's SQL is rewritten for the warehouse's
    problems = []
    for name, doc in knowledge.tables.items():
        if name not in actual:
            problems.append(f"model {name} is not in the warehouse")
            continue
        missing = [c.name for c in doc.columns if c.name not in actual[name]]
        if missing:
            problems.append(f"model {name} has no column {', '.join(missing)}")

    def first(sql: str, where: str) -> int | None:
        """The first value the query returns, or None after noting why it failed."""
        try:
            rows = warehouse.run(sql, max_rows=1).rows
        except QueryError as err:
            problems.append(f"{where}: {err}")
            return None
        return rows[0][0] if rows else None

    for metric in knowledge.metrics:
        sql = f"select {metric.sql} from {metric.table} where false"
        first(transpile(sql, dialect, warehouse.dialect), f"metric {metric.name}")
    for obj in knowledge.ontology.objects:
        sql = f"select count({obj.key}) - count(distinct {obj.key}) from {obj.table}"
        if repeated := first(sql, f"object {obj.name}"):
            problems.append(f"object {obj.name}: {obj.key} repeats in {rows(repeated)}")
        sql = f"select count(*) from {obj.table} where {obj.key} is null"
        if empty := first(sql, f"object {obj.name}"):
            problems.append(f"object {obj.name}: {obj.key} is empty in {rows(empty)}")
    for link in knowledge.ontology.links:
        name, column = link.source
        start, end = knowledge.ontology.object(name), knowledge.ontology.object(link.to)
        sql = (
            f"select count(*) from {start.table} as f where f.{column} is not null and not exists"
            f" (select 1 from {end.table} as t where t.{end.key} = f.{column})"
        )
        if lost := first(sql, f"link {link.name}"):
            problems.append(f"link {link.name}: no {link.to} for {rows(lost)}")
    for number, example in enumerate(knowledge.questions.examples, start=1):
        first(transpile(example.sql, dialect, warehouse.dialect), f"example {number}")
    return problems


def quoted(value: str) -> str:
    """A value as an SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def labels(warehouse: Warehouse, obj: ObjectType, keys: list[str]) -> dict[str, str]:
    """The label of each object among `keys` that the warehouse holds, by its key."""
    if not obj.label:
        raise PackError(f"object {obj.name} has no label to name it by")
    listed = ", ".join(quoted(key) for key in dict.fromkeys(keys))
    sql = f"select {obj.key}, {obj.label} from {obj.table} where {obj.key} in ({listed})"
    return {str(key): str(label) for key, label in warehouse.run(sql, max_rows=10_000).rows}


def find_key(
    warehouse: Warehouse, ontology: Ontology, object_name: str, name: str, keys: list[str]
) -> str:
    """The key of the one object among `keys` that a person's name for it names: its label,
    or the label's first words, ignoring case, so Mowi finds MOWI ASA. No match, or several,
    is a PackError that says which names come close, or which matched."""
    if object_name not in {o.name for o in ontology.objects}:
        raise PackError(f"the ontology has no object {object_name}")
    known = labels(warehouse, ontology.object(object_name), keys)
    wanted = name.casefold().split()
    found = [k for k, label in known.items() if label.casefold().split()[: len(wanted)] == wanted]
    if len(found) == 1:
        return found[0]
    if found:
        names = ", ".join(sorted(known[k] for k in found))
        raise PackError(f"{name!r} names several: {names}. Write more of the name.")
    close = difflib.get_close_matches(
        name.upper(), [label.upper() for label in known.values()], 3, 0.5
    )
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise PackError(f"No {object_name} with documents is named {name!r}.{hint}")


def check_documents(warehouse: Warehouse, ontology: Ontology, documents: DocumentList) -> list[str]:
    """Every document that is about an object the warehouse does not hold."""
    if documents.about not in {o.name for o in ontology.objects}:
        return [f"documents are about {documents.about}, which the ontology does not describe"]
    obj = ontology.object(documents.about)
    known = labels(warehouse, obj, [d.about for d in documents.documents])
    return [
        f"document {d.id}: no {documents.about} {d.about}"
        for d in documents.documents
        if d.about not in known
    ]
