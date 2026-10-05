"""Structured extraction: a document in; checked values out, or a clear reason why not.

A pack describes what to extract in a YAML file, extractions/<name>.yaml: instructions for the
model and the fields to return. The engine turns the fields into a JSON schema, asks the model
for exactly that shape, and checks the result before anyone can use it.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
    create_model,
    model_validator,
)

from vardex.llm import Answer, Effort, Message, ModelClient, Request
from vardex.packs import PackError, describe_problem
from vardex.prompts import EXTRACT

FieldType = Literal["text", "integer", "amount", "choice"]
# An amount is a JSON number in the schema, so the model cannot answer "1 480 312" as text,
# and an exact Decimal in Python, so no float ever touches it.
Amount = Annotated[Decimal, WithJsonSchema({"type": "number"})]
PYTHON_TYPES: dict[str, Any] = {"text": str, "integer": int, "amount": Amount}
MAX_VALUES = 16  # structured outputs accept at most 16 fields that may be empty in one schema
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class FieldSpec(BaseModel):
    """One field to extract, as a pack describes it."""

    model_config = ConfigDict(extra="forbid")

    type: FieldType
    description: str
    choices: list[str] | None = None  # only for type: choice
    required: bool = True  # a required field left empty makes the extraction fail
    quote: bool = False  # also ask for the words in the document the value comes from

    @model_validator(mode="after")
    def choices_only_for_choice(self) -> "FieldSpec":
        if (self.type == "choice") != bool(self.choices):
            raise ValueError("a field of type choice needs choices, and no other type takes them")
        return self


class ExtractionSpec(BaseModel):
    """The contents of extractions/<name>.yaml in a pack."""

    model_config = ConfigDict(extra="forbid")

    description: str
    instructions: str
    fields: dict[str, FieldSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def names_and_size(self) -> "ExtractionSpec":
        for name in self.fields:
            if not FIELD_NAME.match(name) or name == "problem" or name.endswith("_quote"):
                raise ValueError(
                    f"{name!r} is not a usable field name: use lowercase letters, digits and _, "
                    "and avoid 'problem' and names ending in _quote"
                )
        values = 1 + len(self.fields) + sum(f.quote for f in self.fields.values())
        if values > MAX_VALUES:
            raise ValueError(f"{values} values with quotes and problem; the limit is {MAX_VALUES}")
        return self


@dataclass(frozen=True)
class Extraction:
    """Values that passed every check, and the answer they came from."""

    values: dict[str, Any]
    answer: Answer


class ExtractionError(Exception):
    """Raised when a document gives no usable values. The message says why."""

    def __init__(self, reason: str, answer: Answer | None = None) -> None:
        super().__init__(reason)
        self.answer = answer  # so the call can still be logged and paid for


def load_extraction(pack: Path, name: str) -> ExtractionSpec:
    """Read and validate extractions/<name>.yaml in a pack folder."""
    path = pack / "extractions" / f"{name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in (pack / "extractions").glob("*.yaml"))
        raise PackError(f"No extraction {name!r} in {pack}. Known: {', '.join(known) or 'none'}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ExtractionSpec.model_validate(data)
    except yaml.YAMLError as err:
        raise PackError(f"{path} is not valid YAML:\n{err}") from err
    except ValidationError as err:
        problems = "\n".join(f"  {describe_problem(e)}" for e in err.errors())
        raise PackError(f"{path} is not a valid extraction:\n{problems}") from err


def output_model(spec: ExtractionSpec) -> type[BaseModel]:
    """The Pydantic model the answer must match. Every value may be empty, so the model can
    say "not in this document" instead of inventing something."""
    fields: dict[str, Any] = {
        "problem": (
            str | None,
            Field(description="Why the document cannot give what the task asks for, or empty."),
        )
    }
    for name, spec_field in spec.fields.items():
        if spec_field.type == "choice":
            kind: Any = Literal[tuple(spec_field.choices or [])]
        else:
            kind = PYTHON_TYPES[spec_field.type]
        need = "Required." if spec_field.required else "Empty if the document does not show it."
        fields[name] = (kind | None, Field(description=f"{spec_field.description} {need}"))
        if spec_field.quote:
            if spec_field.type == "amount":
                about = f"{name} exactly as printed, such as 1,480,312, 1 480 312 or (1 234)."
            else:
                about = f"The words in the document that {name} comes from, copied verbatim."
            fields[f"{name}_quote"] = (str | None, Field(description=about))
    return create_model("Extracted", **fields)


def extract(
    client: ModelClient, spec: ExtractionSpec, document: str, *, effort: Effort | None = "low"
) -> Extraction:
    """Extract the values a spec describes from one document, and check them."""
    schema = output_model(spec)
    request = Request(
        system=EXTRACT.format(instructions=spec.instructions.strip()),
        messages=[Message("user", f"<document>\n{document}\n</document>")],
        effort=effort,
        output_schema=schema,
        cache_system=True,  # the same instructions for every document
    )
    answer = client.send(request)
    if answer.stop_reason != "end":
        raise ExtractionError(f"the model stopped early ({answer.stop_reason})", answer)
    try:
        values = schema.model_validate_json(answer.text).model_dump()
    except ValidationError as err:
        problems = "; ".join(describe_problem(e) for e in err.errors())
        raise ExtractionError(f"the answer does not match the schema: {problems}", answer) from err
    problems = check(spec, values, document)
    if problems:
        raise ExtractionError("; ".join(problems), answer)
    return Extraction(values=values, answer=answer)


def check(spec: ExtractionSpec, values: dict[str, Any], document: str) -> list[str]:
    """Everything wrong with extracted values, as readable sentences. Empty means usable."""
    if values["problem"]:
        return [f"the model reports: {values['problem']}"]
    problems = []
    text = squeeze(document)
    for name, spec_field in spec.fields.items():
        value = values[name]
        if value is None:
            if spec_field.required:
                problems.append(f"{name} is missing")
            continue
        if not spec_field.quote:
            continue
        quote = values[f"{name}_quote"]
        if not quote:
            problems.append(f"{name} has no quote")
        elif squeeze(quote) not in text:
            problems.append(f"the quote for {name} is not in the document: {quote!r}")
        elif spec_field.type == "amount" and digits(value) != digits(quote):
            problems.append(f"{name} is {value}, but the document says {quote!r}")
    return problems


def squeeze(text: str) -> str:
    """Text with every run of whitespace, including non-breaking spaces, made one space."""
    return " ".join(text.split())


def digits(value: Any) -> str:
    """Only the digits of a number or a quote: 1 480 312 and 1480312 both give 1480312."""
    if isinstance(value, Decimal):
        value = format(value.normalize(), "f")
    return re.sub(r"\D", "", str(value))
