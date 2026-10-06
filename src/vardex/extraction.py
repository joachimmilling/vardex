"""Structured extraction: a document in; checked values out, or a clear reason why not.

A pack describes what to extract in a YAML file, extractions/<name>.yaml: instructions for the
model and the fields to return. The engine turns the fields into a JSON schema, asks the model
for exactly that shape, and checks the result before anyone can use it.
"""

import json
import re
from dataclasses import dataclass, replace
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

from vardex.llm import Answer, Effort, Message, ModelClient, ModelError, Request
from vardex.packs import PackError, describe_problem
from vardex.prompts import EXAMPLE, EXAMPLES, EXTRACT, REPAIR

FieldType = Literal["text", "integer", "amount", "choice"]
# An amount is a JSON number in the schema, so the model cannot answer "1 480 312" as text,
# and an exact Decimal in Python, so no float ever touches it.
Amount = Annotated[Decimal, WithJsonSchema({"type": "number"})]
PYTHON_TYPES: dict[str, Any] = {"text": str, "integer": int, "amount": Amount}
MAX_VALUES = 16  # structured outputs accept at most 16 fields that may be empty in one schema
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_REPAIRS = 3  # each repair is another paid call


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


class ExampleSpec(BaseModel):
    """A worked example: a document and the values it should give. A field left out is empty."""

    model_config = ConfigDict(extra="forbid")

    document: str
    values: dict[str, Any] = {}


class ExtractionSpec(BaseModel):
    """The contents of extractions/<name>.yaml in a pack."""

    model_config = ConfigDict(extra="forbid")

    description: str
    instructions: str
    fields: dict[str, FieldSpec] = Field(min_length=1)
    examples: list[ExampleSpec] = []

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
    """Values that passed every check, and every answer it took to get them, in order."""

    values: dict[str, Any]
    answers: list[Answer]


class ExtractionError(Exception):
    """Raised when a document gives no usable values. The message says why."""

    def __init__(self, reason: str, answers: list[Answer] | None = None) -> None:
        super().__init__(reason)
        self.answers = answers or []  # so every call can still be logged and paid for


def load_extraction(pack: Path, name: str) -> ExtractionSpec:
    """Read and validate extractions/<name>.yaml in a pack folder."""
    path = pack / "extractions" / f"{name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in (pack / "extractions").glob("*.yaml"))
        raise PackError(f"No extraction {name!r} in {pack}. Known: {', '.join(known) or 'none'}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = ExtractionSpec.model_validate(data)
    except yaml.YAMLError as err:
        raise PackError(f"{path} is not valid YAML:\n{err}") from err
    except ValidationError as err:
        problems = "\n".join(f"  {describe_problem(e)}" for e in err.errors())
        raise PackError(f"{path} is not a valid extraction:\n{problems}") from err
    for number, example in enumerate(spec.examples, start=1):
        problems = check_example(spec, example)
        if problems:
            raise PackError(f"{path}: example {number} fails its checks: {'; '.join(problems)}")
    return spec


def example_answer(spec: ExtractionSpec, example: ExampleSpec) -> dict[str, Any]:
    """The full answer an example stands for, in the schema's order, with left-out values empty."""
    return {name: example.values.get(name) for name in output_model(spec).model_fields}


def check_example(spec: ExtractionSpec, example: ExampleSpec) -> list[str]:
    """Everything wrong with an example, checked like a real answer. Empty means usable."""
    schema = output_model(spec)
    unknown = sorted(set(example.values) - set(schema.model_fields))
    if unknown:
        return [f"{', '.join(unknown)} is not a field of this extraction"]
    try:
        values = schema.model_validate(example_answer(spec, example)).model_dump()
    except ValidationError as err:
        return [describe_problem(e) for e in err.errors()]
    if values["problem"]:
        return []  # an example of a document that cannot be used only needs the right shape
    return check(spec, values, example.document)


def system_prompt(spec: ExtractionSpec) -> str:
    """The pack's instructions and worked examples, inside the engine's extraction prompt."""
    schema = output_model(spec)
    examples = "\n".join(
        EXAMPLE.format(
            document=example.document.strip(),
            answer=json.dumps(
                schema.model_validate(example_answer(spec, example)).model_dump(),
                ensure_ascii=False,
                default=json_number,
            ),
        )
        for example in spec.examples
    )
    return EXTRACT.format(
        instructions=spec.instructions.strip(),
        examples=EXAMPLES.format(examples=examples) if examples else "",
    )


def json_number(value: Decimal) -> int | float:
    """An amount as a JSON number, the way the schema asks the model to give it."""
    return int(value) if value == value.to_integral_value() else float(value)


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


def build_request(spec: ExtractionSpec, document: str, *, effort: Effort | None = "low") -> Request:
    """The request extract sends for one document. --show-prompt prints this same request."""
    return Request(
        system=system_prompt(spec),
        messages=[Message("user", f"<document>\n{document}\n</document>")],
        effort=effort,
        output_schema=output_model(spec),
        cache_system=True,  # the same instructions for every document
    )


def extract(
    client: ModelClient,
    spec: ExtractionSpec,
    document: str,
    *,
    effort: Effort | None = "low",
    repairs: int = 1,
) -> Extraction:
    """Extract the values a spec describes from one document, and check them.

    When an answer fails the schema or the checks, the model gets its answer and the problems
    back and answers again, at most `repairs` times. A problem the model reports is final.
    """
    if not 0 <= repairs <= MAX_REPAIRS:
        raise ValueError(f"repairs must be from 0 to {MAX_REPAIRS}, not {repairs}")
    schema = output_model(spec)
    request = build_request(spec, document, effort=effort)
    answers: list[Answer] = []
    while True:
        try:
            answer = client.send(request)
        except ModelError as err:
            if not answers:
                raise
            # keep the answers already paid for, so they are still logged
            raise ExtractionError(f"the repair failed: {err}", answers) from err
        answers.append(answer)
        if answer.stop_reason != "end":
            raise ExtractionError(f"the model stopped early ({answer.stop_reason})", answers)
        try:
            values = schema.model_validate_json(answer.text).model_dump()
        except ValidationError as err:
            problems = [describe_problem(e) for e in err.errors()]
            reason = f"the answer does not match the schema: {'; '.join(problems)}"
        else:
            problems = check(spec, values, document)
            if not problems:
                return Extraction(values=values, answers=answers)
            if values["problem"]:
                raise ExtractionError("; ".join(problems), answers)
            reason = "; ".join(problems)
        if len(answers) > repairs:
            raise ExtractionError(reason, answers)
        request = repair_request(request, answer, problems)


def repair_request(request: Request, answer: Answer, problems: list[str]) -> Request:
    """The request with the model's answer and its problems added, asking it to answer again."""
    problem_list = "\n".join(f"- {problem}" for problem in problems)
    return replace(
        request,
        messages=[
            *request.messages,
            Message("assistant", answer.text),
            Message("user", REPAIR.format(problems=problem_list)),
        ],
    )


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
