"""A log of model calls: one JSON line per call, with its time, model, tokens and cost."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vardex.llm import Answer

LOG_FILE = Path("logs/calls.jsonl")


class CallRecord(BaseModel):
    """One line of the call log. Cost is a string so no precision is lost."""

    model_config = ConfigDict(frozen=True)

    time: datetime
    model: str
    input_tokens: int
    output_tokens: int  # thinking included
    cache_write_tokens: int
    cache_read_tokens: int
    thinking_tokens: int | None
    seconds: float
    cost_usd: Decimal | None  # None when the model is not in the price table

    @classmethod
    def from_answer(cls, answer: Answer, time: datetime | None = None) -> "CallRecord":
        return cls(
            time=time or datetime.now(UTC),
            model=answer.model,
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            cache_write_tokens=answer.usage.cache_write_tokens,
            cache_read_tokens=answer.usage.cache_read_tokens,
            thinking_tokens=answer.thinking_tokens,
            seconds=answer.seconds,
            cost_usd=answer.cost_usd,
        )


def log_call(answer: Answer, path: Path = LOG_FILE) -> None:
    """Append one line for this call to the log, creating its folder if needed.

    Raises OSError when the log cannot be written.
    """
    line = CallRecord.from_answer(answer).model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(line + "\n")
