"""A call log: one JSON line per model call, appended to a local file."""

import json
from datetime import UTC, datetime
from pathlib import Path

from vardex.llm import Answer


def log_call(answer: Answer, path: Path) -> None:
    """Append one line describing a call to a JSON Lines file, creating it if needed."""
    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": answer.model,
        "input_tokens": answer.usage.input_tokens,
        "output_tokens": answer.usage.output_tokens,
        "thinking_tokens": answer.thinking_tokens,
        "cache_read_tokens": answer.usage.cache_read_tokens,
        "seconds": round(answer.seconds, 3),
        "cost_usd": str(answer.cost_usd) if answer.cost_usd is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
