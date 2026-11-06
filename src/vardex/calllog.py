"""A call log: one JSON line per model call, appended to a local file."""

import json
from datetime import UTC, datetime
from pathlib import Path

from vardex.llm import Answer, Hook, Request, Send
from vardex.tracing import current


def log_call(answer: Answer, path: Path, trace_id: str = "") -> None:
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
        "trace_id": trace_id or None,  # the request this call was part of
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def log_calls(path: Path) -> Hook:
    """A hook that logs every call that comes back, whoever made it. A log that cannot be
    written is skipped: losing a line is better than losing the answer it describes."""

    def hook(model: str, request: Request, send: Send) -> Answer:
        answer = send(request)
        try:
            log_call(answer, path, current().trace_id)
        except OSError:
            pass
        return answer

    return hook
