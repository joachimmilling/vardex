"""What a pack promises about itself in production, and a check of its traces against it.

A dashboard is read by somebody who thinks to look. A limit is read by a scheduler every
hour, which never forgets, and exits 1 when the pack breaks a promise: too many failures,
too many answers that needed a second try, too slow, too dear, or too many people saying an
answer was wrong.

A rate over five requests says nothing, so no rate or time is judged below `min_requests`.
"""

from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vardex.dashboard import Figures
from vardex.packs import read_yaml


class Limits(BaseModel):
    """The most one command may reach before somebody should hear about it."""

    model_config = ConfigDict(extra="forbid")

    error_rate: float | None = Field(default=None, ge=0, le=1)
    repair_rate: float | None = Field(default=None, ge=0, le=1)
    p95_seconds: float | None = Field(default=None, gt=0)
    cost_per_request: Decimal | None = Field(default=None, gt=0)
    bad_feedback: int | None = Field(default=None, ge=0)  # answers marked bad in the window


class Monitoring(BaseModel):
    """The contents of a pack's monitoring.yaml."""

    model_config = ConfigDict(extra="forbid")

    window_days: float = Field(default=7, gt=0)
    min_requests: int = Field(default=20, ge=1)
    commands: dict[str, Limits] = Field(min_length=1)


def monitoring_path(pack: Path) -> Path:
    return pack / "monitoring.yaml"


def load_monitoring(pack: Path) -> Monitoring | None:
    """The pack's limits, or None when it sets none."""
    path = monitoring_path(pack)
    return read_yaml(path, Monitoring, "monitoring file") if path.is_file() else None


def breaches(rows: list[Figures], monitoring: Monitoring) -> list[str]:
    """Every limit a command has gone past, as a sentence a person can act on."""
    found = []
    for row in rows:
        limits = monitoring.commands.get(row.group)
        if limits is None:
            continue
        if limits.bad_feedback is not None and row.bad > limits.bad_feedback:
            found.append(f"{row.group}: {row.bad} answers marked bad (limit {limits.bad_feedback})")
        if row.requests < monitoring.min_requests:
            continue  # too few to judge a rate or a time
        checks = [
            ("failed", row.error_rate, limits.error_rate, "{:.0%}"),
            ("needed a second try", row.repair_rate, limits.repair_rate, "{:.0%}"),
            ("p95", row.p95_seconds, limits.p95_seconds, "{:.1f} s"),
            ("cost per request", row.cost_per_request, limits.cost_per_request, "${:.4f}"),
        ]
        for what, value, limit, shape in checks:
            if limit is not None and value > limit:
                found.append(
                    f"{row.group}: {what} {shape.format(value)} over {row.requests} requests "
                    f"(limit {shape.format(limit)})"
                )
    return found
