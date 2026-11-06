"""A pack's figures from its traces: how many requests, how many failed, what they cost, how
long they took, and what people thought of the answers.

Every number here is added up from spans, so it says what the system did rather than what it
was meant to do, and every number can be followed back to the requests it came from. Nothing
is kept beside the traces: delete a day's traces, and that day is gone from the figures too.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from html import escape
from pathlib import Path

from vardex.charts import Series, render
from vardex.evals import wilson
from vardex.pricing import format_usd
from vardex.traces import Trace

Grouping = Callable[[Trace], str]

GROUPINGS: dict[str, Grouping] = {
    "command": lambda trace: trace.command,
    "model": lambda trace: trace.model or "no model",
    "version": lambda trace: trace.pack_version or "unknown",
    "day": lambda trace: trace.start.date().isoformat(),
}


@dataclass(frozen=True)
class Figures:
    """One group's figures: a command, a model, a pack version or a day."""

    group: str
    requests: int
    failed: int
    repaired: int  # needed a second attempt at some step to pass its checks
    calls: int
    cost_usd: Decimal
    unpriced: int  # model calls with no price, left out of the cost
    input_tokens: int
    output_tokens: int
    p50_seconds: float
    p95_seconds: float
    good: int  # answers people marked good
    bad: int  # and bad

    @property
    def error_rate(self) -> float:
        return self.failed / self.requests if self.requests else 0.0

    @property
    def repair_rate(self) -> float:
        return self.repaired / self.requests if self.requests else 0.0

    @property
    def cost_per_request(self) -> Decimal:
        return self.cost_usd / self.requests if self.requests else Decimal(0)


def percentile(values: list[float], share: float) -> float:
    """The value below which `share` of the values lie, by the nearest rank: a value that
    really occurred, never one made up between two. With few values, p95 is the worst one,
    which is the honest answer."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(share * len(ordered)))
    return ordered[rank - 1]


def figures(
    traces: list[Trace], by: Grouping, verdicts: dict[str, str] | None = None
) -> list[Figures]:
    """The figures for each group of traces, the busiest group first. `verdicts` holds what
    people said about each trace, by trace id: good or bad."""
    verdicts = verdicts or {}
    groups: dict[str, list[Trace]] = {}
    for trace in traces:
        groups.setdefault(by(trace), []).append(trace)
    rows = []
    for group, members in groups.items():
        seconds = [t.seconds for t in members]
        said = [verdicts.get(t.trace_id) for t in members]
        rows.append(
            Figures(
                group=group,
                requests=len(members),
                failed=sum(t.failed for t in members),
                repaired=sum(t.repaired for t in members),
                calls=sum(len(t.calls) for t in members),
                cost_usd=sum((t.cost_usd for t in members), Decimal(0)),
                unpriced=sum(t.unpriced for t in members),
                input_tokens=sum(t.input_tokens for t in members),
                output_tokens=sum(t.output_tokens for t in members),
                p50_seconds=percentile(seconds, 0.5),
                p95_seconds=percentile(seconds, 0.95),
                good=said.count("good"),
                bad=said.count("bad"),
            )
        )
    return sorted(rows, key=lambda row: (-row.requests, row.group))


COLUMNS = ("", "requests", "failed", "repaired", "cost", "per request", "p50", "p95", "good", "bad")


def cells(row: Figures) -> list[str]:
    return [
        row.group,
        f"{row.requests:,}",
        f"{row.failed} ({row.error_rate:.0%})",
        f"{row.repaired} ({row.repair_rate:.0%})",
        format_usd(row.cost_usd) + ("*" if row.unpriced else ""),
        format_usd(row.cost_per_request),
        f"{row.p50_seconds:.1f} s",
        f"{row.p95_seconds:.1f} s",
        str(row.good),
        str(row.bad),
    ]


def format_figures(rows: list[Figures]) -> str:
    """The figures as a table for a terminal."""
    table = [list(COLUMNS), *(cells(row) for row in rows)]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    lines = [
        "  ".join(
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(line, widths, strict=True))
        )
        for line in table
    ]
    if any(row.unpriced for row in rows):
        lines.append("* some calls used a model with no price, and are not in the cost")
    return "\n".join(lines)


def by_day(traces: list[Trace], first: date, last: date) -> tuple[list[str], list[Figures]]:
    """One row per day from `first` to `last`, a day with no requests included as zeros, so
    a chart of them shows the quiet days rather than skipping them."""
    rows = {row.group: row for row in figures(traces, GROUPINGS["day"])}
    labels, days = [], []
    day = first
    while day <= last:
        labels.append(day.strftime("%d %b"))
        empty = Figures(day.isoformat(), 0, 0, 0, 0, Decimal(0), 0, 0, 0, 0.0, 0.0, 0, 0)
        days.append(rows.get(day.isoformat(), empty))
        day = date.fromordinal(day.toordinal() + 1)
    return labels, days


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --ink: #1c1917; --soft: #78716c; --line: #d6d3d1; --page: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #f5f5f4; --soft: #a8a29e; --line: #44403c; --page: #1c1917; }}
  }}
  body {{ background: var(--page); color: var(--ink); margin: 0 auto; max-width: 780px;
         padding: 24px 16px; font: 15px/1.5 'Schibsted Grotesk', system-ui, sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  p.meta {{ color: var(--soft); margin: 0 0 24px; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 28px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: right;
            white-space: nowrap; font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--soft); font-weight: 500; }}
  svg {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
{body}
</body>
</html>
"""


def html_table(heading: str, rows: list[Figures]) -> str:
    head = "".join(f"<th>{escape(c)}</th>" for c in COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells(row)) + "</tr>"
        for row in rows
    )
    return (
        f"<h2>{escape(heading)}</h2><div class='scroll'><table><tr>{head}</tr>{body}</table></div>"
    )


def write_dashboard(
    path: Path,
    title: str,
    traces: list[Trace],
    first: date,
    last: date,
    verdicts: dict[str, str] | None = None,
) -> Path:
    """A page a person can open: the figures by command and by pack version, and charts of
    cost and of the slowest requests, day by day."""
    labels, days = by_day(traces, first, last)
    charts = ""
    if len(labels) <= 60:  # a chart of more days than that is a smear; the tables still hold
        cost = Series("Cost (USD)", [float(row.cost_usd) for row in days])
        slow = Series("p95 seconds", [row.p95_seconds for row in days])
        charts = render("Cost per day", "bar", labels, [cost]) + render(
            "Slowest requests (p95) per day", "line", labels, [slow]
        )
    body = "".join(
        [
            html_table("By command", figures(traces, GROUPINGS["command"], verdicts)),
            html_table("By pack version", figures(traces, GROUPINGS["version"], verdicts)),
            charts,
        ]
    )
    meta = f"{len(traces):,} requests, {first.isoformat()} to {last.isoformat()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        PAGE.format(title=escape(title), meta=escape(meta), body=body), encoding="utf-8"
    )
    return path


def compare(old: Figures, new: Figures) -> list[str]:
    """What changed between two groups, such as two versions of a pack, said only as strongly
    as the numbers allow. A rate is given with its 95% interval; when the intervals of the two
    overlap, the honest verdict is that there are not yet enough requests to tell."""
    lines = []
    for what, of in (("failed", lambda r: r.failed), ("needed a second try", lambda r: r.repaired)):
        low_old, high_old = wilson(of(old), old.requests)
        low_new, high_new = wilson(of(new), new.requests)
        share_old = of(old) / old.requests if old.requests else 0.0
        share_new = of(new) / new.requests if new.requests else 0.0
        if high_new < low_old:
            verdict = "better"
        elif low_new > high_old:
            verdict = "worse"
        else:
            verdict = "not enough requests to tell"
        lines.append(
            f"{what}: {share_old:.0%} ({low_old:.0%}–{high_old:.0%}) → {share_new:.0%} "
            f"({low_new:.0%}–{high_new:.0%}): {verdict}"
        )
    lines.append(
        f"cost per request: {format_usd(old.cost_per_request)} → {format_usd(new.cost_per_request)}"
    )
    lines.append(f"p95: {old.p95_seconds:.1f} s → {new.p95_seconds:.1f} s")
    return lines
