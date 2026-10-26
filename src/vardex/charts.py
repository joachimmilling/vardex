"""Charts, as SVG files a person can open.

A model answering "revenue for the five largest seafood companies over three years" has the
numbers and cannot draw them. This writes the picture instead, by hand, with no plotting
library: a chart is a few hundred bytes of XML, and a dependency that draws one brings fonts,
rasterisers and a release cycle with it.

The file is written for somebody to open on its own, so it carries its own stylesheet. Colours
come from custom properties with a dark-mode fallback rather than being written into each
shape, so one chart looks right on a white page and in a dark editor.

Nothing here knows about a pack, a warehouse or a model: it takes labels and numbers and
returns a path. `make_chart` wraps it as a tool.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

MAX_POINTS = 60  # labels × series; beyond this the bars are thinner than their gaps
MAX_SERIES = 4  # the palette has four, and a legend longer than that is a table

WIDTH = 740
HEIGHT = 420
PAD_LEFT = 72
PAD_RIGHT = 24
PAD_TOP = 56
PAD_BOTTOM = 72

Kind = str  # "bar" or "line"

STYLE = """
  svg { --ink: #1c1917; --soft: #78716c; --line: #d6d3d1; --page: #ffffff;
        --s1: #1c1917; --s2: #78716c; --s3: #a8a29e; --s4: #be3a2b; }
  @media (prefers-color-scheme: dark) {
    svg { --ink: #f5f5f4; --soft: #a8a29e; --line: #44403c; --page: #1c1917;
          --s1: #f5f5f4; --s2: #a8a29e; --s3: #78716c; --s4: #e4604f; }
  }
  .page { fill: var(--page); }
  .axis { stroke: var(--line); stroke-width: 1; fill: none; }
  .grid { stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 4; }
  .t { fill: var(--ink); font: 500 13px 'Schibsted Grotesk', system-ui, sans-serif; }
  .t-head { fill: var(--ink); font: 600 17px 'Schibsted Grotesk', system-ui, sans-serif; }
  .t-sm { fill: var(--soft); font: 400 11px 'Schibsted Grotesk', system-ui, sans-serif; }
  .s1 { fill: var(--s1); stroke: var(--s1); }
  .s2 { fill: var(--s2); stroke: var(--s2); }
  .s3 { fill: var(--s3); stroke: var(--s3); }
  .s4 { fill: var(--s4); stroke: var(--s4); }
  .mark { stroke: none; }
  .path { fill: none; stroke-width: 2.5; }
"""


class ChartError(ValueError):
    """The chart cannot be drawn as asked: too many points, or series of unequal length."""


@dataclass(frozen=True)
class Series:
    """One named line or set of bars."""

    name: str
    values: list[float]


@dataclass(frozen=True)
class Chart:
    """A chart that has been written: where it went, and how to describe it."""

    path: Path
    aria_label: str
    summary: str


# Norwegian letters have no ASCII decomposition, so stripping accents deletes them outright
# and "Brønnøysund" becomes "brnnysund". Spell them out before anything else touches the text.
NORWEGIAN = {"æ": "ae", "ø": "o", "å": "a", "Æ": "ae", "Ø": "o", "Å": "a"}


def slug(text: str) -> str:
    """A filename-safe version of a title, so a chart can be found by eye in the folder."""
    spelt = "".join(NORWEGIAN.get(char, char) for char in text)
    plain = unicodedata.normalize("NFKD", spelt).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", plain.lower())).strip("-") or "chart"


def _nice(value: float) -> str:
    """A number as a person would write it: no trailing zeros, thousands separated."""
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ")


def _axis_max(values: list[float]) -> float:
    top = max(values + [0.0])
    if top <= 0:
        return 1.0
    step = 10 ** (len(str(int(top))) - 1)
    return float(((int(top) // step) + 1) * step)


def describe(title: str, kind: Kind, labels: list[str], series: list[Series]) -> str:
    """What the chart says, for somebody who cannot see it. This is the aria-label, and it is
    the only part of a chart that a screen reader, a search index or a model ever gets."""
    shape = "A bar chart" if kind == "bar" else "A line chart"
    parts = [f"{shape} titled {title}, over {labels[0]} to {labels[-1]}"]
    for one in series:
        first, last = one.values[0], one.values[-1]
        direction = "rising" if last > first else "falling" if last < first else "flat"
        parts.append(
            f"{one.name} {direction} from {_nice(first)} at {labels[0]} to "
            f"{_nice(last)} at {labels[-1]}, highest {_nice(max(one.values))}"
        )
    return ". ".join(parts) + "."


def render(title: str, kind: Kind, labels: list[str], series: list[Series]) -> str:
    """The chart as SVG text."""
    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    top = _axis_max([v for one in series for v in one.values])
    label = escape(describe(title, kind, labels, series))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{label}">',
        f"<style>{STYLE}</style>",
        f'<rect class="page" x="0" y="0" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="t-head" x="{PAD_LEFT}" y="30">{escape(title)}</text>',
    ]

    # The value axis: four gridlines, each labelled, so a reader can judge a bar without a ruler.
    for n in range(5):
        value = top * n / 4
        y = PAD_TOP + plot_h - plot_h * n / 4
        right = WIDTH - PAD_RIGHT
        out.append(f'<line class="grid" x1="{PAD_LEFT}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="t-sm" x="{PAD_LEFT - 10}" y="{y + 4:.1f}" text-anchor="end">'
            f"{escape(_nice(value))}</text>"
        )
    out.append(
        f'<line class="axis" x1="{PAD_LEFT}" y1="{PAD_TOP + plot_h}" '
        f'x2="{WIDTH - PAD_RIGHT}" y2="{PAD_TOP + plot_h}"/>'
    )

    slots = len(labels)
    slot_w = plot_w / slots
    for i, text in enumerate(labels):
        x = PAD_LEFT + slot_w * (i + 0.5)
        out.append(
            f'<text class="t-sm" x="{x:.1f}" y="{PAD_TOP + plot_h + 20}" '
            f'text-anchor="middle">{escape(text)}</text>'
        )

    if kind == "bar":
        group_w = slot_w * 0.72
        bar_w = group_w / len(series)
        for s, one in enumerate(series):
            for i, value in enumerate(one.values):
                height = plot_h * (value / top) if top else 0
                x = PAD_LEFT + slot_w * (i + 0.5) - group_w / 2 + bar_w * s
                y = PAD_TOP + plot_h - height
                out.append(
                    f'<rect class="mark s{s + 1}" x="{x:.1f}" y="{y:.1f}" '
                    f'width="{bar_w - 2:.1f}" height="{height:.1f}"/>'
                )
                # The value in text, but only where there is room for it to be read.
                if bar_w > 34 and height > 18:
                    out.append(
                        f'<text class="t-sm" x="{x + (bar_w - 2) / 2:.1f}" y="{y - 5:.1f}" '
                        f'text-anchor="middle">{escape(_nice(value))}</text>'
                    )
    else:
        for s, one in enumerate(series):
            points = [
                (
                    PAD_LEFT + slot_w * (i + 0.5),
                    PAD_TOP + plot_h - (plot_h * (value / top) if top else 0),
                )
                for i, value in enumerate(one.values)
            ]
            steps = (f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
            path = " ".join(steps)
            out.append(f'<path class="path s{s + 1}" d="{path}"/>')
            for (x, y), value in zip(points, one.values, strict=True):
                out.append(f'<circle class="mark s{s + 1}" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>')
                if slots <= 8:
                    out.append(
                        f'<text class="t-sm" x="{x:.1f}" y="{y - 10:.1f}" '
                        f'text-anchor="middle">{escape(_nice(value))}</text>'
                    )

    if len(series) > 1 or series[0].name:
        x = PAD_LEFT
        y = HEIGHT - 22
        for s, one in enumerate(series):
            out.append(f'<rect class="mark s{s + 1}" x="{x}" y="{y - 9}" width="11" height="11"/>')
            out.append(f'<text class="t" x="{x + 17}" y="{y}">{escape(one.name)}</text>')
            x += 28 + 8 * len(one.name)

    out.append("</svg>")
    return "\n".join(out)


def charts_path(data_dir: Path, pack: str) -> Path:
    """Where this pack's charts are kept."""
    return data_dir / pack / "charts"


def write_chart(
    folder: Path, title: str, kind: Kind, labels: list[str], series: list[Series]
) -> Chart:
    """Draw the chart and put it in `folder`, stamped so two runs never collide."""
    if kind not in ("bar", "line"):
        raise ChartError(f"kind must be 'bar' or 'line', not {kind!r}.")
    if not labels:
        raise ChartError("A chart needs at least one label.")
    if not series:
        raise ChartError("A chart needs at least one series.")
    if len(series) > MAX_SERIES:
        raise ChartError(
            f"{len(series)} series is more than this chart can show. "
            f"Draw at most {MAX_SERIES}, or make more than one chart."
        )
    points = len(labels) * len(series)
    if points > MAX_POINTS:
        raise ChartError(
            f"{len(labels)} labels × {len(series)} series is {points} points, more than the "
            f"{MAX_POINTS} this chart can show. Group the data or chart fewer of them."
        )
    for one in series:
        if len(one.values) != len(labels):
            raise ChartError(
                f"Series {one.name!r} has {len(one.values)} values but there are "
                f"{len(labels)} labels. Every series needs one value per label."
            )

    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The stamp is only to the second, and one run can draw the same title twice in less than
    # that. Never overwrite a chart somebody may already have been shown.
    path = folder / f"{slug(title)}-{stamp}.svg"
    nth = 2
    while path.exists():
        path = folder / f"{slug(title)}-{stamp}-{nth}.svg"
        nth += 1
    path.write_text(render(title, kind, labels, series), encoding="utf-8")
    return Chart(
        path=path,
        aria_label=describe(title, kind, labels, series),
        summary=summary(labels, series),
    )


def summary(labels: list[str], series: list[Series]) -> str:
    """One line about the data, for the model. Not the data itself: it already has that, and
    sending it back doubles the tokens for nothing."""
    names = ", ".join(one.name for one in series)
    return f"{len(series)} series ({names}) over {len(labels)} points, {labels[0]} to {labels[-1]}."
