"""Draw the map of Vardex for the README, in a light and a dark version.

Each block carries the release that adds it. Released blocks are solid, the latest release is
trail red, and planned blocks are dashed. After a minor release, set RELEASED and run:

    uv run python docs/assets/draw_map.py
"""

from pathlib import Path

RELEASED = "0.3"

THEMES = {
    "light": {
        "bg": "#f6f5f1",
        "paper": "#fdfdfb",
        "ink": "#111315",
        "muted": "#4b545b",
        "line": "#b8bfc2",
        "accent": "#c8391a",
        "accent_bright": "#db4324",
        "accent_soft": "#fdf0eb",
    },
    "dark": {
        "bg": "#111315",
        "paper": "#181b1e",
        "ink": "#e4e8e9",
        "muted": "#a8b1b6",
        "line": "#353c42",
        "accent": "#f28e74",
        "accent_bright": "#f5694a",
        "accent_soft": "#2e1812",
    },
}

# Rows of the engine, top to bottom
LAYERS = [
    (170, "CHANNELS", "how you reach it"),
    (236, "CAPABILITIES", "what it does"),
    (302, "MODELS", "any provider"),
    (368, "DATA", "what it knows"),
    (434, "FOUNDATION", "how it ships"),
]

# (version, x, y, width, title, subtitle); width None = a column through every layer
BLOCKS = [
    ("0.1", 140, 60, 480, "Norwegian companies pack", "open data: Brønnøysund, Statistics Norway"),
    ("0.14", 630, 60, 346, "Your own pack", "vardex new pack, a separate package"),
    ("0.1", 140, 170, 160, "CLI", "ask, cost, extract"),
    ("0.12", 310, 170, 160, "Web app, API", "HTTP and a web UI"),
    ("0.8", 480, 170, 160, "MCP server", "any MCP client"),
    ("0.9", 650, 170, 160, "Approval inbox", "approve, undo"),
    ("0.5", 140, 236, 160, "Text-to-SQL", "the analyst recipe"),
    ("0.7", 310, 236, 160, "Cited answers", "from documents"),
    ("0.8", 480, 236, 160, "Agent, tools", "permission levels"),
    ("0.9", 650, 236, 160, "Workflows", "scheduled, dry-run"),
    ("0.2", 140, 302, 216, "Model client", "tokens, time, cost"),
    ("0.3", 366, 302, 217, "Prompts", "structured outputs"),
    ("0.11", 593, 302, 217, "Open models", "routing between models"),
    ("0.4", 140, 368, 330, "Warehouse", "ingestion into SQL tables"),
    ("0.7", 480, 368, 330, "Document index", "retrieval with citations"),
    ("0.1", 140, 434, 413, "Package", "CLI, tests, CI, Docker"),
    ("0.12", 563, 434, 413, "Deployment", "pip install vardex"),
    ("0.6", 822, 170, None, "Evals", "is it right?"),
    ("0.10", 876, 170, None, "Tracing", "what happened?"),
    ("0.13", 930, 170, None, "Guardrails", "security tests, audit log"),
]


def version(v: str) -> tuple[int, ...]:
    return tuple(int(part) for part in v.split("."))


def state(v: str) -> str:
    if v == RELEASED:
        return "latest"
    return "released" if version(v) < version(RELEASED) else "planned"


def badge(v: str, cx: float, cy: float) -> str:
    s = state(v)
    box = {"latest": "badge-accent", "released": "badge", "planned": "badge-planned"}[s]
    text = {"latest": "bt-accent", "released": "bt", "planned": "bt-planned"}[s]
    return (
        f'<rect class="{box}" x="{cx - 17}" y="{cy - 10}" width="34" height="20" rx="10"/>'
        f'<text class="{text}" x="{cx}" y="{cy + 4}" text-anchor="middle">{v}</text>'
    )


def block(v: str, x: int, y: int, w: int | None, title: str, sub: str) -> str:
    box = {"latest": "box-accent", "released": "box", "planned": "box-planned"}[state(v)]
    if w is None:
        cx, mid = x + 23, y + 145
        return (
            f'<rect class="{box}" x="{x}" y="{y}" width="46" height="250" rx="10"/>'
            + badge(v, cx, y + 22)
            + f'<text class="t" x="{cx - 1}" y="{mid}" text-anchor="middle" '
            f'transform="rotate(-90 {cx - 1} {mid})">{title}</text>'
            f'<text class="t-sm" x="{cx + 14}" y="{mid}" text-anchor="middle" '
            f'transform="rotate(-90 {cx + 14} {mid})">{sub}</text>'
        )
    return (
        f'<rect class="{box}" x="{x}" y="{y}" width="{w}" height="52" rx="10"/>'
        + badge(v, x + 25, y + 26)
        + f'<text class="t" x="{x + 48}" y="{y + 23}">{title}</text>'
        f'<text class="t-sm" x="{x + 48}" y="{y + 40}">{sub}</text>'
    )


def describe() -> str:
    parts = ", ".join(f"{title} ({v})" for v, _, _, _, title, _ in BLOCKS)
    return (
        "Map of Vardex and the release that adds each block. Packs on top plug into the engine, "
        "which has five layers (channels, capabilities, models, data, foundation) and three "
        "columns through every layer: evals, tracing and guardrails. Vardex 1.0 frames the whole "
        f"map. Released up to {RELEASED}; dashed blocks are planned. The blocks: {parts}."
    )


def style(c: dict[str, str]) -> str:
    return f"""
    text {{ font-family: "Schibsted Grotesk", -apple-system, "Segoe UI", Helvetica, Arial,
      sans-serif; }}
    .bg {{ fill: {c["bg"]}; }}
    .box {{ fill: {c["paper"]}; stroke: {c["line"]}; stroke-width: 1.3; }}
    .box-planned {{ fill: {c["bg"]}; stroke: {c["line"]}; stroke-width: 1.3;
      stroke-dasharray: 5 4; }}
    .box-accent {{ fill: {c["accent_soft"]}; stroke: {c["accent_bright"]}; stroke-width: 1.8; }}
    .zone {{ fill: none; stroke: {c["line"]}; stroke-width: 1.2; stroke-dasharray: 4 4; }}
    .badge {{ fill: {c["ink"]}; }}
    .badge-accent {{ fill: {c["accent_bright"]}; }}
    .badge-planned {{ fill: none; stroke: {c["muted"]}; stroke-width: 1.2; }}
    .bt, .bt-accent {{ fill: {c["bg"]}; font-size: 11px; font-weight: 700; }}
    .bt-planned {{ fill: {c["muted"]}; font-size: 11px; font-weight: 700; }}
    .t {{ fill: {c["ink"]}; font-size: 13.5px; font-weight: 600; }}
    .t-sm {{ fill: {c["muted"]}; font-size: 11.5px; font-weight: 500; }}
    .t-lbl {{ fill: {c["muted"]}; font-size: 11.5px; font-weight: 600; font-style: italic; }}
    .t-kicker {{ fill: {c["muted"]}; font-size: 11px; font-weight: 700;
      letter-spacing: 0.08em; }}
    .t-ink {{ fill: {c["ink"]}; }}
    .edge {{ stroke: {c["muted"]}; stroke-width: 1.5; fill: none; }}
    .head {{ fill: {c["muted"]}; }}
    """


def legend() -> str:
    items = [("box", "released"), ("box-accent", "latest release"), ("box-planned", "planned")]
    s, x = "", 140
    for cls, label in items:
        s += (
            f'<rect class="{cls}" x="{x}" y="522" width="22" height="14" rx="4"/>'
            f'<text class="t-sm" x="{x + 30}" y="533">{label}</text>'
        )
        x += 140
    return s


def svg(c: dict[str, str]) -> str:
    label = describe()
    s = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 552" width="1000" height="552" '
        f'role="img" aria-label="{label}"><title>{label}</title>'
        f"<style>{style(c)}</style>"
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto"><path class="head" d="M0,0 L8,4 L0,8 z"/></marker></defs>'
        '<rect class="bg" width="1000" height="552" rx="16"/>'
    )
    # Vardex 1.0: the frame around everything
    s += (
        '<rect class="zone" x="4" y="4" width="992" height="504" rx="16"/>'
        + badge("1.0", 38, 28)
        + '<text x="64" y="33"><tspan class="t">Vardex 1.0</tspan>'
        '<tspan class="t-sm" dx="10">a stable pack format and complete documentation</tspan>'
        "</text>"
    )
    s += (
        '<text class="t-kicker" x="16" y="83">PACKS</text>'
        '<text class="t-sm" x="16" y="100">your domain</text>'
        '<path class="edge" d="M380 112 L380 136" marker-end="url(#arrow)"/>'
        '<text class="t-lbl" x="391" y="130">plugs in</text>'
        '<rect class="zone" x="132" y="140" width="854" height="358" rx="14"/>'
        '<text class="t-kicker t-ink" x="976" y="158" text-anchor="end">VARDEX ENGINE</text>'
    )
    for y, name, sub in LAYERS:
        s += (
            f'<text class="t-kicker" x="16" y="{y + 23}">{name}</text>'
            f'<text class="t-sm" x="16" y="{y + 40}">{sub}</text>'
        )
    s += "".join(block(*b) for b in BLOCKS)
    return s + legend() + "</svg>\n"


def main() -> None:
    here = Path(__file__).parent
    for name, colours in THEMES.items():
        (here / f"vardex-map-{name}.svg").write_text(svg(colours), encoding="utf-8")


if __name__ == "__main__":
    main()
