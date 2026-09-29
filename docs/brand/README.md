# Vardex brand files

The Vardex mark is a *varde*, a cairn: four stacked stones with the top one painted trail-marker red.
The full design system (tokens, components, terminal colours, number formats) lives in the Vardex
design system; these are the files the repository itself needs.

| File | Use |
|---|---|
| `vardex-lockup.svg`, `vardex-lockup-dark.svg` | Mark and wordmark, for light and dark backgrounds |
| `vardex-mark.svg`, `vardex-mark-dark.svg` | The mark alone |
| `vardex-favicon.svg`, `vardex-favicon-dark.svg` | 16 px favicon, hinted for light and dark browser chrome |
| `vardex-app-icon.svg` | The mark on a dark tile: GitHub avatar, PyPI, app icons |
| `vardex-ascii.txt` | The CLI version of the mark, in Unicode and plain ASCII |

Colours: stones `#111315` (dark: `#e4e8e9`), capstone `#db4324` (dark: `#f5694a`).

Rules: only the top stone is red; keep clear space of one stone's height around the mark; use the
favicon drawing at 16 px rather than scaling the mark down.

The README banners in `docs/assets/` have all text converted to outlines, so they look the same on
every machine. The fonts are Schibsted Grotesk and JetBrains Mono, both under the SIL Open Font
License.
