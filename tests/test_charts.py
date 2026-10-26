"""Charts: what is drawn, what is refused, and what the model is told about it."""

import re
from pathlib import Path

import pytest

from vardex.charts import (
    MAX_POINTS,
    ChartError,
    Series,
    describe,
    render,
    slug,
    write_chart,
)
from vardex.llm import ToolCall
from vardex.permissions import Permissions
from vardex.tools import build_toolbox

YEARS = ["2022", "2023", "2024"]
REVENUE = [Series("Demo Sjømat AS", [120.0, 150.0, 210.0])]


def test_a_chart_is_written_and_named_after_its_title(tmp_path: Path):
    chart = write_chart(tmp_path, "Revenue by year", "bar", YEARS, REVENUE)
    assert chart.path.parent == tmp_path
    assert chart.path.name.startswith("revenue-by-year-")
    assert chart.path.suffix == ".svg"
    assert chart.path.read_text(encoding="utf-8").startswith("<svg")


def test_two_charts_with_one_title_do_not_overwrite_each_other(tmp_path: Path):
    first = write_chart(tmp_path, "Revenue", "bar", YEARS, REVENUE)
    second = write_chart(tmp_path, "Revenue", "line", YEARS, REVENUE)
    assert first.path != second.path
    assert len(list(tmp_path.glob("*.svg"))) == 2


def test_the_svg_hard_codes_no_colours():
    """Colours come from custom properties with a dark-mode fallback, so one file reads on a
    white page and in a dark editor. A fill written into a shape would only suit one of them."""
    svg = render("Revenue", "bar", YEARS, REVENUE)
    starts = ("<rect", "<path", "<circle", "<line", "<text")
    shapes = [line for line in svg.splitlines() if line.lstrip().startswith(starts)]
    for shape in shapes:
        assert not re.search(r'(fill|stroke)="#', shape), shape
    assert "prefers-color-scheme: dark" in svg


def test_the_svg_says_what_it_shows():
    svg = render("Revenue by year", "line", YEARS, REVENUE)
    assert 'role="img"' in svg
    assert 'aria-label="A line chart titled Revenue by year' in svg
    for year in YEARS:
        assert f">{year}<" in svg


def test_the_description_names_the_direction_and_the_ends():
    label = describe("Revenue by year", "bar", YEARS, REVENUE)
    assert "rising from 120 at 2022 to 210 at 2024" in label
    assert "highest 210" in label

    falling = [Series("Demo Nedtur AS", [90.0, 60.0, 30.0])]
    assert "falling from 90" in describe("Revenue", "bar", YEARS, falling)


def test_too_many_series_is_refused_with_a_reason(tmp_path: Path):
    five = [Series(f"Demo {n} AS", [1.0, 2.0, 3.0]) for n in range(5)]
    with pytest.raises(ChartError, match="5 series is more than"):
        write_chart(tmp_path, "Too many", "bar", YEARS, five)


def test_too_many_points_is_refused_with_a_reason(tmp_path: Path):
    labels = [str(n) for n in range(MAX_POINTS + 1)]
    series = [Series("Demo AS", [float(n) for n in range(MAX_POINTS + 1)])]
    with pytest.raises(ChartError, match="more than the 60"):
        write_chart(tmp_path, "Too long", "line", labels, series)


def test_a_series_of_the_wrong_length_is_refused(tmp_path: Path):
    with pytest.raises(ChartError, match="2 values but there are 3 labels"):
        write_chart(tmp_path, "Short", "bar", YEARS, [Series("Demo AS", [1.0, 2.0])])


def test_an_unknown_kind_is_refused(tmp_path: Path):
    with pytest.raises(ChartError, match="bar.*line"):
        write_chart(tmp_path, "Pie", "pie", YEARS, REVENUE)


def test_slug_keeps_norwegian_titles_usable():
    assert slug("Omsetning i Brønnøysund, 2024") == "omsetning-i-bronnoysund-2024"
    assert slug("!!!") == "chart"


def chart_call(**arguments) -> ToolCall:
    return ToolCall(id="c1", name="make_chart", arguments=arguments)


def chart_toolbox(folder: Path, **extra):
    return build_toolbox(
        Permissions(tools={"make_chart": "read", "describe_tool": "read"}),
        charts=folder,
        **extra,
    )


def test_the_tool_returns_the_path_and_the_description_but_not_the_numbers(tmp_path: Path):
    box = chart_toolbox(tmp_path)
    result = box.call(
        chart_call(
            title="Revenue by year",
            kind="bar",
            labels=YEARS,
            series=[{"name": "Demo Sjømat AS", "values": [120, 150, 210]}],
        )
    )
    assert not result.is_error
    assert result.structured is not None
    assert Path(result.structured["path"]).is_file()
    assert "A bar chart titled Revenue by year" in result.content
    # The model already has the numbers; sending them back doubles the tokens for nothing.
    assert "120, 150, 210" not in result.content


def test_the_tool_turns_a_refusal_into_something_the_model_can_read(tmp_path: Path):
    box = chart_toolbox(tmp_path)
    result = box.call(
        chart_call(
            title="Too many",
            kind="bar",
            labels=YEARS,
            series=[{"name": f"Demo {n} AS", "values": [1, 2, 3]} for n in range(5)],
        )
    )
    assert result.is_error and "5 series is more than" in result.content


def test_a_dry_run_still_writes_the_chart(tmp_path: Path):
    """A chart nobody can open is not a dry run of anything, and it changes nothing outside
    Vardex: the file lands in the pack's own data folder."""
    box = chart_toolbox(tmp_path, dry_run=True)
    result = box.call(
        chart_call(
            title="Revenue",
            kind="line",
            labels=YEARS,
            series=[{"name": "Demo Sjømat AS", "values": [1, 2, 3]}],
        )
    )
    assert not result.is_error
    assert list(tmp_path.glob("*.svg"))


def test_a_pack_that_connects_no_charts_folder_has_no_chart_tool():
    box = build_toolbox(Permissions(tools={"describe_tool": "read"}))
    assert "make_chart" not in box.tools
