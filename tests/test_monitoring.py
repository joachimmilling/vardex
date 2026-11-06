"""Limits a pack sets on itself, and comparing two versions of it."""

from decimal import Decimal

import pytest

from vardex.dashboard import Figures, compare
from vardex.monitoring import Monitoring, breaches, load_monitoring
from vardex.packs import PackError


def row(group="answer", requests=100, failed=0, repaired=0, cost="0.70", p95=10.0, bad=0):
    return Figures(
        group, requests, failed, repaired, requests * 2, Decimal(cost), 0, 0, 0, 3.0, p95, 0, bad
    )


LIMITS = Monitoring.model_validate(
    {
        "min_requests": 20,
        "commands": {
            "answer": {
                "error_rate": 0.05,
                "repair_rate": 0.15,
                "p95_seconds": 30,
                "cost_per_request": "0.02",
                "bad_feedback": 3,
            }
        },
    }
)


def test_a_command_within_its_limits_breaks_nothing():
    assert breaches([row(), row("cite", failed=90)], LIMITS) == []  # cite has no limits


def test_each_limit_passed_is_named_with_its_number():
    found = breaches([row(failed=8, repaired=20, cost="2.50", p95=41.0, bad=4)], LIMITS)
    assert found == [
        "answer: 4 answers marked bad (limit 3)",
        "answer: failed 8% over 100 requests (limit 5%)",
        "answer: needed a second try 20% over 100 requests (limit 15%)",
        "answer: p95 41.0 s over 100 requests (limit 30.0 s)",
        "answer: cost per request $0.0250 over 100 requests (limit $0.0200)",
    ]


def test_a_rate_over_too_few_requests_is_not_judged_but_bad_feedback_is():
    found = breaches([row(requests=5, failed=3, bad=4)], LIMITS)
    assert found == ["answer: 4 answers marked bad (limit 3)"]


def test_a_misspelt_limit_is_an_error(tmp_path):
    (tmp_path / "monitoring.yaml").write_text("commands:\n  answer:\n    eror_rate: 0.1\n")
    with pytest.raises(PackError, match="eror_rate"):
        load_monitoring(tmp_path)
    assert load_monitoring(tmp_path / "elsewhere") is None


def test_a_comparison_says_only_what_the_numbers_allow():
    worse = compare(row(requests=400, repaired=20), row(requests=400, repaired=80))
    assert worse[1].endswith(": worse")
    small = compare(row(requests=20, repaired=1), row(requests=20, repaired=3))
    assert small[1].endswith("not enough requests to tell")
    assert small[0].startswith("failed: 0% (0%–16%) → 0%")
