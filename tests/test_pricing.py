from datetime import date
from decimal import Decimal

import pytest

from vardex.pricing import (
    UnknownModelError,
    Usage,
    cost_usd,
    format_usd,
    monthly_cost_usd,
    stale_prices_warning,
)


def test_a_million_tokens_each_way_on_sonnet():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-sonnet-5-5", usage) == Decimal("12")


def test_output_tokens_cost_five_times_input_tokens():
    tokens_in = cost_usd("claude-opus-5-5", Usage(input_tokens=1000, output_tokens=0))
    tokens_out = cost_usd("claude-opus-5-5", Usage(input_tokens=0, output_tokens=1000))
    assert tokens_out == 5 * tokens_in


def test_cache_reads_are_cheaper_than_fresh_input():
    fresh = cost_usd("claude-sonnet-5-5", Usage(input_tokens=5000, output_tokens=0))
    cached = cost_usd(
        "claude-sonnet-5-5", Usage(input_tokens=0, output_tokens=0, cache_read_tokens=5000)
    )
    assert cached < fresh / 5


def test_dated_model_id_has_the_same_price_as_its_alias():
    usage = Usage(input_tokens=1234, output_tokens=567)
    assert cost_usd("claude-haiku-4-5-20251001", usage) == cost_usd("claude-haiku-4-5", usage)


def test_unknown_model_is_an_error_not_a_zero():
    with pytest.raises(UnknownModelError):
        cost_usd("gpt-imaginary", Usage(input_tokens=1, output_tokens=1))


def test_monthly_cost_of_the_analyst_example():
    usage = Usage(input_tokens=8000, output_tokens=1500)
    assert monthly_cost_usd("claude-sonnet-5-5", usage, requests_per_day=50) == Decimal("46.5")


def test_money_is_exact():
    # 0.1 + 0.2 is not 0.3 in floats. With Decimal, a thousand tiny calls add up exactly.
    one_call = cost_usd("claude-haiku-4-5", Usage(input_tokens=1, output_tokens=0))
    assert sum([one_call] * 1000) == Decimal("0.001")


def test_format_usd():
    assert format_usd(Decimal("0.0310")) == "$0.0310"
    assert format_usd(Decimal("1650")) == "$1,650.00"


def test_prices_checked_90_days_ago_are_not_stale():
    assert stale_prices_warning(date(2026, 4, 1), checked="2026-01-01") is None


def test_prices_checked_91_days_ago_are_stale_and_the_warning_says_where_to_check():
    warning = stale_prices_warning(date(2026, 4, 2), checked="2026-01-01")
    assert warning is not None
    assert "91 days ago" in warning
    assert "platform.claude.com/docs/en/about-claude/pricing" in warning
