"""What model calls cost. Prices are in US dollars per million tokens."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

MILLION = Decimal(1_000_000)
PRICES_CHECKED = "2026-09-29"
PRICES_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
STALE_AFTER_DAYS = 90
# The Message Batches API halves every token price. Checked 2026-10-01.
BATCH_DISCOUNT = Decimal("0.5")


@dataclass(frozen=True)
class Price:
    input: Decimal
    output: Decimal
    cache_write: Decimal  # writing to the 5-minute prompt cache
    cache_read: Decimal  # reading from the prompt cache


PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(Decimal("1"), Decimal("5"), Decimal("1.25"), Decimal("0.10")),
    "claude-sonnet-5-5": Price(Decimal("2"), Decimal("10"), Decimal("2.50"), Decimal("0.20")),
    "claude-opus-5-5": Price(Decimal("4"), Decimal("20"), Decimal("5"), Decimal("0.20")),
    "claude-fable-5-1": Price(Decimal("10"), Decimal("50"), Decimal("12.50"), Decimal("0.25")),
}
ALIASES = {"claude-haiku-4-5-20251001": "claude-haiku-4-5"}


@dataclass(frozen=True)
class Usage:
    """Tokens used by one call, as the API reports them."""

    input_tokens: int  # input that was not read from or written to the cache
    output_tokens: int  # the answer, including any thinking
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


class UnknownModelError(KeyError):
    """Raised when a model has no entry in the price table."""


def price_for(model: str) -> Price:
    """Look up a model's price, accepting dated model IDs as well as aliases."""
    try:
        return PRICES[ALIASES.get(model, model)]
    except KeyError as err:
        raise UnknownModelError(f"No price for {model!r}. Add it to vardex/pricing.py.") from err


def cost_usd(model: str, usage: Usage, batch: bool = False) -> Decimal:
    """The cost of one call, in US dollars. Batch calls get BATCH_DISCOUNT."""
    price = price_for(model)
    full = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_write_tokens * price.cache_write
        + usage.cache_read_tokens * price.cache_read
    ) / MILLION
    return full * (1 - BATCH_DISCOUNT) if batch else full


def monthly_cost_usd(
    model: str, usage: Usage, requests_per_day: int, days: int = 30, batch: bool = False
) -> Decimal:
    """The cost of a month of identical calls, in US dollars."""
    return cost_usd(model, usage, batch) * requests_per_day * days


def format_usd(amount: Decimal) -> str:
    """Dollars with cents, or with four decimals for amounts under one dollar."""
    return f"${amount:,.4f}" if amount < 1 else f"${amount:,.2f}"


def stale_prices_warning(today: date, checked: str = PRICES_CHECKED) -> str | None:
    """A warning if the price table was last checked more than STALE_AFTER_DAYS ago."""
    age = (today - date.fromisoformat(checked)).days
    if age <= STALE_AFTER_DAYS:
        return None
    return (
        f"Warning: prices were last checked {age} days ago ({checked}). "
        f"Check them at {PRICES_URL} and update vardex/pricing.py."
    )
