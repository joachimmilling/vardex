"""Hard spending caps: refused before the call, and counted after it."""

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from fakes import FakeModel

from vardex.budgets import Budget, BudgetError, Spend, estimate_tokens, worst_case_usd
from vardex.llm import Image, Message, Request, ToolSpec
from vardex.permissions import Budgets


def request(text: str = "What was revenue?") -> Request:
    return Request(system="You answer questions.", messages=[Message("user", text)])


def test_a_call_is_refused_before_it_is_sent():
    model = FakeModel("never sent")
    budget = Budget(model, caps=Budgets(per_run=Decimal("0.000001")))
    with pytest.raises(BudgetError) as refused:
        budget.send(request())
    assert "budget for this question is spent" in str(refused.value)
    assert model.requests == []  # nothing left the machine, so nothing was charged


def test_without_a_cap_nothing_is_refused():
    budget = Budget(FakeModel("fine"), caps=Budgets())
    assert budget.send(request()).text == "fine"
    assert budget.remaining() is None


def test_what_a_call_costs_is_added_to_the_run():
    budget = Budget(FakeModel("one", "two"), caps=Budgets(per_run=Decimal("1")))
    budget.send(request())
    budget.send(request())
    assert budget.run_usd == Decimal("0.008")
    assert budget.remaining() == Decimal("0.992")


def test_the_cap_holds_however_many_calls_are_made():
    model = FakeModel(*["reply"] * 100)
    budget = Budget(model, caps=Budgets(per_run=Decimal("0.05")))
    with pytest.raises(BudgetError):
        for _ in range(100):
            budget.send(request())
    assert budget.run_usd <= Decimal("0.05")


def test_the_worst_case_is_priced_not_the_average():
    # 1,000 input tokens at $2 and 4,096 output at $10, per million, on Sonnet 5.5
    worst = worst_case_usd("claude-sonnet-5-5", 1000, 4096)
    assert worst == Decimal("0.04296")
    assert worst_case_usd("a-model-nobody-priced", 1000, 4096) is None


def test_a_model_with_no_price_is_counted_as_unpriced_rather_than_free():
    class Unpriced(FakeModel):
        def send(self, request, on_text=None):
            answer = super().send(request, on_text)
            return type(answer)(**{**answer.__dict__, "cost_usd": None})

    budget = Budget(Unpriced("reply"), caps=Budgets(per_run=Decimal("1")))
    budget.send(request())
    assert budget.unpriced == 1
    assert budget.run_usd == Decimal(0)


def test_the_estimate_counts_the_tools_the_replies_and_the_pictures():
    plain = estimate_tokens(request("a" * 100))
    with_tool = estimate_tokens(
        Request(
            system="You answer questions.",
            messages=[Message("user", "a" * 100)],
            tools=(ToolSpec("look_up", "b" * 100, {}),),
        )
    )
    assert with_tool > plain
    with_picture = estimate_tokens(
        Request(
            system="s",
            messages=[Message("user", "x", images=(Image(data=b"png"),))],
        )
    )
    assert with_picture >= 1600  # a page of pictures is not free


def test_the_estimate_guesses_high():
    # Two characters to a token is well under the truth for prose, which is the safe side
    text = "The revenue of the group rose by twelve per cent in 2025." * 10
    assert estimate_tokens(request(text)) > len(text) / 4


# Spending across commands


def test_the_day_is_kept_in_a_file_so_a_cap_holds_across_commands(tmp_path):
    path = tmp_path / "spend.json"
    first = Budget(FakeModel("one"), caps=Budgets(per_day=Decimal("1")), spend=Spend(path))
    first.send(request())
    assert first.day_usd == Decimal("0.004")
    # A second command, a new process as far as this is concerned, sees what the first spent
    second = Budget(FakeModel("two"), caps=Budgets(per_day=Decimal("1")), spend=Spend(path))
    assert second.day_usd == Decimal("0.004")
    second.send(request())
    assert second.day_usd == Decimal("0.008")


def test_a_day_that_is_already_spent_refuses_the_first_call(tmp_path):
    path = tmp_path / "spend.json"
    Spend(path).add(Decimal("19.999"))
    budget = Budget(FakeModel("never"), caps=Budgets(per_day=Decimal("20")), spend=Spend(path))
    with pytest.raises(BudgetError) as refused:
        budget.send(request())
    assert "budget for today is spent" in str(refused.value)


def test_only_todays_spending_counts(tmp_path):
    path = tmp_path / "spend.json"
    Spend(path, today=date(2026, 1, 1)).add(Decimal("5"))
    assert Spend(path, today=date(2026, 1, 2)).read() == Decimal(0)


def test_an_unreadable_record_counts_as_nothing_spent(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("this is not json", encoding="utf-8")
    assert Spend(path).read() == Decimal(0)  # it fails towards answering, not towards silence


def test_the_record_keeps_thirty_days(tmp_path):
    path = tmp_path / "spend.json"
    for day in range(40):
        Spend(path, today=date(2026, 1, 1) + timedelta(days=day)).add(Decimal("1"))
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 30
