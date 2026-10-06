from dataclasses import replace
from decimal import Decimal

import pytest
from fakes import FakeModel

from vardex.llm import Conversation, Message, ModelError, ask
from vardex.prompts import ASK


def test_ask_sends_the_question_with_the_engine_prompt():
    model = FakeModel("A nine-digit number.")

    answer = ask(model, "What is an organisation number?")

    assert answer.text == "A nine-digit number."
    request = model.requests[0]
    assert request.system == ASK
    assert request.messages == [Message("user", "What is an organisation number?")]
    assert request.effort is None
    assert request.output_schema is None


def test_ask_passes_on_effort_and_streaming():
    model = FakeModel("Nine digits.")
    pieces = []

    answer = ask(model, "q", effort="low", on_text=pieces.append)

    assert model.requests[0].effort == "low"
    assert pieces == ["Nine digits."]
    assert answer.first_text_seconds == 0.5


def test_a_conversation_sends_every_earlier_turn_and_caches_it():
    model = FakeModel("Nine digits.", "Yes, a checksum.")
    conversation = Conversation(model, effort="low")

    conversation.send("What is an organisation number?")
    conversation.send("Does it have a check digit?")

    request = model.requests[1]
    assert request.messages == [
        Message("user", "What is an organisation number?"),
        Message("assistant", "Nine digits."),
        Message("user", "Does it have a check digit?"),
    ]
    assert request.cache_conversation
    assert request.effort == "low"
    assert request.system == ASK
    assert len(model.requests[0].messages) == 1  # an earlier request is never changed
    assert len(conversation.answers) == 2
    assert conversation.cost_usd == Decimal("0.008")


def test_a_failed_call_leaves_the_conversation_unchanged():
    model = FakeModel("Nine digits.", ModelError("Could not reach the API"), "Yes.")
    conversation = Conversation(model)
    conversation.send("What is an organisation number?")

    with pytest.raises(ModelError):
        conversation.send("Lost?")

    assert conversation.messages == [
        Message("user", "What is an organisation number?"),
        Message("assistant", "Nine digits."),
    ]
    assert len(conversation.answers) == 1
    conversation.send("Check digit?")
    assert [m.text for m in model.requests[2].messages][-1] == "Check digit?"
    assert len(model.requests[2].messages) == 3


def test_the_cost_of_a_conversation_is_unknown_when_one_turn_is():
    model = FakeModel("One.", "Two.")
    conversation = Conversation(model)
    conversation.send("1")
    conversation.answers[0] = replace(conversation.answers[0], cost_usd=None)
    conversation.send("2")
    assert conversation.cost_usd is None
