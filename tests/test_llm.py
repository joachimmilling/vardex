from fakes import FakeModel

from vardex.llm import Message, ask
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
