from types import SimpleNamespace

from vardex.config import Settings
from vardex.llm import ask


class FakeMessages:
    """Stands in for client.messages, so tests never call the real API."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text_block = SimpleNamespace(type="text", text="A nine-digit number that identifies…")
        return SimpleNamespace(content=[text_block])


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_ask_sends_the_question_and_returns_the_text():
    client = FakeClient()
    settings = Settings(anthropic_api_key="test", model="test-model")

    answer = ask("What is an organisation number?", settings, client=client)

    assert answer.startswith("A nine-digit number")
    call = client.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [{"role": "user", "content": "What is an organisation number?"}]
