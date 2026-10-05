"""One round of tool calling, by hand: the model asks our code to check organisation numbers.

Run with:  uv run python examples/tool_call.py
The script calls the API directly, so you can see every message. Repeating the round until the
model stops asking for tools is what an agent does; this script stops after one round.
"""

import json

import anthropic

from vardex.config import load_settings

QUESTION = "Are 923609016 and 923609017 valid Norwegian organisation numbers?"
WEIGHTS = (3, 2, 7, 6, 5, 4, 3, 2)  # Brønnøysund's weights for the modulus 11 check

TOOLS = [
    {
        "name": "check_organisation_number",
        "description": (
            "Check whether a Norwegian organisation number is valid, using the control digit "
            "the Brønnøysund Register Centre uses. Call it for every number the user asks about; "
            "never check a number yourself."
        ),
        "strict": True,  # the input is guaranteed to match the schema
        "input_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "Nine digits; spaces are allowed."}
            },
            "required": ["number"],
            "additionalProperties": False,
        },
    }
]


def check_organisation_number(number: str) -> dict:
    """The modulus 11 check: the ninth digit must match the weighted sum of the first eight."""
    digits = number.replace(" ", "")
    if len(digits) != 9 or not digits.isdigit():
        return {"number": number, "valid": False, "reason": "not nine digits"}
    remainder = sum(int(d) * w for d, w in zip(digits[:8], WEIGHTS, strict=True)) % 11
    control = 0 if remainder == 0 else 11 - remainder
    if control == 10:
        return {"number": digits, "valid": False, "reason": "no number can have this control"}
    if control != int(digits[8]):
        return {"number": digits, "valid": False, "reason": f"the control should be {control}"}
    return {"number": digits, "valid": True, "reason": "the control digit matches"}


def show(title: str, content: object) -> None:
    print(f"--- {title}\n{content}\n")


def main() -> None:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages = [{"role": "user", "content": QUESTION}]
    show("1. You ask", QUESTION)

    first = client.messages.create(
        model=settings.model, max_tokens=4096, tools=TOOLS, messages=messages
    )
    calls = [block for block in first.content if block.type == "tool_use"]
    show(
        f"2. The model answers with stop_reason={first.stop_reason!r} and asks for tools",
        "\n".join(f"{call.name}({json.dumps(call.input)})  id={call.id}" for call in calls),
    )

    results = []
    for call in calls:
        result = check_organisation_number(**call.input)
        results.append(
            {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result)}
        )
    show("3. Your code runs them and sends the results back", json.dumps(results, indent=2))

    messages.append({"role": "assistant", "content": first.content})  # unchanged, thinking too
    messages.append({"role": "user", "content": results})  # all results in one message
    second = client.messages.create(
        model=settings.model, max_tokens=4096, tools=TOOLS, messages=messages
    )
    text = "".join(block.text for block in second.content if block.type == "text")
    show(f"4. The model answers with stop_reason={second.stop_reason!r}", text)


if __name__ == "__main__":
    main()
