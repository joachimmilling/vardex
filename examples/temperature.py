"""How temperature reshapes a model's choice of the next token.

Run with:  uv run python examples/temperature.py
The scores are made up. A real model scores every token in a vocabulary of 100,000 or more.
"""

import math

# Scores (logits) a model might give each candidate for the next token after
# "An organisation number in Norway has nine"
logits = {"digits": 7.0, "numbers": 5.2, "figures": 4.4, "characters": 4.0, "lives": 0.5}


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """Turn scores into probabilities. Temperature divides the scores first."""
    scaled = {token: score / temperature for token, score in scores.items()}
    top = max(scaled.values())
    weights = {token: math.exp(score - top) for token, score in scaled.items()}
    total = sum(weights.values())
    return {token: weight / total for token, weight in weights.items()}


print(f"{'temperature':<13}" + "".join(f"{token:>12}" for token in logits))
for temperature in (0.2, 0.5, 1.0, 1.5):
    probabilities = softmax(logits, temperature)
    print(f"{temperature:<13}" + "".join(f"{p:>12.1%}" for p in probabilities.values()))
