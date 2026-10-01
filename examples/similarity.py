"""Meaning as geometry: cosine similarity between toy embeddings.

Run with:  uv run python examples/similarity.py
Real embeddings have hundreds or thousands of dimensions and come from a model.
These have three, set by hand, so you can see what is going on. The dimensions mean:
how much the word is about money, about fish, and about people.
"""

import math

vectors = {
    "revenue": [0.95, 0.10, 0.20],
    "driftsinntekter": [0.85, 0.20, 0.35],  # Norwegian: operating revenue
    "profit": [0.80, 0.05, 0.45],
    "salmon": [0.10, 0.95, 0.05],
    "laks": [0.25, 0.88, 0.15],  # Norwegian: salmon
    "employees": [0.20, 0.05, 0.95],
}


def cosine(a: list[float], b: list[float]) -> float:
    """1.0 means pointing the same way (same meaning); 0.0 means unrelated."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


for query in ("revenue", "salmon"):
    others = [word for word in vectors if word != query]
    ranked = sorted(others, key=lambda word: cosine(vectors[query], vectors[word]), reverse=True)
    print(f"Closest to {query!r}:")
    for word in ranked:
        print(f"  {word:<17}{cosine(vectors[query], vectors[word]):.2f}")
