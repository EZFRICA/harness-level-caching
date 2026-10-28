"""Words beside meaning: BM25 over block texts, fused with the vector ranking by reciprocal rank.

Taken from the owner's apu-urithi (library/keywords.py, measured there on 2026-10-15: on 38
labelled questions the vector alone ranked the right part first 30 times, the fusion 32).
Copied rather than imported: sibling projects are read only. Unchanged except names.
"""

from __future__ import annotations

import math
import re
from collections import Counter

RRF_K = 60  # reciprocal rank fusion constant as published
_WORD = re.compile(r"[^\W_]+")


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def bm25(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """The BM25 score of each document for the query; 0 where no word is shared."""
    tokens = [words(d) for d in docs]
    if not tokens:
        return []
    n, avg = len(tokens), (sum(map(len, tokens)) / len(tokens)) or 1.0
    df = Counter(t for ts in tokens for t in set(ts))
    asked = set(words(query))
    scores = []
    for ts in tokens:
        tf, score = Counter(ts), 0.0
        for t in asked:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(ts) / avg))
        scores.append(score)
    return scores


def fuse(by_meaning: list[str], by_words: list[str]) -> list[str]:
    """Reciprocal rank fusion of two rankings of ids, best first; ties keep the meaning's order."""
    fused: dict[str, float] = {}
    for ranking in (by_meaning, by_words):
        for rank, key in enumerate(ranking):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    order = {key: i for i, key in enumerate(by_meaning)}
    return sorted(fused, key=lambda key: (-fused[key], order.get(key, len(order))))
