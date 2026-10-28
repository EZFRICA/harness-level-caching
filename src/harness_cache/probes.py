"""Recall probes: does an answer contain the facts the conversation established?

A scenario line may carry `"expect": ["fact", "alt a|alt b", ...]`. Each fact is found when one
of its alternatives appears in the answer, after lower-casing, removing accents and collapsing
spaces. The score is deterministic: no judge model, so two runs are compared on the same rule.
It measures recall of planted facts, not the overall quality of an answer.
"""

from __future__ import annotations

import re
import unicodedata


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Thousands separators inside a number ("9,800", "9 800") would hide "9800" (seen on 2026-10-25).
    text = re.sub(r"(?<=\d)[,.\u202f\u00a0 ](?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def score(answer: str, expect: list[str]) -> list[bool]:
    a = fold(answer)
    return [any(fold(alt) in a for alt in fact.split("|") if alt.strip()) for fact in expect]


def summary(items: list[dict]) -> dict:
    facts = sum(len(i["found"]) for i in items)
    found = sum(sum(i["found"]) for i in items)
    return {"probes": len(items), "facts": facts, "found": found,
            "answered": sum(all(i["found"]) for i in items), "items": items}
