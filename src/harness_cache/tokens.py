"""Token estimation and lexical features.

`approx_tokens` is a fixed, provider-independent estimate (chars/4). It is used
for budgets and for the stable-prefix metric so that numbers are comparable
across LLM backends; provider-reported counts are recorded separately.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+", re.UNICODE)

_STOP = frozenset(
    """
    the and for are but not you your with this that have from what when where which who
    will would can could should about into than then them they there their our out was were
    les des une est pas pour que qui dans sur par avec son ses aux mais ou donc car mes tes
    nos vos leur leurs elle ils elles nous vous cette ces comme tout tous plus moins tres
    quoi quel quelle quels quelles faire fait peux veux suis etre avoir ete aussi bien
    """.split()
)


def approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def lexical_set(text: str) -> frozenset[str]:
    return frozenset(
        w for w in (m.group(0).lower() for m in _WORD.finditer(text)) if len(w) >= 3 and w not in _STOP
    )


def containment(query: frozenset[str], doc: frozenset[str]) -> tuple[float, int]:
    """Fraction of query terms present in the document, and the raw overlap count."""
    if not query:
        return 0.0, 0
    overlap = len(query & doc)
    return overlap / len(query), overlap
