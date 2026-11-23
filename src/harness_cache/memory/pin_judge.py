"""Who decides which folded statements H10 pins (HC_PIN_JUDGE).

`rule` is states_a_fact, a few lines of code under version control. `decisions` asks the hosted
Gemini Decisions API one yes/no question per message the rule already accepts, so a question is
never pinned and few calls are made. The hosted model names no version in its answers, so before
each fold it replays a frozen probe set; a probability that moved more than the tolerance from its
frozen value sends the run back to the rule for good, and a failed call sends that fold back to
the rule. Both are recorded. It runs in the cold path only (at a fold), never in a turn.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

QUESTION = {"states_fact": {
    "type": "noul",
    "instructions": ("Does this message state a durable fact about the user's life or work (a name, number, code, "
                     "date, place, preference or reference) that a later question could ask for word for word?"),
    "criteria": {"true_description": "States such a fact, or corrects one.",
                 "false_description": "A question, a request, small talk, an opinion, or a vague progress note."}}}
MODEL = "gemini-decision-1"
THRESHOLD = 0.5
TOLERANCE = 0.10
# the repository's copy when run from a checkout, the mounted /scenarios in the container
_IN_REPO = Path(__file__).resolve().parents[3] / "scenarios" / "labels" / "pin_probes_v1.jsonl"
PROBES = _IN_REPO if _IN_REPO.exists() else Path("/scenarios/labels/pin_probes_v1.jsonl")


def load_probes(path: Path = PROBES) -> list[tuple[str, float]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [(r["text"], r["p"]) for r in rows if "p" in r]


@dataclass
class Verdict:
    keep: set[str]
    calls: int = 0
    tokens: int = 0
    fallback: str | None = None  # why the rule decided this fold instead of the API


@dataclass
class DecisionsJudge:
    """`ask(texts)` returns, per text, (probability, input tokens); the API or a fake."""
    ask: Callable
    rule: Callable[[str], bool]
    probes: list[tuple[str, float]]
    threshold: float = THRESHOLD
    tolerance: float = TOLERANCE
    drifted: str | None = field(default=None)

    async def judge(self, texts: list[str]) -> Verdict:
        candidates = [t for t in dict.fromkeys(texts) if self.rule(t)]
        if self.drifted:
            return Verdict(set(candidates), fallback=self.drifted)
        try:
            probe = await self.ask([t for t, _ in self.probes])
            calls, tokens = len(self.probes), sum(n for _, n in probe.values())
            for text, frozen in self.probes:
                if abs(probe[text][0] - frozen) > self.tolerance:
                    self.drifted = f"probe drift: {probe[text][0]:.3f} against {frozen:.3f} frozen"
                    return Verdict(set(candidates), calls, tokens, self.drifted)
            got = await self.ask(candidates) if candidates else {}
        except Exception as e:  # the API failed: this fold falls back to the rule, the next one tries again
            return Verdict(set(candidates), fallback=f"api error: {type(e).__name__}")
        return Verdict({t for t in candidates if got[t][0] >= self.threshold}, calls + len(got),
                       tokens + sum(n for _, n in got.values()))


def gemini_asker(api_key: str, parallel: int = 5) -> Callable:
    from google import genai
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    async def ask(texts: list[str]) -> dict[str, tuple[float, int]]:
        gate = asyncio.Semaphore(parallel)

        async def one(t: str):
            async with gate:
                r = await client.aio.decisions.create(model=MODEL, input=t, questions=QUESTION)
                return t, (float(r.answers["states_fact"].noul), int(r.usage.total_input_tokens))
        return dict(await asyncio.gather(*(one(t) for t in texts)))
    return ask
