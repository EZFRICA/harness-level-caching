"""The same choice as scripts/eval_fact_classifier.py (which folded messages H10 should pin), asked
to the hosted Gemini Decisions API with one yes/no question per message. A paid call: about 180
input tokens per message, no output tokens. The key is read from .env by its variable name and
never printed.

    uv run --with "google-genai>=3.2" python scripts/eval_fact_decisions.py [--key-var GEMINI_API_KEY_2] [--json out.json]

Fixed before the run: the question below, the 0.5 threshold, and the two variants compared with
the rule (the API alone, and the rule and the API together).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_fact_classifier import planted, score  # noqa: E402

from harness_cache.engine import states_a_fact  # noqa: E402

MODEL = "gemini-decision-1"
THRESHOLD = 0.5
QUESTION = {"states_fact": {
    "type": "noul",
    "instructions": ("Does this message state a durable fact about the user's life or work (a name, number, code, "
                     "date, place, preference or reference) that a later question could ask for word for word?"),
    "criteria": {"true_description": "States such a fact, or corrects one.",
                 "false_description": "A question, a request, small talk, an opinion, or a vague progress note."}}}


def key_from_env_file(name: str) -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} is not set in .env")


async def ask_all(client, texts: list[str], parallel: int = 5) -> dict[str, dict]:
    gate = asyncio.Semaphore(parallel)
    out: dict[str, dict] = {}

    async def one(text: str):
        async with gate:
            t0 = time.perf_counter()
            r = await client.aio.decisions.create(model=MODEL, input=text, questions=QUESTION)
            out[text] = {"p": r.answers["states_fact"].noul, "ms": (time.perf_counter() - t0) * 1000,
                         "tokens": r.usage.total_input_tokens, "output_tokens": r.usage.total_output_tokens}
    await asyncio.gather(*(one(t) for t in texts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-var", default="GEMINI_API_KEY_2")
    ap.add_argument("--json")
    args = ap.parse_args()
    from google import genai

    msgs = [json.loads(l)["user"] for l in (ROOT / "scenarios/long_facts_en_1500_v1.jsonl").read_text().splitlines()
            if '"user"' in l]
    counts = Counter(msgs)
    distinct = sorted(counts)
    positives = planted() & set(distinct)
    client = genai.Client(api_key=key_from_env_file(args.key_var), http_options={"api_version": "v1beta"})
    answers = asyncio.run(ask_all(client, distinct))
    api = {m for m in distinct if answers[m]["p"] >= THRESHOLD}
    rule = {m for m in distinct if states_a_fact(m)}
    ms = sorted(a["ms"] for a in answers.values())
    tokens = sum(a["tokens"] for a in answers.values())
    result = {
        "model": MODEL, "model_version_in_response": None, "key_variable": args.key_var,
        "calls": len(answers), "input_tokens": tokens, "output_tokens": sum(a["output_tokens"] for a in answers.values()),
        "cost_usd_at_0.10_per_million": round(tokens * 0.10 / 1e6, 6),
        "latency_ms": {"median": round(ms[len(ms) // 2]), "p95": round(ms[int(0.95 * len(ms))]), "max": round(ms[-1])},
        "threshold": THRESHOLD,
        "results": [score("rule states_a_fact", rule, positives, counts),
                    score("Decisions API", api, positives, counts),
                    score("rule and Decisions API", rule & api, positives, counts)],
        "lowest_positive_probability": round(min(answers[m]["p"] for m in positives), 3),
        "highest_negative_probability": round(max(answers[m]["p"] for m in distinct if m not in positives), 3),
        "probabilities": {m: round(answers[m]["p"], 4) for m in distinct},
    }
    summary = {k: v for k, v in result.items() if k != "probabilities"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
