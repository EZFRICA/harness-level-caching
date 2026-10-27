"""Implicit prompt cache life of Gemini models, through the Gemini API (AI Studio) and Vertex AI.

For each (model, entry point, idle delay) and several samples, a unique prefix of about 6.5k
tokens is sent once, then sent again after the delay with about 2.5k new tokens appended
(Gemini 4 caches a previous request only when the next one adds at least 2050 tokens, per its
caching page). The cached token count of the second request says whether the prefix survived.
A refresh arm reads the prefix half way through to see whether a hit extends the life.

Spend is counted from the token counts the API returns, at the prices in pricing.toml, and the
run stops starting new arms past MAX_USD. Keys come from .env and gcloud; none is printed.

Usage: uv run python scripts/cache_ttl_probe_v2.py <out_dir> [vertex_project]  (no project: Gemini API only)
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx

PROJECT = sys.argv[2] if len(sys.argv) > 2 else ""
MODELS = ["gemini-3.5-flash-lite", "gemini-4-argon"]
ENTRIES = ["aistudio", "vertex"] if len(sys.argv) > 2 else ["aistudio"]
DELAYS_MIN = [0.25, 1, 2, 3, 5, 8, 15, 30]
REFRESH = [(8, 4)]  # (delay, read at this minute)
SAMPLES = 3
MAX_USD = 3.0
PRICES = tomllib.loads(Path("pricing.toml").read_text())["models"]
spent = 0.0
_token = {"value": "", "at": 0.0}


def gemini_key() -> str:
    for line in Path(".env").read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("GEMINI_API_KEY is not set in .env")


def vertex_token() -> str:
    if time.time() - _token["at"] > 600:
        _token["value"] = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                         text=True).stdout.strip()
        _token["at"] = time.time()
    return _token["value"]


def text(seed: str, words: int) -> str:
    rng = random.Random(seed)
    vocab = ["memory", "agent", "cache", "prefix", "travel", "budget", "project", "measure", "latency",
             "cost", "tool", "session", "turn", "block", "order", "stable", "token", "server", "window"]
    return " ".join(rng.choice(vocab) + str(rng.randint(0, 999)) for _ in range(words))


def charge(model: str, u: dict) -> None:
    global spent
    p = PRICES[model]
    prompt, cached = u.get("promptTokenCount", 0), u.get("cachedContentTokenCount", 0)
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    spent += ((prompt - cached) * p["input"] + cached * p["cache_read"] + out * p["output"]) / 1e6


async def call(entry: str, model: str, system: str, user: str, c: httpx.AsyncClient) -> dict:
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0}}
    if entry == "aistudio":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": gemini_key()}
    else:
        url = (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/publishers/google/"
               f"models/{model}:generateContent")
        headers = {"Authorization": f"Bearer {vertex_token()}", "x-goog-user-project": PROJECT}
    for attempt in range(4):
        try:
            r = await c.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            r = None
            err = type(e).__name__
        if r is not None and r.status_code == 200:
            u = r.json().get("usageMetadata", {})
            charge(model, u)
            return {"status": 200, "prompt": u.get("promptTokenCount", 0), "cached": u.get("cachedContentTokenCount", 0)}
        if r is not None and r.status_code not in (429, 500, 502, 503):
            return {"status": r.status_code, "error": r.text[:200]}
        await asyncio.sleep(5 * 2 ** attempt)
    return {"status": "failed", "error": err if r is None else r.status_code}


async def arm(entry, model, delay, refresh_at, sample, c, out: Path, t0: float):
    if spent > MAX_USD:
        return
    tag = f"{entry}|{model}|{delay}|{refresh_at}|{sample}|{t0}"
    system = "You are a cache probe.\n" + text(tag, 2600)
    first = await call(entry, model, system, "Answer OK. " + text(tag + "q1", 20), c)
    if refresh_at is not None:
        await asyncio.sleep(refresh_at * 60)
        await call(entry, model, system, "Answer OK. " + text(tag + "r", 20), c)
        await asyncio.sleep((delay - refresh_at) * 60)
    else:
        await asyncio.sleep(delay * 60)
    second = await call(entry, model, system, "Answer OK. " + text(tag + "q2", 800), c)
    row = {"entry": entry, "model": model, "delay_min": delay, "refresh_at_min": refresh_at, "sample": sample,
           "first": first, "second": second, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(entry, model, delay, refresh_at, sample, "cached", second.get("cached"), "of", second.get("prompt"),
          f"spent {spent:.3f}", flush=True)


async def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=180) as c:
        tasks = []
        for entry in ENTRIES:
            for model in MODELS:
                for s in range(SAMPLES):
                    tasks += [arm(entry, model, d, None, s, c, out_dir / "results.jsonl", t0) for d in DELAYS_MIN]
                    tasks += [arm(entry, model, d, r, s, c, out_dir / "results.jsonl", t0) for d, r in REFRESH]
        await asyncio.gather(*tasks)
    print(f"spent {spent:.3f} USD, counted from the API's token counts", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
