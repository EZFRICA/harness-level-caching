"""How long does a provider keep an implicit prompt cache after the last request?

For each provider and each idle delay, a unique prefix (about 6000 tokens, never sent before) is
sent once, then sent again after the delay with 2500 new tokens appended (Gemini 4 caches the
previous request only when the new one adds at least 2050 tokens, per its caching page). The
cached token count of the second request says whether the prefix survived the idle time.
A refresh arm reads the prefix half way through the delay, to see whether a hit extends the life.

Usage: uv run python scripts/cache_ttl_probe.py <out_dir> [providers...]
Keys come from .env by variable name and are never printed. Spend is capped (MAX_USD).
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

DELAYS_MIN = [0.25, 2, 5, 10, 15, 20, 30, 45, 60]
REFRESH = [(25, 10)]  # (delay, read after this many minutes): does a hit extend the life?
MAX_USD = 1.0
PRICE_IN = {"gemini-3.5-flash-lite": 0.30, "gemini-4-argon": 2.00, "deepseek-ai/DeepSeek-V4-Flash-0731": 0.14}
spent = 0.0


def env(name: str) -> str:
    for line in Path(".env").read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"{name} is not set in .env")


def text(seed: str, words: int) -> str:
    rng = random.Random(seed)
    vocab = ["memoire", "agent", "cache", "prefixe", "voyage", "abidjan", "nantes", "budget", "projet", "mesure",
             "latence", "cout", "outil", "session", "tour", "bloc", "ordre", "stable", "jeton", "serveur"]
    return " ".join(rng.choice(vocab) + str(rng.randint(0, 999)) for _ in range(words))


async def gemini(model: str, system: str, user: str, key: str, c: httpx.AsyncClient) -> dict:
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 8, "temperature": 0}}
    r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                     headers={"x-goog-api-key": key}, json=body)
    u = r.json().get("usageMetadata", {}) if r.status_code == 200 else {}
    return {"status": r.status_code, "prompt": u.get("promptTokenCount", 0), "cached": u.get("cachedContentTokenCount", 0),
            "output": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}


async def nebius(model: str, system: str, user: str, key: str, c: httpx.AsyncClient) -> dict:
    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = await c.post("https://api.tokenfactory.nebius.com/v1/chat/completions",
                     headers={"Authorization": f"Bearer {key}"}, json=body)
    u = r.json().get("usage", {}) if r.status_code == 200 else {}
    return {"status": r.status_code, "prompt": u.get("prompt_tokens", 0),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "output": u.get("completion_tokens", 0)}


async def call(provider: str, model: str, system: str, user: str, keys: dict, c: httpx.AsyncClient) -> dict:
    global spent
    est = (len(system) + len(user)) / 4 * PRICE_IN[model] / 1e6
    if spent + est > MAX_USD:
        return {"status": "budget"}
    spent += est
    f = gemini if provider == "gemini" else nebius
    for attempt in range(4):
        try:
            out = await f(model, system, user, keys[provider], c)
            if out["status"] in (429, 500, 502, 503) and attempt < 3:
                await asyncio.sleep(5 * 2 ** attempt)
                continue
            return out
        except httpx.HTTPError as e:
            if attempt == 3:
                return {"status": f"error {type(e).__name__}"}
            await asyncio.sleep(5 * 2 ** attempt)


async def arm(provider, model, delay, refresh_at, keys, c, out: Path, t0: float):
    tag = f"{model}|{delay}|{refresh_at}|{t0}"
    system = "You are a memory probe.\n" + text(tag, 2400)
    first = await call(provider, model, system, "Answer OK. " + text(tag + "q1", 30), keys, c)
    if refresh_at is not None:
        await asyncio.sleep(refresh_at * 60)
        await call(provider, model, system, "Answer OK. " + text(tag + "r", 30), keys, c)
        await asyncio.sleep((delay - refresh_at) * 60)
    else:
        await asyncio.sleep(delay * 60)
    second = await call(provider, model, system, "Answer OK. " + text(tag + "q2", 1000), keys, c)
    row = {"provider": provider, "model": model, "delay_min": delay, "refresh_at_min": refresh_at,
           "first": first, "second": second, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(model, delay, refresh_at, "cached", second.get("cached"), "of", second.get("prompt"), flush=True)


async def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[2:] or list(PRICE_IN)
    keys = {"gemini": env("GEMINI_API_KEY"), "nebius": env("NEBIUS_API_KEY")}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as c:
        tasks = []
        for model in wanted:
            provider = "nebius" if "/" in model else "gemini"
            tasks += [arm(provider, model, d, None, keys, c, out_dir / "results.jsonl", t0) for d in DELAYS_MIN]
            tasks += [arm(provider, model, d, r, keys, c, out_dir / "results.jsonl", t0) for d, r in REFRESH]
        await asyncio.gather(*tasks)
    print(f"estimated input spend {spent:.3f} USD", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
