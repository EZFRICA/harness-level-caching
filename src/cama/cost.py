"""The cost of a run, from provider token counts and our price table (`pricing.toml`).

Hot path, per model call: uncached input, cached input, output (thinking included).
Cold path, attributed to the run: extraction calls, prefix cache creations, and cache
storage over the time each cache was alive. An unpriced model yields `priced: False` and no
money figures: an absent cost is never shown as zero.

The budget guard (`Budget`, `MeteredLLM`) checks an estimate before every paid call and
records the actual cost after it; crossing `CAMA_MAX_USD` stops the run cleanly.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from cama.tokens import approx_tokens
from cama.types import LLMResult, Message, ToolSpec

M = 1_000_000


def load_pricing(path: Path) -> dict:
    return tomllib.loads(path.read_text())["models"]


def pricing_hash(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=6).hexdigest() if path.exists() else "none"


def price_for(pricing: dict, model: str) -> dict | None:
    p = pricing.get(model)
    return p if p and p.get("status") in ("official", "local", "assumption") and "input" in p else None


def call_cost(p: dict, prompt: int, cached: int, output: int) -> dict:
    uncached = max(0, prompt - cached)
    return {"input": uncached * p["input"] / M, "cache_read": cached * p["cache_read"] / M,
            "output": output * p["output"] / M}


def run_cost(manifest: dict, turns: list[dict], tasks: list[dict], pricing: dict) -> dict:
    cfg = manifest["config"]
    model = cfg.get("model") if cfg.get("llm") != "echo" else "echo"
    p = price_for(pricing, model)
    if p is None:
        return {"priced": False, "model": model, "reason": "model not priced in pricing.toml"}
    per_turn, hot = [], {"input": 0.0, "cache_read": 0.0, "output": 0.0}
    for t in turns:
        c = call_cost(p, t["llm"]["prompt_tokens"], t["llm"]["cached_tokens"], t["llm"]["output_tokens"])
        for k in hot:
            hot[k] += c[k]
        per_turn.append({"turn": t["turn"], "usd": sum(c.values())})
    ext = [x for x in tasks if x["task"].startswith("extract") and x.get("ok")]
    # Extraction calls are cached like any call (review finding 2026-10-24: they were priced as
    # wholly uncached while the budget guard counted their cache); runs before that have no field.
    extraction = sum(sum(call_cost(p, x.get("prompt_tokens", 0), x.get("cached_tokens", 0),
                                   x.get("output_tokens", 0)).values()) for x in ext)
    creates = [x for x in tasks if x["task"].startswith("cache") and x.get("ok") and x.get("action") == "create"]
    cache_write = sum(x.get("tokens", 0) for x in creates) * p["cache_write"] / M
    end = max([x["ts"] for x in tasks + turns if "ts" in x] or [0])
    storage = 0.0
    for i, x in enumerate(creates):  # a cache lives until the next creation replaces it, or the run ends
        until = creates[i + 1]["ts"] if i + 1 < len(creates) else end
        storage += x.get("tokens", 0) * max(0.0, until - x["ts"]) / 3600 * p["cache_storage_hour"] / M
    cold = {"extraction": extraction, "cache_write": cache_write, "cache_storage": storage}
    total_hot, total_cold = sum(hot.values()), sum(cold.values())
    assumptions = []
    if p.get("cache_write_status") == "assumption" and creates:
        assumptions.append("cache creation priced as input (not stated by the provider)")
    if cfg.get("embedder") == "gemini":
        assumptions.append("remote embedding calls not priced")
    return {"priced": True, "model": model, "status": p["status"], "read_on": p["read_on"],
            "source": p["source"], "hot": hot, "cold": cold, "hot_usd": total_hot, "cold_usd": total_cold,
            "total_usd": total_hot + total_cold, "per_turn": per_turn, "assumptions": assumptions}


# ── the budget guard ────────────────────────────────────────────────────────

class BudgetExceeded(RuntimeError):
    retryable = False


@dataclass
class Budget:
    """Reserve the estimate before a call, settle with the actual cost after it. Reserving (not
    only checking) keeps concurrent calls, a turn and a cold task, from crossing the cap together."""

    max_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    log: list[dict] = field(default_factory=list)

    async def agrant(self, estimate_usd: float, what: str) -> None:
        if self.spent_usd + estimate_usd > self.max_usd:
            raise BudgetExceeded(f"{what}: {self.spent_usd:.4f} spent + {estimate_usd:.4f} estimated "
                                 f"> cap {self.max_usd:.2f} USD")
        self.spent_usd += estimate_usd

    async def arecord(self, usd: float, estimate_usd: float, what: str) -> None:
        self.spent_usd += usd - estimate_usd
        self.calls += 1
        self.log.append({"what": what, "usd": usd})


@dataclass
class SharedBudget(Budget):
    """One cap per run for every worker of a cluster, reserved atomically in the coordinator.
    Review finding: a budget per engine made the real cap agents x workers x 2 USD."""

    coord: object = None
    run_id: str = ""

    async def agrant(self, estimate_usd: float, what: str) -> None:
        if not await self.coord.reserve(self.run_id, self.max_usd, estimate_usd):
            raise BudgetExceeded(f"{what}: the run {self.run_id} would cross its shared cap {self.max_usd:.2f} USD")
        self.spent_usd += estimate_usd

    async def arecord(self, usd: float, estimate_usd: float, what: str) -> None:
        await self.coord.adjust(self.run_id, usd - estimate_usd)
        await super().arecord(usd, estimate_usd, what)


class MeteredLLM:
    """Wraps a paid model: a budget is granted before each call, the actual cost recorded after."""

    EST_OUTPUT_TOKENS = 600  # pessimistic, thinking included

    def __init__(self, inner, price: dict, budget: Budget, cache_ttl_s: int):
        self.inner = inner
        self.name = inner.name
        self.price = price
        self.budget = budget
        self.cache_ttl_s = cache_ttl_s

    def __getattr__(self, item):  # create_cache presence is how the engine detects caching support
        return getattr(self.inner, item)

    async def generate(self, system: str, messages: list[Message], **kw) -> LLMResult:
        est_in = approx_tokens(system) + sum(approx_tokens(m.text or str(m.result or "")) for m in messages)
        est = (est_in * self.price["input"] + self.EST_OUTPUT_TOKENS * self.price["output"]) / M
        await self.budget.agrant(est, "model call")
        try:
            out = await self.inner.generate(system, messages, **kw)
        except BaseException:
            await self.budget.arecord(0.0, est, "model call failed")  # a failed call is not billed
            raise
        await self.budget.arecord(sum(call_cost(self.price, out.prompt_tokens, out.cached_tokens,
                                                out.output_tokens).values()), est, "model call")
        return out

    async def create_cache(self, system: str, tools: list[ToolSpec] | None, ttl_s: int) -> tuple[str, int]:
        est_tokens = approx_tokens(system) + 200
        est = est_tokens * (self.price["cache_write"] + self.price["cache_storage_hour"] * ttl_s / 3600) / M
        await self.budget.agrant(est, "cache creation")
        try:
            name, tokens = await self.inner.create_cache(system, tools, ttl_s)
        except BaseException:
            await self.budget.arecord(0.0, est, "cache creation failed")
            raise
        # Storage is charged up front for the full TTL: the guard must not be outrun by caches alive.
        await self.budget.arecord(tokens * (self.price["cache_write"] + self.price["cache_storage_hour"] * ttl_s / 3600)
                                  / M, est, "cache creation")
        return name, tokens
