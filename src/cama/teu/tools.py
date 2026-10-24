"""Simulated tools for agentic experiments.

Deterministic answers (a seeded function of the arguments and, for time-sensitive tools,
of the time bucket), configurable latency and failure rate, so a run can be repeated and a
cache hit can be checked against what a miss would have computed.

Cache classes (decision 0006, skill add-a-cache-level):
  pure   same arguments, same answer forever          (calculator)
  ttl    same answer within `ttl_s`                    (weather, fx_rate, search_notes)
  never  has a side effect; never served from cache   (book_hotel)
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import operator
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from cama.types import ToolSpec


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    cache_class: str  # pure | ttl | never
    ttl_s: float
    latency_ms: float
    fn: Callable[[dict, float], Awaitable[dict] | dict]
    normalise: Callable[[dict], dict] = field(default=lambda a: a)
    simulated: bool = True  # False: a real call that takes its own time (web_search); never slept


def _seeded(*parts: object) -> random.Random:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "little"))


def _expr(a: dict) -> dict:
    """Canonical form of an arithmetic expression; an unparsable one keeps its raw text, so it
    never shares a key with a valid one. Review finding: stripping every space turned the
    syntax error "1 2+3" into "12+3"."""
    raw = str(a.get("expression", "")).strip()
    try:
        return {"expression": ast.unparse(ast.parse(raw, mode="eval"))}
    except SyntaxError:
        return {"expression": raw}


def _city(a: dict) -> dict:
    return {**a, "city": str(a.get("city", "")).strip().title()}


def _weather(args: dict, now: float) -> dict:
    bucket = int(now // 600)  # the answer changes every 10 minutes, as the TTL assumes
    r = _seeded("weather", args["city"], args.get("date", "today"), bucket)
    return {"city": args["city"], "date": args.get("date", "today"),
            "temp_c": r.randint(18, 36), "rain_mm": round(r.random() * 12, 1)}


def _fx(args: dict, now: float) -> dict:
    pair = f"{args['base'].upper()}/{args['quote'].upper()}"
    if pair == "EUR/XOF":
        rate = 655.957  # fixed parity
    else:
        rate = round(0.5 + _seeded("fx", pair, int(now // 60)).random() * 2, 4)
    return {"pair": pair, "rate": rate}


_NOTES = {
    "devfest": "DevFest Abidjan talk on 14 December, slides due 1 December, live demo on the Pixel.",
    "budget": "Travel budget 1200 euros maximum; hotel on the Plateau at 75 euros per night.",
    "pixel": "Pixel 4 XL, 6 GB RAM, Snapdragon 855; low-end profile pins to the four A55 cores.",
    "cache": "Gemini explicit caching needs 1024 tokens; implicit caching was not seen below 5000.",
    "visa": "French passport; check the e-visa requirement for Cote d'Ivoire before departure.",
}


def _search(args: dict, now: float) -> dict:
    q = str(args.get("query", "")).lower()
    hits = [{"note": k, "text": v} for k, v in _NOTES.items() if k in q or any(w in v.lower() for w in q.split() if len(w) > 4)]
    return {"query": args.get("query"), "results": hits[:3]}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 12:
            raise ToolError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("only arithmetic is allowed")


def _calc(args: dict, now: float) -> dict:
    expr = str(args.get("expression", ""))
    return {"expression": expr, "value": round(_eval(ast.parse(expr, mode="eval").body), 6)}


class Bookings:
    """The side effect of book_hotel: every call creates a booking, cached or not."""

    def __init__(self) -> None:
        self.made: list[dict] = []

    def book(self, args: dict, now: float) -> dict:
        ref = f"BK{len(self.made) + 1:04d}"
        self.made.append({"ref": ref, **args})
        return {"ref": ref, "city": args["city"], "nights": args["nights"], "status": "confirmed"}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def default_tools(bookings: Bookings) -> dict[str, Tool]:
    s = {"type": "string"}
    tools = [
        Tool(ToolSpec("weather", "Weather forecast for a city.", _obj({"city": s, "date": s}, ["city"])),
             "ttl", 600, 300, _weather, _city),
        Tool(ToolSpec("fx_rate", "Exchange rate between two currencies, ISO codes.",
                      _obj({"base": s, "quote": s}, ["base", "quote"])),
             "ttl", 60, 150, _fx, lambda a: {"base": str(a["base"]).upper(), "quote": str(a["quote"]).upper()}),
        Tool(ToolSpec("search_notes", "Search the user's notes.", _obj({"query": s}, ["query"])),
             "ttl", 3600, 200, _search, lambda a: {"query": " ".join(str(a.get("query", "")).lower().split())}),
        Tool(ToolSpec("calculator", "Evaluate an arithmetic expression.", _obj({"expression": s}, ["expression"])),
             "pure", 0, 5, _calc, _expr),
        Tool(ToolSpec("book_hotel", "Book a hotel. This makes a real reservation.",
                      _obj({"city": s, "nights": {"type": "integer"}}, ["city", "nights"])),
             "never", 0, 500, bookings.book, _city),
    ]
    return {t.spec.name: t for t in tools}


@dataclass
class FaultConfig:
    error_rate: float = 0.0
    latency_scale: float = 1.0
    seed: int = 7


class ToolRunner:
    """Executes a tool for real: latency, injected failures, the side effect if any."""

    def __init__(self, tools: dict[str, Tool], faults: FaultConfig, clock: Callable[[], float] = time.time):
        self.tools = tools
        self.faults = faults
        self.clock = clock
        self._rng = random.Random(faults.seed)

    async def run(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name}")
        if tool.simulated:
            await asyncio.sleep(tool.latency_ms * self.faults.latency_scale / 1000)
        if self._rng.random() < self.faults.error_rate:
            raise ToolError(f"{name}: injected failure")
        # Run on the same normalised arguments the cache keys on: a hit must equal what a miss computes.
        out = tool.fn(tool.normalise(args), self.clock())
        return await out if asyncio.iscoroutine(out) else out
