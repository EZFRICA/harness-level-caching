"""A real web search tool (Tavily) for H3, made repeatable by a tape.

Real search results change from one call to the next, so two arms of an experiment would
not see the same tool answers. The tape fixes that: the first time a normalised query is
seen it is sent to Tavily and the trimmed response is appended to the tape with its measured
latency; every later sighting, in any run that reads the same tape, is replayed with that
latency. A hit of the H3 cache is then compared with exactly what a miss would return, and
the time a hit saves is the latency a real call took.

Modes: `tape` (replay what is recorded, call live and record otherwise) and `replay` (never
call: an unrecorded query is a tool error). Live calls are capped per process.
Prices and credits: `pricing.toml`, section `tools`; sources in the docs read on 2026-10-24.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from cama.teu.tools import Tool, ToolError
from cama.types import ToolSpec

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
CONTENT_CHARS = 500  # per result: enough to answer, small enough to keep prompts comparable
TTL_S = 3600  # results of a general web search are taken as valid for an hour within a session


def normalise(args: dict) -> dict:
    return {"query": " ".join(str(args.get("query", "")).lower().split())}


def trim(body: dict, query: str) -> dict:
    results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "content": str(r.get("content", ""))[:CONTENT_CHARS]}
               for r in (body.get("results") or [])[:MAX_RESULTS]]
    return {"query": query, "results": results}


class WebTape:
    """Append-only JSONL: one line per recorded query. The first record of a query wins."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries.setdefault(e["query"], e)

    def get(self, query: str) -> dict | None:
        return self.entries.get(query)

    def put(self, query: str, value: dict, latency_ms: float) -> None:
        if query in self.entries:
            return
        e = {"query": query, "value": value, "latency_ms": round(latency_ms, 1), "recorded_at": time.time()}
        self.entries[query] = e
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")


class WebSearch:
    def __init__(self, tape: WebTape, mode: str = "tape", api_key: str = "", max_live_calls: int = 300,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep,
                 embed=None, tau: float = 0.0):
        if mode not in ("tape", "replay"):
            raise ValueError(f"unknown web mode {mode!r}")
        self.tape = tape
        self.mode = mode
        self.api_key = api_key
        self.max_live_calls = max_live_calls
        self.live_calls = 0
        self.replayed = 0
        self.transport = transport
        self.sleep = sleep
        # Hypothesis (semantic key, as in Cortex): a query close enough to one already answered in
        # this session gets that answer. Off when tau is 0. Every match is recorded for inspection.
        self.embed = embed
        self.tau = tau
        self.seen: list[tuple[str, object, dict, float]] = []  # query, unit vector, answer, when
        self.semantic: list[dict] = []

    async def __call__(self, args: dict, now: float) -> dict:
        query = args["query"]
        if not query:
            raise ToolError("web_search: empty query")
        vec = None
        if self.embed is not None and self.tau > 0:
            vec = (await self.embed([query], "query"))[0]
            best = max(((float(vec @ v), q, val) for q, v, val, at in self.seen if now - at < TTL_S),
                       default=None, key=lambda x: x[0])
            if best is not None and best[0] >= self.tau:
                self.semantic.append({"query": query, "matched": best[1], "sim": round(best[0], 3)})
                return best[2]
        value = await self._answer(query)
        if vec is not None:
            self.seen.append((query, vec, value, now))
        return value

    async def _answer(self, query: str) -> dict:
        rec = self.tape.get(query)
        if rec is not None:
            self.replayed += 1
            await self.sleep(rec["latency_ms"] / 1000)  # the time the real call took
            return rec["value"]
        if self.mode == "replay":
            raise ToolError(f"web_search: query not on the tape: {query!r}")
        if not self.api_key:
            raise ToolError("web_search: no API key for a live call")
        if self.live_calls >= self.max_live_calls:
            raise ToolError(f"web_search: live call cap {self.max_live_calls} reached")
        self.live_calls += 1
        t = time.perf_counter()
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            r = await client.post(TAVILY_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                  json={"query": query, "search_depth": "basic", "max_results": MAX_RESULTS,
                                        "include_answer": False})
        if r.status_code != 200:
            raise ToolError(f"web_search: HTTP {r.status_code}")
        value = trim(r.json(), query)
        self.tape.put(query, value, (time.perf_counter() - t) * 1000)
        return value

    def median_latency_ms(self, default: float = 1000.0) -> float:
        lat = sorted(e["latency_ms"] for e in self.tape.entries.values())
        return lat[len(lat) // 2] if lat else default


def web_search_tool(search: WebSearch) -> Tool:
    """`latency_ms` is the tape's median: it is what the loop credits a hit with. The runner does
    not sleep it (`simulated=False`); the call itself takes its real or recorded time."""
    spec = ToolSpec("web_search", "Search the web. Returns titles, URLs and short extracts.",
                    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
    return Tool(spec, "ttl", TTL_S, search.median_latency_ms(), search, normalise, simulated=False)
