"""Load generator for the cluster: many agents, each replaying a scenario, routed to workers.

Routing is rendezvous hashing over the workers believed healthy, so an agent always goes to
the same worker while it lives. On a connection error, a 5xx or a 409 (another worker owns
the agent) the request moves to the next worker in the agent's ranking, after a short
backoff; a worker that failed is avoided for `cooldown_s`. Every request is recorded, with
the worker that answered, its status, its latency, its attempts and whether it failed over.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np


def rank(agent_id: str, workers: list[str]) -> list[str]:
    return sorted(workers, key=lambda w: hashlib.blake2b(f"{agent_id}|{w}".encode(), digest_size=8).digest(),
                  reverse=True)


@dataclass
class LoadConfig:
    workers: list[str]
    agents: int
    turns: int
    concurrency: int
    scenario: Path
    out: Path
    think_ms: float = 0.0
    timeout_s: float = 60.0
    connect_timeout_s: float = 1.5
    max_attempts: int = 12
    cooldown_s: float = 3.0


@dataclass
class Load:
    cfg: LoadConfig
    suspect: dict = field(default_factory=dict)  # worker -> time until which it is avoided
    rows: list = field(default_factory=list)

    def _healthy_first(self, agent_id: str) -> list[str]:
        order = rank(agent_id, self.cfg.workers)
        now = time.time()
        return [w for w in order if self.suspect.get(w, 0) <= now] + [w for w in order if self.suspect.get(w, 0) > now]

    async def _post(self, client, agent_id: str, path: str, body: dict) -> tuple[dict, str, int, int, bool]:
        first = None
        last_err = ""
        for attempt in range(1, self.cfg.max_attempts + 1):
            busy = False
            for w in self._healthy_first(agent_id):
                first = first or w
                try:
                    r = await client.post(f"{w}/agents/{agent_id}/{path}", json=body)
                except httpx.HTTPError as e:
                    self.suspect[w] = time.time() + self.cfg.cooldown_s
                    last_err = f"{w}: {type(e).__name__}"
                    continue
                if r.status_code == 200:
                    return r.json(), w, 200, attempt, w != first
                last_err = f"{w}: {r.status_code}"
                if r.status_code == 402:  # budget: never retried
                    return {"error": "budget"}, w, 402, attempt, w != first
                if r.status_code == 503 and r.headers.get("Retry-After"):
                    busy = True  # alive but at capacity: wait and retry, do not shun it
                    continue
                if r.status_code >= 500:
                    self.suspect[w] = time.time() + self.cfg.cooldown_s
            await asyncio.sleep(1.0 + random.random() if busy else min(2.0, 0.25 * 2 ** (attempt - 1)))
        return {"error": last_err}, first or "", 599, self.cfg.max_attempts, True

    async def _agent(self, client, sem, agent_id: str, seed: list[dict], users: list[str]) -> None:
        async with sem:
            await self._post(client, agent_id, "seed", {"blocks": seed})
            for i, text in enumerate(users[: self.cfg.turns]):
                t = time.perf_counter()
                body, w, status, attempts, failover = await self._post(client, agent_id, "turn", {"text": text})
                row = {"ts": time.time(), "agent": agent_id, "i": i + 1, "worker": w, "status": status,
                       "ms": (time.perf_counter() - t) * 1000, "attempts": attempts, "failover": failover,
                       "server_hot_ms": body.get("hot_ms"), "epoch": body.get("epoch"),
                       "recovery": body.get("recovery"), "cache_hit": (body.get("prefix_cache") or {}).get("hit"),
                       "h3_hits": (body.get("h3") or {}).get("hits")}
                self.rows.append(row)
                with self.cfg.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if self.cfg.think_ms:
                    await asyncio.sleep(self.cfg.think_ms / 1000)

    async def run(self) -> dict:
        lines = [json.loads(x) for x in self.cfg.scenario.read_text().splitlines() if x.strip()]
        seed = next(x["seed"] for x in lines if "seed" in x)
        users = [x["user"] for x in lines if "user" in x]
        self.cfg.out.parent.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self.cfg.concurrency)
        t0 = time.perf_counter()
        # A killed container does not refuse connections, it stays silent: without a short connect
        # timeout a client waits the whole request timeout before failing over (measured: 45 s).
        timeout = httpx.Timeout(self.cfg.timeout_s, connect=self.cfg.connect_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            await asyncio.gather(*(self._agent(client, sem, f"a{n:03d}", seed, users) for n in range(self.cfg.agents)))
        return summarize(self.rows, time.perf_counter() - t0)


def _takeovers(rows: list[dict]) -> int:
    """How many times an agent's epoch moved up: each is one takeover by another worker."""
    n, last = 0, {}
    for r in sorted((r for r in rows if r.get("epoch")), key=lambda r: (r["agent"], r["i"])):
        if r["agent"] in last and r["epoch"] > last[r["agent"]]:
            n += 1
        last[r["agent"]] = r["epoch"]
    return n


def summarize(rows: list[dict], wall_s: float) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    ms = [r["ms"] for r in ok]
    by_worker: dict[str, int] = {}
    for r in ok:
        by_worker[r["worker"]] = by_worker.get(r["worker"], 0) + 1
    return {
        "requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "failovers": sum(r["failover"] for r in rows), "wall_s": wall_s,
        "throughput_rps": len(ok) / wall_s if wall_s else 0.0,
        "p50_ms": float(np.percentile(ms, 50)) if ms else None, "p95_ms": float(np.percentile(ms, 95)) if ms else None,
        "p99_ms": float(np.percentile(ms, 99)) if ms else None, "by_worker": by_worker,
        "takeovers": _takeovers(rows),
    }
