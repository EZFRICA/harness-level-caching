"""Regression tests for the findings of the step 1 to 5 review; each docstring names the defect."""

import asyncio
import json
from pathlib import Path

import pytest

from harness_cache.adapters.local import EchoLLM
from harness_cache.agent.loop import AgentLoop
from harness_cache.cluster.coordinator import Lease, MemoryCoordinator, StaleEpoch
from harness_cache.cluster.recovery import recover
from harness_cache.config import Config
from harness_cache.cost import BudgetExceeded, SharedBudget
from harness_cache.engine import Engine
from harness_cache.metrics import RunRecorder
from harness_cache.teu.cache import ToolCache
from harness_cache.teu.tools import Bookings, FaultConfig, ToolRunner, default_tools
from harness_cache.types import Message

SCENARIO = Path(__file__).parent.parent / "scenarios" / "agentic_fr_v1.jsonl"


def _cfg(tmp_path, **kw):
    return Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                  tool_latency_scale=0, **kw)


async def test_a_cache_failure_mid_turn_never_books_twice():
    """Review: the fallback replayed the whole loop, so a hotel already booked was booked again."""
    import sqlite3

    class CacheDiesAtStep2(EchoLLM):
        calls = 0

        async def generate(self, system, messages, *, cached_content=None, **kw):
            CacheDiesAtStep2.calls += 1
            if cached_content and messages[-1].role == "tool":
                raise RuntimeError("404 cached content not found")
            return await super().generate(system, messages, cached_content=None, **kw)

    bookings = Bookings()
    tools = default_tools(bookings)
    loop = AgentLoop(CacheDiesAtStep2(), ToolRunner(tools, FaultConfig(latency_scale=0)),
                     ToolCache(sqlite3.connect(":memory:")), max_steps=4)
    res = await loop.run("sys", [Message("user", "Réserve l'hôtel à Abidjan pour trois nuits.")],
                         [t.spec for t in tools.values()], "some-cache")
    assert len(bookings.made) == 1 and res.cache_error and res.stopped == "answer"


async def test_a_snapshot_ahead_of_the_shared_journal_is_never_used():
    """Review: a snapshot taken with unshipped rows made a later recovery skip a new owner's rows."""
    coord = MemoryCoordinator()
    lease = await coord.acquire("ag", "w1", 60)
    rows = [{"seq": 1, "ts": 1.0, "op": "create", "block_id": "b", "payload": {
        "kind": "projet", "title": "B", "content": "v1", "pinned": False, "version": 1, "active": True}}]
    await coord.append(lease, rows)
    await coord.put_snapshot(lease, 5, {"seq": 5, "blocks": {"b": {"kind": "projet", "title": "B", "content": "lost",
                             "pinned": False, "version": 9, "active": True, "modified_at": 1.0}}})

    class Store:
        content = None
        def max_seq(self): return 0
        def unsynced(self, n): return []
        def unsynced_count(self): return 0
        db = type("DB", (), {"execute": lambda *a: type("R", (), {"fetchall": lambda s: []})()})()
        def rebuild(self, blocks, vectors, journal): Store.content = blocks["b"]["content"]

    class Emb:
        async def embed(self, texts, kind):
            import numpy as np
            return np.ones((len(texts), 4), dtype="float32") / 2

    out = await recover(coord, "ag", Store(), Emb())
    assert out["from_snapshot"] == 0 and Store.content == "v1"


async def test_a_stale_owner_cannot_publish_a_snapshot():
    coord = MemoryCoordinator(clock=lambda: 0.0)
    old = await coord.acquire("ag", "w1", 10)
    coord.clock = lambda: 100.0
    await coord.acquire("ag", "w2", 10)
    with pytest.raises(StaleEpoch):
        await coord.put_snapshot(old, 1, {"seq": 1, "blocks": {}})


async def test_one_cap_holds_for_every_worker_of_a_run():
    """Review: one budget per engine made the real cap agents x workers x 2 USD."""
    coord = MemoryCoordinator()
    a, b = SharedBudget(1.0, coord=coord, run_id="r"), SharedBudget(1.0, coord=coord, run_id="r")
    await a.agrant(0.6, "x")
    with pytest.raises(BudgetExceeded):
        await b.agrant(0.6, "y")
    await a.arecord(0.1, 0.6, "x")  # settled lower: room is given back
    await b.agrant(0.6, "y")


def test_paid_remote_embeddings_are_refused_without_a_price(tmp_path):
    """Review: Gemini embeddings were paid calls outside the budget guard."""
    with pytest.raises(ValueError, match="gemini embeddings"):
        Engine(_cfg(tmp_path, embedder="gemini"), RunRecorder(tmp_path / "r", "x"))


async def test_a_tool_runs_on_the_arguments_its_cache_key_is_made_of():
    """Review: weather was drawn on the raw city and '1 2+3' was served as '12+3'."""
    import sqlite3
    tools = default_tools(Bookings())
    runner = ToolRunner(tools, FaultConfig(latency_scale=0), clock=lambda: 1000.0)
    cache = ToolCache(sqlite3.connect(":memory:"), clock=lambda: 1000.0)
    w1 = await runner.run("weather", {"city": "abidjan"})
    await cache.astore(tools["weather"], {"city": "abidjan"}, w1)
    assert (await cache.alookup(tools["weather"], {"city": "Abidjan"})).value == await runner.run("weather", {"city": "Abidjan"})
    assert (await cache.alookup(tools["calculator"], {"expression": "1 2+3"})).outcome == "miss"
    await cache.astore(tools["calculator"], {"expression": "12+3"}, {"value": 15})
    assert (await cache.alookup(tools["calculator"], {"expression": "1 2+3"})).outcome == "miss"


async def test_a_turn_never_waits_for_the_cold_path_nor_creates_a_cache(tmp_path):
    """The invariant had no test: a blocked cold task must not delay a turn."""
    eng = Engine(_cfg(tmp_path, cache_admission="always"), RunRecorder(tmp_path / "r", "x"), session="x")
    created = []
    real = eng.prefix_cache.ensure

    async def counting(*a, **k):
        created.append(1)
        return await real(*a, **k)
    eng.prefix_cache.ensure = counting
    gate = asyncio.Event()

    async def blocked_extraction():
        await gate.wait()
        return {}
    eng._extract_pending = blocked_extraction
    await eng.start()
    steps = [json.loads(x) for x in SCENARIO.read_text().splitlines() if x.strip()]
    await eng.seed(steps[0]["seed"])
    for s in steps[1:4]:
        before = len(created)
        await asyncio.wait_for(eng.turn(s["user"]), 5)
        assert len(created) == before, "a cache was created inside a turn"
    gate.set()
    await eng.close()
