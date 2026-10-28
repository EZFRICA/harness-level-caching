"""H2 (embeddings), H3 (tool results) and the agent loop."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness_cache.adapters.local import EchoLLM, HashEmbedder
from harness_cache.agent.loop import AgentLoop
from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.memory.embed_cache import CachedEmbedder
from harness_cache.metrics import RunRecorder
from harness_cache.teu.cache import ToolCache
from harness_cache.teu.tools import Bookings, FaultConfig, ToolRunner, default_tools
from harness_cache.types import Message

SCENARIO = Path(__file__).parent.parent / "scenarios" / "agentic_fr_v1.jsonl"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _h3(enabled=True, clock=None):
    clock = clock or Clock()
    bookings = Bookings()
    tools = default_tools(bookings)
    return tools, ToolCache(sqlite3.connect(":memory:"), enabled=enabled, clock=clock), bookings, clock


def test_a_pure_tool_is_served_from_cache_after_its_first_call():
    tools, cache, _, _ = _h3()
    calc = tools["calculator"]
    assert cache.lookup(calc, {"expression": "12*75"}).cause == "cold"
    cache.store(calc, {"expression": "12*75"}, {"value": 900})
    hit = cache.lookup(calc, {"expression": " 12 * 75 "})
    assert hit.outcome == "hit" and hit.value == {"value": 900}, "normalised arguments share a key"


def test_a_ttl_tool_expires_exactly_at_its_ttl():
    tools, cache, _, clock = _h3()
    fx = tools["fx_rate"]
    cache.store(fx, {"base": "eur", "quote": "xof"}, {"rate": 655.957})
    clock.t += fx.ttl_s - 0.001
    assert cache.lookup(fx, {"base": "EUR", "quote": "XOF"}).outcome == "hit"
    clock.t += 0.001
    assert cache.lookup(fx, {"base": "EUR", "quote": "XOF"}).cause == "expired"


def test_a_side_effect_tool_is_never_stored_nor_served():
    tools, cache, _, _ = _h3()
    book = tools["book_hotel"]
    cache.store(book, {"city": "Abidjan", "nights": 3}, {"ref": "BK0001"})
    look = cache.lookup(book, {"city": "Abidjan", "nights": 3})
    assert look.outcome == "bypass" and look.cause == "side_effect" and look.value is None


def test_a_disabled_tool_cache_bypasses_every_lookup():
    tools, cache, _, _ = _h3(enabled=False)
    cache.store(tools["calculator"], {"expression": "1+1"}, {"value": 2})
    assert cache.lookup(tools["calculator"], {"expression": "1+1"}).cause == "disabled"


@settings(max_examples=150, deadline=None)
@given(a=st.integers(-50, 50), b=st.integers(1, 50), op=st.sampled_from("+-*/"), dt=st.floats(0, 5000))
def test_a_cache_hit_returns_what_a_fresh_call_computed_when_it_was_stored(a, b, op, dt):
    """Pure: identical forever. TTL: identical to the value computed at fill time, never older than the TTL."""
    import asyncio
    tools, cache, _, clock = _h3()
    runner = ToolRunner(tools, FaultConfig(latency_scale=0), clock=clock)
    for name, args in (("calculator", {"expression": f"{a}{op}{b}"}), ("weather", {"city": "abidjan"})):
        fresh = asyncio.run(runner.run(name, args))
        cache.store(tools[name], args, fresh)
        clock.t += dt
        look = cache.lookup(tools[name], args)
        if tools[name].cache_class == "pure" or dt < tools[name].ttl_s:
            assert look.outcome == "hit" and look.value == json.loads(json.dumps(fresh))
        else:
            assert look.outcome == "miss"
        clock.t -= dt


async def test_embeddings_are_reused_within_their_space_and_survive_a_restart():
    db = sqlite3.connect(":memory:")
    emb = CachedEmbedder(HashEmbedder(), db)
    v1 = await emb.embed(["bonjour Abidjan", "autre texte"], "document")
    v2 = await emb.embed(["bonjour Abidjan"], "document")
    assert emb.counters() == (1, 2) and np.array_equal(v1[0], v2[0])
    await emb.embed(["bonjour Abidjan"], "query")
    assert emb.misses == 3, "query and document are different keys"
    again = CachedEmbedder(HashEmbedder(), db)
    await again.embed(["autre texte"], "document")
    assert again.counters() == (1, 0)
    seeded = CachedEmbedder(HashEmbedder(), db)
    await seeded.embed(["bonjour Abidjan", "autre texte"], "document")
    assert seeded.counters() == (2, 0), "a restarted worker re-seeding the same blocks pays nothing"
    other_space = CachedEmbedder(HashEmbedder(dim=128), db)
    await other_space.embed(["autre texte"], "document")
    assert other_space.counters() == (0, 1), "another space never reads these vectors"


async def test_failing_tools_are_retried_then_reported_to_the_model_and_the_turn_completes():
    tools, cache, _, _ = _h3()
    runner = ToolRunner(tools, FaultConfig(error_rate=1.0, latency_scale=0))
    loop = AgentLoop(EchoLLM(), runner, cache, max_steps=3)
    res = await loop.run("sys", [Message("user", "Quelle est la météo à Abidjan ?")],
                         [t.spec for t in tools.values()], None)
    assert res.tools[0]["outcome"] == "error" and res.tools[0]["attempts"] == 2
    assert res.stopped == "answer" and "error" in res.text


@pytest.mark.parametrize("tool_cache", [True, False])
async def test_agentic_scenario_end_to_end(tmp_path, tool_cache):
    cfg = Config(mode="extract", data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 tool_cache=tool_cache, tool_latency_scale=0, cache_admission="always")
    rec = RunRecorder(cfg.runs_dir, "agentic")
    rec.manifest(cfg.as_dict())
    eng = Engine(cfg, rec, session="t")
    await eng.start()
    steps = [json.loads(x) for x in SCENARIO.read_text().splitlines() if x.strip()]
    await eng.seed(steps[0]["seed"])
    bookings_asked = sum("serve" in s["user"].lower() for s in steps[1:])
    for s in steps[1:]:
        await eng.turn(s["user"])
        await eng.drain()
    made = len(eng.bookings.made)
    await eng.close()
    turns = [json.loads(x) for x in (cfg.runs_dir / "agentic" / "turns.jsonl").read_text().splitlines()]
    h3 = [t["h3"] for t in turns]
    assert made == bookings_asked, "every booking request made a booking, cached or not"
    assert sum(h["bypass"] for h in h3) >= bookings_asked
    assert (sum(h["hits"] for h in h3) > 0) == tool_cache
    assert any(t["llm"]["calls"] >= 2 for t in turns), "tool turns take several model calls"
    assert sum(t["prefix_cache"]["hit"] for t in turns) > 0
    assert sum(t["h2"]["misses"] for t in turns) > 0, "query embeddings are counted from the start of the turn"


async def test_a_prefix_cache_is_created_only_once_the_prefix_was_sent_unchanged(tmp_path):
    """Measured: creating on every memory change made 200 caches in 200 turns; topk made 198 for 2 hits."""
    from harness_cache.memory.extractor import ExtractionReport

    async def stable_memory(batch):
        return ExtractionReport()

    async def run(name, freeze):
        cfg = Config(mode="extract", data_dir=tmp_path / name, runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                     tool_latency_scale=0)
        eng = Engine(cfg, RunRecorder(cfg.runs_dir, name), session=name)
        if freeze:
            eng.extractor.run_many = stable_memory
        await eng.start()
        steps = [json.loads(x) for x in SCENARIO.read_text().splitlines() if x.strip()]
        await eng.seed(steps[0]["seed"])
        hits = []
        for s in steps[1:6]:
            hits.append((await eng.turn(s["user"])).record["prefix_cache"]["hit"])
            await eng.drain()
        creations = eng.prefix_cache.creations
        await eng.close()
        return hits, creations

    hits, creations = await run("stable", freeze=True)
    assert hits == [False, True, True, True, True] and creations == 1
    hits, creations = await run("churning", freeze=False)
    assert creations == 0 and not any(hits), "memory rewritten every turn never earns a cache"


def test_generated_scenarios_are_reproducible_and_shaped_as_announced():
    from harness_cache.stress.gen import SUBJECTS, generate
    a, b = generate("zipf", 120, seed=5), generate("zipf", 120, seed=5)
    assert a == b and generate("zipf", 120, seed=6) != a
    churn = generate("churn", len(SUBJECTS) * 2)
    users = [s["user"] for s in churn[1:]]
    assert len({u.split()[-1] for u in users[: len(SUBJECTS)]}) > len(SUBJECTS) // 2, "churn changes topic every turn"
    assert all(not b.get("active", True) for b in churn[0]["seed"][1:]), "every topic starts in L3, so it page faults"
