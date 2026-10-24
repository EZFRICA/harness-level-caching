"""The distributed contract, offline: the in-memory coordinator follows the same rules as Postgres.

The same contract tests run against the real Postgres container when CAMA_LIVE_PG is set to a DSN.
"""

import json
import os
from pathlib import Path

import pytest

from cama.cluster.coordinator import MemoryCoordinator, PostgresCoordinator, StaleEpoch
from cama.cluster.service import NotOwner, Worker
from cama.config import Config

SCENARIO = Path(__file__).parent.parent / "scenarios" / "agentic_fr_v1.jsonl"


class Clock:
    def __init__(self):
        self.t = 1_000.0

    def __call__(self):
        return self.t


def _coords():
    yield "memory", lambda: MemoryCoordinator(clock=Clock())
    if os.environ.get("CAMA_LIVE_PG"):
        yield "postgres", lambda: PostgresCoordinator(os.environ["CAMA_LIVE_PG"])


@pytest.fixture(params=[name for name, _ in _coords()])
async def coord(request):
    make = dict(_coords())[request.param]
    c = make()
    await c.connect()
    if isinstance(c, PostgresCoordinator):  # isolate each test
        async with c.pool.acquire() as conn:
            await conn.execute("TRUNCATE leases, journal, snapshots, prefix_caches, tool_cache, embed_cache, "
                               "workers, events")
    yield c
    await c.close()


def _ttl(coord):
    return 1.0 if isinstance(coord, PostgresCoordinator) else 10


async def _expire(coord, lease):
    """Let a lease run out: move the fake clock, or really wait on Postgres."""
    if isinstance(coord, MemoryCoordinator):
        coord.clock.t = lease.until + 1
    else:
        import asyncio
        import time
        await asyncio.sleep(max(0.0, lease.until - time.time()) + 0.3)


async def test_one_writer_per_agent_and_a_takeover_increments_the_epoch(coord):
    a = await coord.acquire("ag", "w1", _ttl(coord))
    assert a.epoch == 1 and await coord.acquire("ag", "w2", 10) is None
    assert (await coord.acquire("ag", "w1", _ttl(coord))).epoch == 1, "re-acquiring one's own lease keeps the epoch"
    await _expire(coord, await coord.lease_of("ag"))
    b = await coord.acquire("ag", "w2", 10)
    assert b is not None and b.epoch == 2 and b.worker == "w2"


async def test_a_stale_writer_is_fenced_and_appends_are_idempotent(coord):
    a = await coord.acquire("ag", "w1", _ttl(coord))
    row = {"seq": 1, "ts": 1.0, "op": "create", "block_id": "b", "payload": {"x": 1}}
    assert await coord.append(a, [row]) == 1
    assert await coord.append(a, [row]) == 0, "re-sending a row adds nothing"
    await _expire(coord, a)
    b = await coord.acquire("ag", "w2", 10)
    with pytest.raises(StaleEpoch):
        await coord.append(a, [{**row, "seq": 2}])
    assert await coord.append(b, [{**row, "seq": 2}]) == 1
    assert await coord.renew(a, 10) is None, "the former owner learns it lost the agent"
    assert [r["seq"] for r in await coord.journal("ag")] == [1, 2]


async def test_an_expired_lease_nobody_took_can_be_renewed(coord):
    """A coordinator outage longer than the TTL must not, by itself, stop the owner."""
    a = await coord.acquire("ag", "w1", _ttl(coord))
    await _expire(coord, a)
    assert (await coord.renew(a, 10)).epoch == 1


def _cfg(tmp_path, name):
    return Config(data_dir=tmp_path / name / "d", runs_dir=tmp_path / "runs", archive_dir=tmp_path / "a",
                  tool_latency_scale=0, snapshot_every=10, cache_admission="always")


def _scenario():
    lines = [json.loads(x) for x in SCENARIO.read_text().splitlines() if x.strip()]
    return lines[0]["seed"], [x["user"] for x in lines[1:]]


async def test_a_dead_worker_is_taken_over_and_its_memory_rebuilt_from_the_shared_tier(tmp_path):
    clock = Clock()
    coord = MemoryCoordinator(clock=clock)
    seed, users = _scenario()
    w1 = Worker(_cfg(tmp_path, "w1"), "w1", coord, "t", lease_ttl_s=10)
    await w1.coord.connect()
    await w1.seed("ag", seed)
    for text in users[:12]:
        await w1.turn("ag", text)
        await w1.slots["ag"].engine.drain()
    before = w1.slots["ag"].engine.store.snapshot_data()
    old_lease = w1.slots["ag"].lease["lease"]

    w2 = Worker(_cfg(tmp_path, "w2"), "w2", coord, "t", lease_ttl_s=10)
    with pytest.raises(NotOwner):
        await w2.turn("ag", users[12])
    clock.t += 11  # w1 died without releasing: its lease expires
    out = await w2.turn("ag", users[12])
    slot = w2.slots["ag"]
    assert out["epoch"] == 2 and slot.recovery["action"] == "rebuilt"
    rebuilt = {k: {f: v[f] for f in ("kind", "content", "version")} for k, v in before["blocks"].items()}
    after = slot.engine.store.snapshot_data()
    assert {k: {f: after["blocks"][k][f] for f in ("kind", "content", "version")} for k in rebuilt} == rebuilt
    with pytest.raises(StaleEpoch):
        await coord.append(old_lease, [{"seq": 999, "ts": 0, "op": "update", "block_id": "x", "payload": {}}])
    await slot.engine.close()
    await w1.slots["ag"].engine.close()


async def test_tool_results_and_prefix_caches_are_shared_between_workers(tmp_path):
    coord = MemoryCoordinator()
    seed, users = _scenario()
    w1 = Worker(_cfg(tmp_path, "w1"), "w1", coord, "t", lease_ttl_s=60)
    w2 = Worker(_cfg(tmp_path, "w2"), "w2", coord, "t", lease_ttl_s=60)
    await w1.seed("a1", seed)
    await w2.seed("a2", seed)
    weather = "Quelle est la météo prévue à Abidjan pour mon arrivée ?"
    await w1.turn("a1", weather)
    await w1.slots["a1"].engine.drain()
    out = await w2.turn("a2", weather)
    assert out["h3"]["hits"] == 1, "w2 reused the weather w1 fetched"
    await w2.slots["a2"].engine.drain()
    out = await w2.turn("a2", users[1])
    assert out["prefix_cache"]["hit"]
    for w, a in ((w1, "a1"), (w2, "a2")):
        await w.slots[a].engine.close()


async def test_the_http_service_accepts_a_turn_body(tmp_path):
    """Measured: the first live load got 216 errors out of 216; the body was read as a query parameter."""
    import httpx
    from cama.cluster.service import create_app

    w = Worker(_cfg(tmp_path, "w"), "w", MemoryCoordinator(), "t", lease_ttl_s=60)
    app = create_app(w)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w") as c:
        r = await c.post("/agents/a/turn", json={"text": "bonjour"})
    assert r.status_code == 200, r.text
    assert r.json()["worker"] == "w" and r.json()["epoch"] == 1
    await w.slots["a"].engine.close()


def test_engines_of_one_process_share_one_embedding_model(tmp_path):
    """Measured: each agent loaded its own ONNX model; three agents put a 1 GB worker out of memory."""
    from cama.engine import Engine
    from cama.metrics import RunRecorder

    a = Engine(_cfg(tmp_path, "a"), RunRecorder(tmp_path / "r", "a"), session="a")
    b = Engine(_cfg(tmp_path, "b"), RunRecorder(tmp_path / "r", "b"), session="b")
    assert a.embedder.inner is b.embedder.inner


async def test_replication_keeps_up_under_back_to_back_turns(tmp_path):
    """Measured: with turns arriving back to back the idle gated sync never ran; a crash lost 26 rows."""
    import asyncio
    from cama.engine import Engine
    from cama.metrics import RunRecorder

    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 tool_latency_scale=0, sync_interval_s=0.05, echo_base_ms=20)
    eng = Engine(cfg, RunRecorder(cfg.runs_dir, "x"), session="x")
    await eng.start()
    seed, users = _scenario()
    await eng.seed(seed)
    worst = 0
    for text in users[:10]:
        await eng.turn(text)  # no drain: the next turn starts at once
        await asyncio.sleep(0.06)
        worst = max(worst, len(eng.store.unsynced(10_000)))
    await eng.close()
    assert worst < 20, f"unsynced rows grew to {worst}"


async def test_extraction_coalesces_when_the_cold_path_falls_behind(tmp_path):
    from cama.engine import Engine
    from cama.metrics import RunRecorder

    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 tool_latency_scale=0, echo_base_ms=30)
    rec = RunRecorder(cfg.runs_dir, "x")
    eng = Engine(cfg, rec, session="x")
    await eng.start()
    seed, users = _scenario()
    await eng.seed(seed)
    for text in users[:8]:
        await eng.turn(text)
    await eng.drain()
    await eng.close()
    tasks = [json.loads(x) for x in (cfg.runs_dir / "x" / "tasks.jsonl").read_text().splitlines()]
    ext = [t for t in tasks if t["task"].startswith("extract")]
    assert sum(t["exchanges"] for t in ext) == 8 and len(ext) < 8, "every exchange extracted, in fewer calls"


async def test_a_local_store_unknown_to_the_shared_tier_is_rebuilt_from_it(tmp_path):
    """Measured: a worker's volume kept a previous run's store; it trusted it over an empty shared tier."""
    from cama.cluster.recovery import recover
    from cama.engine import Engine
    from cama.metrics import RunRecorder

    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a")
    eng = Engine(cfg, RunRecorder(cfg.runs_dir, "x"), session="ag")
    seed, _ = _scenario()
    await eng.seed(seed)
    eng.store.mark_synced([r["seq"] for r in eng.store.unsynced()])  # replicated somewhere else, once
    out = await recover(MemoryCoordinator(), "ag", eng.store, eng.embedder)
    assert out["action"] == "rebuilt" and out["diverged"] and eng.store.count()["blocks"] == 0
    await eng.close()


async def test_a_saturated_worker_answers_busy_at_once_instead_of_queueing(tmp_path):
    """Measured: 16 agents on one worker, nothing shed, 0 of 192 requests answered."""
    import asyncio
    import httpx
    from cama.cluster.service import create_app

    w = Worker(_cfg(tmp_path, "w"), "w", MemoryCoordinator(), "t", lease_ttl_s=60, max_inflight=1)
    gate = asyncio.Event()
    real = w._turn

    async def slow(agent_id, text):
        await gate.wait()
        return await real(agent_id, text)
    w._turn = slow
    app = create_app(w)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w") as c:
        first = asyncio.create_task(c.post("/agents/a/turn", json={"text": "bonjour"}))
        await asyncio.sleep(0.05)
        second = await c.post("/agents/b/turn", json={"text": "bonjour"})
        assert second.status_code == 503 and second.headers["Retry-After"] == "1"
        gate.set()
        assert (await first).status_code == 200
    assert w.shed == 1
    await w.slots["a"].engine.close()
