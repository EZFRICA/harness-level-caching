"""The shared tier (decision 0007): leases with epochs, the replicated journal, snapshots and
the shared cache registries.

Two implementations of one contract: `PostgresCoordinator` (asyncpg, the real one) and
`MemoryCoordinator` (the same rules in a dict, for offline tests). The contract tests run
against both; the Postgres run needs `HC_LIVE_PG=1` and a database.

Rules:
- One writer per agent. `acquire` succeeds when the lease is free, expired, or already ours.
  Taking over from another worker increments the epoch.
- A lease is lost only when someone else took it: renewing an expired lease that nobody took
  succeeds, so a Postgres outage does not by itself stop a worker (it cannot be taken over
  either, since takeover needs Postgres).
- Fencing: a journal append carrying an epoch lower than the agent's current one is refused,
  so a paused or partitioned former owner cannot overwrite the new one.
- Appends are idempotent on (agent, seq).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


class StaleEpoch(RuntimeError):
    retryable = False


@dataclass(frozen=True)
class Lease:
    agent_id: str
    worker: str
    epoch: int
    until: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (agent_id TEXT PRIMARY KEY, worker TEXT NOT NULL, epoch BIGINT NOT NULL,
    until DOUBLE PRECISION NOT NULL);
CREATE TABLE IF NOT EXISTS journal (agent_id TEXT NOT NULL, seq BIGINT NOT NULL, epoch BIGINT NOT NULL,
    ts DOUBLE PRECISION NOT NULL, op TEXT NOT NULL, block_id TEXT NOT NULL, payload JSONB NOT NULL,
    worker TEXT NOT NULL, PRIMARY KEY (agent_id, seq));
CREATE TABLE IF NOT EXISTS snapshots (agent_id TEXT NOT NULL, seq BIGINT NOT NULL, ts DOUBLE PRECISION NOT NULL,
    data JSONB NOT NULL, PRIMARY KEY (agent_id, seq));
CREATE TABLE IF NOT EXISTS prefix_caches (hash TEXT NOT NULL, model TEXT NOT NULL, name TEXT NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL, tokens INT NOT NULL, worker TEXT NOT NULL, PRIMARY KEY (hash, model));
CREATE TABLE IF NOT EXISTS tool_cache (key TEXT PRIMARY KEY, value JSONB NOT NULL, expires_at DOUBLE PRECISION,
    worker TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, space TEXT NOT NULL, vector BYTEA NOT NULL);
CREATE TABLE IF NOT EXISTS budgets (run_id TEXT PRIMARY KEY, cap DOUBLE PRECISION NOT NULL,
    spent DOUBLE PRECISION NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS workers (worker TEXT PRIMARY KEY, seen DOUBLE PRECISION NOT NULL, info JSONB NOT NULL);
CREATE TABLE IF NOT EXISTS events (id BIGSERIAL PRIMARY KEY, ts DOUBLE PRECISION NOT NULL, worker TEXT NOT NULL,
    kind TEXT NOT NULL, agent_id TEXT, detail JSONB NOT NULL);
"""


class PostgresCoordinator:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None

    async def connect(self, create_schema: bool = True) -> None:
        import asyncpg

        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=8, timeout=5, command_timeout=5)
        if not create_schema:  # read-only clients (the dashboard) never run DDL
            return
        async with self.pool.acquire() as c:
            # Concurrent CREATE TABLE IF NOT EXISTS races in Postgres; workers start together.
            await c.execute("SELECT pg_advisory_lock(7342001)")
            try:
                await c.execute(SCHEMA)
            finally:
                await c.execute("SELECT pg_advisory_unlock(7342001)")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def acquire(self, agent_id: str, worker: str, ttl_s: float) -> Lease | None:
        now = time.time()
        async with self.pool.acquire() as c, c.transaction():
            row = await c.fetchrow("SELECT worker, epoch, until FROM leases WHERE agent_id=$1 FOR UPDATE", agent_id)
            if row is None:
                epoch = 1
                await c.execute("INSERT INTO leases VALUES ($1,$2,$3,$4)", agent_id, worker, epoch, now + ttl_s)
            elif row["worker"] == worker:
                epoch = row["epoch"]
                await c.execute("UPDATE leases SET until=$2 WHERE agent_id=$1", agent_id, now + ttl_s)
            elif row["until"] < now:
                epoch = row["epoch"] + 1
                await c.execute("UPDATE leases SET worker=$2, epoch=$3, until=$4 WHERE agent_id=$1",
                                agent_id, worker, epoch, now + ttl_s)
            else:
                return None
        return Lease(agent_id, worker, epoch, now + ttl_s)

    async def renew(self, lease: Lease, ttl_s: float) -> Lease | None:
        now = time.time()
        async with self.pool.acquire() as c:
            done = await c.execute("UPDATE leases SET until=$4 WHERE agent_id=$1 AND worker=$2 AND epoch=$3",
                                   lease.agent_id, lease.worker, lease.epoch, now + ttl_s)
        return Lease(lease.agent_id, lease.worker, lease.epoch, now + ttl_s) if done.endswith(" 1") else None

    async def release(self, lease: Lease) -> None:
        async with self.pool.acquire() as c:
            await c.execute("UPDATE leases SET until=0 WHERE agent_id=$1 AND worker=$2 AND epoch=$3",
                            lease.agent_id, lease.worker, lease.epoch)

    async def lease_of(self, agent_id: str) -> Lease | None:
        async with self.pool.acquire() as c:
            r = await c.fetchrow("SELECT worker, epoch, until FROM leases WHERE agent_id=$1", agent_id)
        return Lease(agent_id, r["worker"], r["epoch"], r["until"]) if r else None

    async def append(self, lease: Lease, rows: list[dict]) -> int:
        async with self.pool.acquire() as c, c.transaction():
            cur = await c.fetchval("SELECT epoch FROM leases WHERE agent_id=$1 FOR SHARE", lease.agent_id)
            if cur is not None and lease.epoch < cur:
                raise StaleEpoch(f"{lease.agent_id}: epoch {lease.epoch} < current {cur}")
            n = 0
            for r in rows:
                done = await c.execute(
                    "INSERT INTO journal VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8) ON CONFLICT DO NOTHING",
                    lease.agent_id, r["seq"], lease.epoch, r["ts"], r["op"], r["block_id"],
                    json.dumps(r["payload"]), lease.worker)
                n += done.endswith(" 1")
            return n

    async def journal(self, agent_id: str, after_seq: int = 0) -> list[dict]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("SELECT seq, ts, op, block_id, payload FROM journal WHERE agent_id=$1 AND seq>$2 "
                                 "ORDER BY seq", agent_id, after_seq)
        return [{"seq": r["seq"], "ts": r["ts"], "op": r["op"], "block_id": r["block_id"],
                 "payload": json.loads(r["payload"])} for r in rows]

    async def put_snapshot(self, lease: Lease, seq: int, data: dict) -> None:
        """Fenced like an append: a former owner cannot publish a snapshot of its stale state."""
        async with self.pool.acquire() as c, c.transaction():
            cur = await c.fetchval("SELECT epoch FROM leases WHERE agent_id=$1 FOR SHARE", lease.agent_id)
            if cur is not None and lease.epoch < cur:
                raise StaleEpoch(f"{lease.agent_id}: snapshot epoch {lease.epoch} < current {cur}")
            await c.execute("INSERT INTO snapshots VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT DO NOTHING",
                            lease.agent_id, seq, time.time(), json.dumps(data))

    async def max_seq(self, agent_id: str) -> int:
        async with self.pool.acquire() as c:
            return await c.fetchval("SELECT COALESCE(MAX(seq), 0) FROM journal WHERE agent_id=$1", agent_id)

    async def reserve(self, run_id: str, cap: float, usd: float) -> bool:
        """Atomically reserve spend against a run's cap shared by every worker."""
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO budgets (run_id, cap) VALUES ($1,$2) ON CONFLICT DO NOTHING", run_id, cap)
            row = await c.fetchrow("UPDATE budgets SET spent = spent + $2 WHERE run_id=$1 AND spent + $2 <= cap "
                                   "RETURNING spent", run_id, usd)
        return row is not None

    async def adjust(self, run_id: str, delta: float) -> None:
        async with self.pool.acquire() as c:
            await c.execute("UPDATE budgets SET spent = GREATEST(0, spent + $2) WHERE run_id=$1", run_id, delta)

    async def latest_snapshot(self, agent_id: str) -> tuple[int, dict] | None:
        async with self.pool.acquire() as c:
            r = await c.fetchrow("SELECT seq, data FROM snapshots WHERE agent_id=$1 ORDER BY seq DESC LIMIT 1", agent_id)
        return (r["seq"], json.loads(r["data"])) if r else None

    async def put_prefix(self, h: str, model: str, name: str, expires_at: float, tokens: int, worker: str) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO prefix_caches VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (hash, model) DO UPDATE "
                            "SET name=excluded.name, expires_at=excluded.expires_at, tokens=excluded.tokens, "
                            "worker=excluded.worker", h, model, name, expires_at, tokens, worker)

    async def get_prefix(self, h: str, model: str) -> dict | None:
        async with self.pool.acquire() as c:
            r = await c.fetchrow("SELECT name, expires_at, tokens, worker FROM prefix_caches WHERE hash=$1 AND model=$2",
                                 h, model)
        return dict(r) if r else None

    async def get_tool(self, key: str) -> dict | None:
        async with self.pool.acquire() as c:
            r = await c.fetchrow("SELECT value, expires_at, worker FROM tool_cache WHERE key=$1", key)
        if r is None or (r["expires_at"] is not None and r["expires_at"] <= time.time()):
            return None
        return {"value": json.loads(r["value"]), "worker": r["worker"], "expires_at": r["expires_at"]}

    async def put_tool(self, key: str, value: dict, expires_at: float | None, worker: str) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO tool_cache VALUES ($1,$2::jsonb,$3,$4) ON CONFLICT (key) DO UPDATE SET "
                            "value=excluded.value, expires_at=excluded.expires_at, worker=excluded.worker",
                            key, json.dumps(value), expires_at, worker)

    async def get_embedding(self, key: str) -> bytes | None:
        async with self.pool.acquire() as c:
            return await c.fetchval("SELECT vector FROM embed_cache WHERE key=$1", key)

    async def put_embedding(self, key: str, space: str, vector: bytes) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO embed_cache VALUES ($1,$2,$3) ON CONFLICT DO NOTHING", key, space, vector)

    async def heartbeat(self, worker: str, info: dict) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO workers VALUES ($1,$2,$3::jsonb) ON CONFLICT (worker) DO UPDATE SET "
                            "seen=excluded.seen, info=excluded.info", worker, time.time(), json.dumps(info))

    async def event(self, worker: str, kind: str, agent_id: str | None, detail: dict) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO events (ts, worker, kind, agent_id, detail) VALUES ($1,$2,$3,$4,$5::jsonb)",
                            time.time(), worker, kind, agent_id, json.dumps(detail))

    async def view(self) -> dict:
        """Read-only state for the dashboard."""
        async with self.pool.acquire() as c:
            workers = [dict(r) for r in await c.fetch("SELECT worker, seen, info FROM workers ORDER BY worker")]
            leases = [dict(r) for r in await c.fetch("SELECT agent_id, worker, epoch, until FROM leases ORDER BY agent_id")]
            counts = dict(await c.fetchrow(
                "SELECT (SELECT count(*) FROM journal) journal, (SELECT count(*) FROM prefix_caches) prefixes, "
                "(SELECT count(*) FROM tool_cache) tools, (SELECT count(*) FROM embed_cache) embeddings"))
            events = [dict(r) for r in await c.fetch(
                "SELECT ts, worker, kind, agent_id, detail FROM events ORDER BY id DESC LIMIT 50")]
        for w in workers:
            w["info"] = json.loads(w["info"])
        for e in events:
            e["detail"] = json.loads(e["detail"])
        return {"workers": workers, "leases": leases, "counts": counts, "events": events}


@dataclass
class MemoryCoordinator:
    """The same contract in memory, with an injectable clock: the offline twin of Postgres."""

    clock: object = time.time
    leases: dict = field(default_factory=dict)
    journal_rows: dict = field(default_factory=dict)  # (agent, seq) -> row
    snapshots: dict = field(default_factory=dict)
    prefixes: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    embeddings: dict = field(default_factory=dict)
    workers: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    budgets: dict = field(default_factory=dict)
    down: bool = False  # simulate an outage

    def _check(self) -> None:
        if self.down:
            raise ConnectionError("coordinator unavailable")

    async def connect(self) -> None:
        self._check()

    async def close(self) -> None:
        pass

    async def acquire(self, agent_id, worker, ttl_s):
        self._check()
        now = self.clock()
        cur = self.leases.get(agent_id)
        if cur is None:
            epoch = 1
        elif cur.worker == worker:
            epoch = cur.epoch
        elif cur.until < now:
            epoch = cur.epoch + 1
        else:
            return None
        self.leases[agent_id] = Lease(agent_id, worker, epoch, now + ttl_s)
        return self.leases[agent_id]

    async def renew(self, lease, ttl_s):
        self._check()
        cur = self.leases.get(lease.agent_id)
        if cur is None or cur.worker != lease.worker or cur.epoch != lease.epoch:
            return None
        self.leases[lease.agent_id] = Lease(lease.agent_id, lease.worker, lease.epoch, self.clock() + ttl_s)
        return self.leases[lease.agent_id]

    async def release(self, lease):
        self._check()
        cur = self.leases.get(lease.agent_id)
        if cur and cur.worker == lease.worker and cur.epoch == lease.epoch:
            self.leases[lease.agent_id] = Lease(cur.agent_id, cur.worker, cur.epoch, 0.0)

    async def lease_of(self, agent_id):
        self._check()
        return self.leases.get(agent_id)

    async def append(self, lease, rows):
        self._check()
        cur = self.leases.get(lease.agent_id)
        if cur is not None and lease.epoch < cur.epoch:
            raise StaleEpoch(f"{lease.agent_id}: epoch {lease.epoch} < current {cur.epoch}")
        n = 0
        for r in rows:
            k = (lease.agent_id, r["seq"])
            if k not in self.journal_rows:
                self.journal_rows[k] = {**r, "epoch": lease.epoch, "worker": lease.worker}
                n += 1
        return n

    async def journal(self, agent_id, after_seq=0):
        self._check()
        rows = [r for (a, s), r in self.journal_rows.items() if a == agent_id and s > after_seq]
        return [{k: r[k] for k in ("seq", "ts", "op", "block_id", "payload")} for r in sorted(rows, key=lambda r: r["seq"])]

    async def put_snapshot(self, lease, seq, data):
        self._check()
        cur = self.leases.get(lease.agent_id)
        if cur is not None and lease.epoch < cur.epoch:
            raise StaleEpoch(f"{lease.agent_id}: snapshot epoch {lease.epoch} < current {cur.epoch}")
        self.snapshots.setdefault(lease.agent_id, {}).setdefault(seq, json.loads(json.dumps(data)))

    async def max_seq(self, agent_id):
        self._check()
        return max((s for (a, s) in self.journal_rows if a == agent_id), default=0)

    async def reserve(self, run_id, cap, usd):
        self._check()
        b = self.budgets.setdefault(run_id, {"cap": cap, "spent": 0.0})
        if b["spent"] + usd > b["cap"]:
            return False
        b["spent"] += usd
        return True

    async def adjust(self, run_id, delta):
        self._check()
        if run_id in self.budgets:
            self.budgets[run_id]["spent"] = max(0.0, self.budgets[run_id]["spent"] + delta)

    async def latest_snapshot(self, agent_id):
        self._check()
        snaps = self.snapshots.get(agent_id)
        if not snaps:
            return None
        seq = max(snaps)
        return seq, snaps[seq]

    async def put_prefix(self, h, model, name, expires_at, tokens, worker):
        self._check()
        self.prefixes[(h, model)] = {"name": name, "expires_at": expires_at, "tokens": tokens, "worker": worker}

    async def get_prefix(self, h, model):
        self._check()
        return self.prefixes.get((h, model))

    async def get_tool(self, key):
        self._check()
        r = self.tools.get(key)
        if r is None or (r["expires_at"] is not None and r["expires_at"] <= self.clock()):
            return None
        return {"value": r["value"], "worker": r["worker"], "expires_at": r["expires_at"]}

    async def put_tool(self, key, value, expires_at, worker):
        self._check()
        self.tools[key] = {"value": value, "expires_at": expires_at, "worker": worker}

    async def get_embedding(self, key):
        self._check()
        return self.embeddings.get(key)

    async def put_embedding(self, key, space, vector):
        self._check()
        self.embeddings.setdefault(key, vector)

    async def heartbeat(self, worker, info):
        self._check()
        self.workers[worker] = {"seen": self.clock(), "info": info}

    async def event(self, worker, kind, agent_id, detail):
        self._check()
        self.events.append({"ts": self.clock(), "worker": worker, "kind": kind, "agent_id": agent_id, "detail": detail})
