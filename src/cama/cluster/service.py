"""A CAMA worker: serves agents over HTTP, one writer per agent through a lease.

    POST /agents/{agent_id}/turn   {"text": "..."}  -> the answer and the turn's record
    POST /agents/{agent_id}/seed   {"blocks": [...]}
    GET  /healthz

On the first request for an agent the worker acquires its lease (409 with the owner if
another live worker holds it), rebuilds the local store from the shared tier when it is
behind (recovery), then serves. A background task renews leases and drops any agent whose
lease was taken over; it also sends a heartbeat for the dashboard.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from dataclasses import dataclass

from pydantic import BaseModel

from cama.cluster.recovery import recover
from cama.config import Config
from cama.engine import Engine
from cama.logging import get_logger
from cama.metrics import RunRecorder, rss_mb

log = get_logger(__name__)


class TurnIn(BaseModel):  # module level: FastAPI resolves string annotations from module globals
    text: str


class SeedIn(BaseModel):
    blocks: list[dict]


class Busy(RuntimeError):
    """The worker is at capacity; the client should retry after a short delay."""


class NotOwner(RuntimeError):
    def __init__(self, owner: str):
        super().__init__(f"agent owned by {owner}")
        self.owner = owner


@dataclass
class Slot:
    engine: Engine
    lease: object
    lock: asyncio.Lock
    recovery: dict


class Worker:
    def __init__(self, cfg: Config, worker_id: str, coord, run_id: str, lease_ttl_s: float = 6.0,
                 max_inflight: int = 12, max_opening: int = 2):
        self.cfg = cfg
        self.id = worker_id
        self.coord = coord
        self.run_id = run_id
        self.lease_ttl_s = lease_ttl_s
        self.slots: dict[str, Slot] = {}
        self._opening: dict[str, asyncio.Lock] = {}
        self._bg: asyncio.Task | None = None
        self.started = time.time()
        self.turns = 0
        # Admission control: past `max_inflight` requests the worker answers "busy" at once rather
        # than queue work it cannot finish; opening agents (recovery, seeding) is bounded too.
        self.max_inflight = max_inflight
        self.inflight = 0
        self.shed = 0
        self._opening_gate = asyncio.Semaphore(max_opening)

    async def start(self) -> None:
        # Load and warm the embedding model before serving, off the event loop. Measured: loading
        # it inside the first request blocked the loop for seconds, every coordinator call of the
        # worker timed out, and its agents were taken over by the other workers.
        from cama.adapters import make_embedder
        t = time.perf_counter()
        emb = await asyncio.to_thread(make_embedder, self.cfg)
        await emb.embed(["warm up"], "query")
        self.warmup_ms = (time.perf_counter() - t) * 1000
        await self.coord.connect()
        await self.coord.event(self.id, "worker_start", None, {"pid": os.getpid(), "warmup_ms": self.warmup_ms})
        self._bg = asyncio.create_task(self._maintain())

    async def stop(self) -> None:
        if self._bg:
            self._bg.cancel()
        for agent_id, slot in list(self.slots.items()):
            await self._drop(agent_id, release=True, reason="shutdown")
        await self.coord.close()

    async def slot(self, agent_id: str) -> Slot:
        s = self.slots.get(agent_id)
        if s is not None:
            return s
        lock = self._opening.setdefault(agent_id, asyncio.Lock())
        async with lock, self._opening_gate:
            if agent_id in self.slots:
                return self.slots[agent_id]
            lease = await self.coord.acquire(agent_id, self.id, self.lease_ttl_s)
            if lease is None:
                cur = await self.coord.lease_of(agent_id)
                raise NotOwner(cur.worker if cur else "unknown")
            cfg = dataclasses.replace(self.cfg, store_name=agent_id)
            rec = RunRecorder(cfg.runs_dir, f"{self.run_id}-{self.id}-{agent_id}")
            rec.manifest(cfg.as_dict(), {"worker": self.id, "agent": agent_id, "epoch": lease.epoch})
            holder = {"lease": lease}
            from cama.cost import SharedBudget
            budget = SharedBudget(cfg.max_usd, coord=self.coord, run_id=self.run_id)
            engine = Engine(cfg, rec, session=agent_id, shared=self.coord, worker=self.id,
                            lease=lambda h=holder: h["lease"], budget=budget)
            rec.note(engine.manifest_extra())
            try:
                recovery = await recover(self.coord, agent_id, engine.store, engine.embedder)
                if recovery["action"] == "rebuilt":
                    engine.active.load(engine.store.active_blocks())
                    engine.turn_index = len(engine.store.history(agent_id)) // 2
                # Recovery re-embeds blocks and may outlast part of the lease: renew before serving.
                renewed = await self.coord.renew(lease, self.lease_ttl_s)
                if renewed is None:
                    raise NotOwner("taken over during recovery")
                holder["lease"] = renewed
                await engine.start()
            except BaseException:
                await engine.close()  # no leaked store, recorder or tasks when recovery fails
                raise
            s = Slot(engine, holder, asyncio.Lock(), recovery)
            self.slots[agent_id] = s
            await self.coord.event(self.id, "acquire", agent_id, {"epoch": lease.epoch, "recovery": recovery})
            log.info("acquired %s epoch %d recovery %s", agent_id, lease.epoch, recovery)
            return s

    def admit(self) -> None:
        if self.inflight >= self.max_inflight:
            self.shed += 1
            raise Busy(f"{self.inflight} requests in flight")
        self.inflight += 1

    async def turn(self, agent_id: str, text: str) -> dict:
        self.admit()
        try:
            return await self._turn(agent_id, text)
        finally:
            self.inflight -= 1

    async def _turn(self, agent_id: str, text: str) -> dict:
        s = await self.slot(agent_id)
        async with s.lock:
            if self.slots.get(agent_id) is not s:  # dropped while this request waited for the lock
                raise NotOwner("dropped")
            res = await s.engine.turn(text)
        self.turns += 1
        r = res.record
        return {"text": res.text, "worker": self.id, "epoch": s.lease["lease"].epoch, "turn": r["turn"],
                "hot_ms": r["hot_ms"], "tier": (r["route"] or {}).get("tier"),
                "prefix_cache": r.get("prefix_cache"), "h3": r.get("h3"), "recovery": s.recovery["action"]}

    async def seed(self, agent_id: str, blocks: list[dict]) -> dict:
        self.admit()
        try:
            s = await self.slot(agent_id)
            async with s.lock:
                return await s.engine.seed(blocks)
        finally:
            self.inflight -= 1

    async def _drop(self, agent_id: str, release: bool, reason: str) -> None:
        s = self.slots.pop(agent_id, None)
        if s is None:
            return
        try:
            try:  # let a turn in progress finish rather than close the store under it
                await asyncio.wait_for(s.lock.acquire(), timeout=10)
                s.lock.release()
            except asyncio.TimeoutError:
                pass
            await s.engine.close()
        finally:
            if release:
                try:
                    await self.coord.release(s.lease["lease"])
                except Exception:  # noqa: BLE001: an unreleased lease expires anyway
                    pass
            try:
                await self.coord.event(self.id, "drop", agent_id, {"reason": reason})
            except Exception:  # noqa: BLE001
                pass

    async def _maintain(self) -> None:
        while True:
            await asyncio.sleep(self.lease_ttl_s / 3)
            for agent_id, s in list(self.slots.items()):
                try:
                    new = await self.coord.renew(s.lease["lease"], self.lease_ttl_s)
                except Exception:  # noqa: BLE001: coordinator down; nobody can take over either
                    continue
                try:
                    if new is None:
                        log.warning("lease of %s taken over; dropping it", agent_id)
                        await self._drop(agent_id, release=False, reason="lease taken over")
                    else:
                        s.lease["lease"] = new
                except Exception:  # noqa: BLE001: one agent must never stop the renewals of the others
                    log.exception("maintenance of %s failed", agent_id)
            try:
                await self.coord.heartbeat(self.id, {
                    "agents": sorted(self.slots), "turns": self.turns, "rss_mb": rss_mb(),
                    "inflight": self.inflight, "shed": self.shed,
                    "backlog": sum(s.engine.scheduler.backlog() for s in self.slots.values()),
                    "unsynced": sum(s.engine.store.unsynced_count() for s in self.slots.values()),
                    "uptime_s": time.time() - self.started})
            except Exception:  # noqa: BLE001
                pass


def create_app(worker: Worker):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    from cama.cost import BudgetExceeded

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        await worker.start()
        yield
        await worker.stop()

    app = FastAPI(title=f"cama worker {worker.id}", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"worker": worker.id, "agents": sorted(worker.slots), "turns": worker.turns}

    @app.post("/agents/{agent_id}/turn")
    async def turn(agent_id: str, body: TurnIn):
        try:
            return await worker.turn(agent_id, body.text)
        except NotOwner as e:
            return JSONResponse({"error": "not_owner", "owner": e.owner}, status_code=409)
        except Busy as e:
            return JSONResponse({"error": "busy", "detail": str(e)}, status_code=503, headers={"Retry-After": "1"})
        except BudgetExceeded as e:
            return JSONResponse({"error": "budget", "detail": str(e)}, status_code=402)
        except Exception as e:  # noqa: BLE001: the client retries elsewhere; the cause is logged
            log.exception("turn failed for %s", agent_id)
            raise HTTPException(503, repr(e)[:300]) from e

    @app.post("/agents/{agent_id}/seed")
    async def seed(agent_id: str, body: SeedIn):
        try:
            return await worker.seed(agent_id, body.blocks)
        except NotOwner as e:
            return JSONResponse({"error": "not_owner", "owner": e.owner}, status_code=409)
        except Busy as e:
            return JSONResponse({"error": "busy", "detail": str(e)}, status_code=503, headers={"Retry-After": "1"})

    return app
