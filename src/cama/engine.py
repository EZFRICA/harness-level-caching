"""Engine: one hot-path turn, cold-path work deferred to the scheduler.

Modes (same engine, same metrics, for comparisons):
  cama   route → render the WHOLE active set in stable order → LLM
  topk   route → render only the routed hits, best first (per-query selection,
         like the original APU); isolates the effect of prefix stability
  naive  no memory, full conversation history every turn
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cama.adapters import make_archive, make_embedder, make_llm
from cama.config import Config
from cama.logging import get_logger
from cama.memory.active_set import ActiveSet
from cama.memory.assembler import Assembler
from cama.memory.extractor import Extractor
from cama.memory.router import Router
from cama.metrics import RunRecorder, peak_rss_mb, rss_mb
from cama.scheduler import Scheduler
from cama.store.sqlite_store import SQLiteStore
from cama.types import Block, Kind

log = get_logger(__name__)

PRIORITY_EXTRACT = 2
PRIORITY_SYNC = 3


@dataclass
class TurnResult:
    text: str
    record: dict


class Engine:
    def __init__(self, cfg: Config, recorder: RunRecorder, session: str = "default"):
        if cfg.mode not in {"cama", "topk", "naive"}:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        self.cfg = cfg
        self.recorder = recorder
        self.session = session
        self.store = SQLiteStore(cfg.store_path)
        self.llm = make_llm(cfg)
        self.embedder = make_embedder(cfg)
        self.archive = make_archive(cfg)
        self.store.bind_embedding_space(self.embedder.name, self.embedder.dim)
        self.active = ActiveSet(cfg.capacity, cfg.max_memory_tokens, cfg.victim_window)
        self.active.load(self.store.active_blocks())
        self.router = Router(cfg, self.store, self.active, self.embedder)
        self.assembler = Assembler()
        self.extractor = Extractor(self.llm, self.embedder, self.store, self.active)
        self.scheduler = Scheduler(on_done=recorder.task)
        self.turn_index = len(self.store.history(session)) // 2

    async def start(self) -> None:
        self.scheduler.start()

    async def close(self) -> None:
        await self.scheduler.stop()
        self.store.close()
        self.recorder.close()

    # ── seeding ────────────────────────────────────────────────────────────
    async def seed(self, specs: list[dict]) -> dict:
        """Insert blocks. `active: false` stores a block in L3 only (to provoke page faults)."""
        now = time.time()
        texts = [f"{s['title']}\n{s['content']}" for s in specs]
        vectors = await self.embedder.embed(texts, "document") if texts else []
        inserted, evicted = [], []
        for i, (s, vec) in enumerate(zip(specs, vectors)):
            b = Block(
                id=s["id"], kind=Kind.parse(s.get("kind", "temp")), title=s["title"], content=s["content"],
                pinned=bool(s.get("pinned", False)), vector=vec,
                created_at=now + i * 1e-3, modified_at=now + i * 1e-3, accessed_at=now + i * 1e-3,
            )
            want_active = s.get("active", True)
            admitted = False
            if want_active:
                ev, admitted = self.active.admit(b)
                for v in ev:
                    self.store.set_active(v.id, False)
                    evicted.append(v.id)
            self.store.upsert(b, "seed", active=admitted)
            inserted.append(b.id)
        return {"inserted": inserted, "evicted": evicted}

    # ── hot path ───────────────────────────────────────────────────────────
    async def turn(self, user_text: str) -> TurnResult:
        self.scheduler.idle.clear()
        try:
            return await self._turn(user_text)
        finally:
            self.scheduler.idle.set()

    async def _turn(self, user_text: str) -> TurnResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        timings: dict[str, float] = {}
        route = None

        if cfg.mode == "naive":
            blocks: list[Block] = []
            history = self.store.history(self.session)
        else:
            route = await self.router.route(user_text)
            timings.update({f"route_{k}": v for k, v in route.timings_ms.items()})
            if cfg.mode == "cama":
                blocks = self.active.render_order()
            else:  # topk: pinned blocks always, then this query's hits by score
                pinned = [b for b in self.active.render_order() if b.pinned]
                hits = [self.active.get(i) for i, _ in route.hits]
                blocks = pinned + [b for b in hits if b is not None and not b.pinned]
            history = self.store.history(self.session, last=2 * cfg.window_turns)

        t = time.perf_counter()
        req = self.assembler.build(self.session, blocks, history, user_text)
        timings["assemble"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        out = await self.llm.generate(req.system, req.messages, temperature=cfg.temperature)
        timings["llm"] = (time.perf_counter() - t) * 1000

        self.store.append_history(self.session, "user", user_text)
        self.store.append_history(self.session, "assistant", out.text)
        hot_ms = (time.perf_counter() - t0) * 1000
        self.turn_index += 1

        if cfg.mode != "naive":
            await self.scheduler.push(
                f"extract#{self.turn_index}", lambda u=user_text, a=out.text: self._extract(u, a), PRIORITY_EXTRACT
            )
            if cfg.online:
                await self.scheduler.push(f"sync#{self.turn_index}", self._sync, PRIORITY_SYNC)

        record = {
            "turn": self.turn_index,
            "session": self.session,
            "mode": cfg.mode,
            "hot_ms": hot_ms,
            "cpu_ms": (time.process_time() - cpu0) * 1000,
            "timings_ms": timings,
            "route": None if route is None else {
                "tier": route.tier, "hits": route.hits[:5], "paged_in": route.paged_in,
                "evicted": route.evicted, "rejected": route.rejected,
                "embed_calls": route.embed_calls, "l3_rows_scanned": route.l3_rows_scanned,
            },
            "context": {
                "blocks": req.blocks_rendered, "active_size": len(self.active),
                "memory_tokens_est": req.memory_tokens, "window_tokens_est": req.window_tokens,
                "total_tokens_est": req.total_tokens, "stable_prefix_tokens_est": req.stable_prefix_tokens,
            },
            "llm": {"prompt_tokens": out.prompt_tokens, "cached_tokens": out.cached_tokens,
                    "output_tokens": out.output_tokens},
            "cold_backlog": self.scheduler.backlog(),
            "rss_mb": rss_mb(), "peak_rss_mb": peak_rss_mb(),
            "query_chars": len(user_text),
        }
        self.recorder.turn(record)
        log.info(
            "turn %d mode=%s tier=%s hot=%.0fms llm=%.0fms prompt=%d cached=%d stable_est=%d/%d",
            self.turn_index, cfg.mode, route.tier if route else "-", hot_ms, timings["llm"],
            out.prompt_tokens, out.cached_tokens, req.stable_prefix_tokens, req.total_tokens,
        )
        return TurnResult(out.text, record)

    # ── cold path ──────────────────────────────────────────────────────────
    async def _extract(self, user_text: str, assistant_text: str) -> dict:
        rep = await self.extractor.run(user_text, assistant_text)
        return {
            "ops_proposed": rep.ops_proposed, "appended": rep.appended, "created": rep.created,
            "evicted": rep.evicted, "rejected": rep.rejected, "invalid_ops": rep.invalid_ops,
            "parse_error": rep.parse_error, "prompt_tokens": rep.prompt_tokens,
            "output_tokens": rep.output_tokens, "active_size": len(self.active),
        }

    async def _sync(self) -> dict:
        entries = self.store.unsynced()
        if entries:
            await self.archive.push(entries)
            self.store.mark_synced([e["seq"] for e in entries])
        return {"synced": len(entries)}

    async def drain(self) -> None:
        await self.scheduler.drain()
