"""Engine: one hot-path turn, cold-path work deferred to the scheduler.

Modes (same engine, same metrics, for comparisons):
  extract   route → render the WHOLE active set in stable order → LLM
  topk   route → render only the routed hits, best first (per-query selection,
         like the original APU); isolates the effect of prefix stability
  naive  no memory, full conversation history every turn
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from dataclasses import dataclass

from harness_cache.adapters import make_archive, make_embedder, make_llm
from harness_cache.agent.loop import AgentLoop
from harness_cache.config import Config
from harness_cache.cost import Budget, MeteredLLM, load_pricing, price_for, pricing_hash
from harness_cache.logging import get_logger
from harness_cache.memory.active_set import ActiveSet, render_block
from harness_cache.memory.assembler import Assembler, render_system
from harness_cache.memory.embed_cache import CachedEmbedder
from harness_cache.memory.extractor import Extractor
from harness_cache.memory.prefix_cache import PrefixCache, prefix_hash
from harness_cache.memory.router import Router
from harness_cache.metrics import RunRecorder, peak_rss_mb, rss_mb
from harness_cache.scheduler import Scheduler
from harness_cache.store.sqlite_store import SQLiteStore
from harness_cache.teu.cache import ToolCache
from harness_cache.teu.tools import Bookings, FaultConfig, ToolRunner, default_tools
from harness_cache.tokens import approx_tokens
from harness_cache.types import Block, Kind, Message

log = get_logger(__name__)

SHARED_TIMEOUT_S = 0.25  # a hot path read of the shared tier never waits longer than this


async def _shared(coro):
    """Read the shared tier from the hot path: a slow or silent Postgres is a miss, not a wait."""
    try:
        return await asyncio.wait_for(coro, SHARED_TIMEOUT_S)
    except Exception:  # noqa: BLE001: timeout, outage: the caller treats None as a miss
        return None


PRIORITY_EXTRACT = 2
PRIORITY_CACHE = 3  # after extraction, so the cached prefix is the one the next turn will send
PRIORITY_SYNC = 4


@dataclass
class TurnResult:
    text: str
    record: dict


_ASKS = ("quel", "quelle", "quels", "quelles", "quand", "comment", "combien", "pourquoi", "ou", "qui", "est-ce",
         "calcule", "cherche", "convertis", "resume", "rappelle", "rappelle-moi", "reserve", "trouve", "donne",
         "dis", "explique", "recapitule", "refais", "va-t-il", "ai-je", "a", "what", "when", "how", "why", "where",
         "who", "which", "is", "are", "can", "could", "should", "do", "does", "find", "search", "compute",
         "convert", "summarise", "summarize", "remind", "book", "show", "tell", "give", "list")


def states_a_fact(text: str) -> bool:
    """Hypothesis (HC_EXTRACT_POLICY=statements): a message that is neither a question nor a request
    states something the memory may need now. A French and English heuristic, nothing more."""
    from harness_cache.probes import fold
    t = fold(text)
    first = t.split(" ", 1)[0].strip(",.:;!") if t else ""
    return bool(t) and "?" not in t and first not in _ASKS


COMPACT_SYSTEM = """You maintain the running summary of a conversation between a user and an assistant.
Rewrite the summary to include everything in PREVIOUS SUMMARY plus the CONVERSATION TO FOLD.
Keep every fact the user stated about themselves, their work and their plans: names, numbers,
codes, dates, places, references. When a fact was corrected, keep only the latest value.
Drop small talk and the assistant's suggestions. Output the summary only, as short bullet points."""


def _verbatim_turn(block_id: str) -> int:
    """The turn of a stored user message (H9 ids are msg-<turn>); other blocks sort first."""
    return int(block_id[4:]) if block_id.startswith("msg-") else -1


def window_size(cfg: Config, turns_done: int) -> int:
    """Exchanges sent verbatim. With an epoch E the window only grows for E-1 turns, so the
    messages after the memory are append-only (a cacheable prefix), then it drops E at once."""
    return cfg.window_turns + turns_done % max(1, cfg.window_epoch)


class Engine:
    def __init__(self, cfg: Config, recorder: RunRecorder, session: str = "default", shared=None,
                 worker: str = "local", lease=None, budget=None):
        if cfg.mode == "cama":  # the name of this mode before the study was renamed
            cfg.mode = "extract"
        if cfg.mode not in {"extract", "topk", "naive", "compact"}:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        self.cfg = cfg
        self.recorder = recorder
        self.session = session
        self.shared = shared  # coordinator, in cluster mode
        self.worker = worker
        self.lease = lease  # callable returning the current Lease, in cluster mode
        self._snapshot_seq = 0
        self.store = SQLiteStore(cfg.store_path)
        self.pricing = load_pricing(cfg.pricing_path)
        model_key = "echo" if cfg.llm == "echo" else cfg.model
        self.model_info = self.pricing.get(model_key, {})
        price = price_for(self.pricing, model_key)
        if price is None and not cfg.allow_unpriced:
            raise ValueError(f"{model_key} has no price in {cfg.pricing_path}: the budget cap could not hold "
                             "(set HC_ALLOW_UNPRICED=1 to run anyway, uncapped)")
        self.budget = budget or Budget(cfg.max_usd)
        if cfg.embedder == "gemini" and not cfg.allow_unpriced:
            # Review finding: remote embeddings were paid calls outside the budget guard.
            raise ValueError("gemini embeddings have no price in pricing.toml: the budget cap could not hold "
                             "(set HC_ALLOW_UNPRICED=1 to run anyway, uncapped)")
        inner = make_llm(cfg)
        self.llm = MeteredLLM(inner, price, self.budget, cfg.cache_ttl_s) if price else inner
        self.embedder = CachedEmbedder(make_embedder(cfg), self.store.db, enabled=cfg.embed_cache, shared=shared)
        if shared is not None:
            from harness_cache.cluster.archive import CoordinatorArchive
            self.archive = CoordinatorArchive(shared, lease)
        else:
            self.archive = make_archive(cfg)
        self.bookings = Bookings()
        self.tools = default_tools(self.bookings)
        self.web = None
        if cfg.web_search:  # opt in: a new tool changes the tool declarations, hence the prefix
            from harness_cache.teu.web import WebSearch, WebTape, web_search_tool
            self.web = WebSearch(WebTape(cfg.web_tape), cfg.web_mode, os.environ.get("TAVILY_API_KEY", ""),
                                 cfg.web_max_calls, embed=self.embedder.embed, tau=cfg.web_semantic_tau)
            self.tools["web_search"] = web_search_tool(self.web)
        self.tool_specs = [t.spec for t in self.tools.values()] if cfg.tools else None
        self.tool_cache = ToolCache(self.store.db, enabled=cfg.tool_cache, shared=shared, worker=worker)
        runner = ToolRunner(self.tools, FaultConfig(cfg.tool_error_rate, cfg.tool_latency_scale))
        self.loop = AgentLoop(self.llm, runner, self.tool_cache, cfg.max_steps, temperature=cfg.temperature,
                              llm_attempts=cfg.llm_attempts)
        self.store.bind_embedding_space(self.embedder.name, self.embedder.dim)
        self.active = ActiveSet(cfg.capacity, cfg.max_memory_tokens, cfg.victim_window)
        self.active.load(self.store.active_blocks())
        self.router = Router(cfg, self.store, self.active, self.embedder)
        self.assembler = Assembler()
        self.extractor = Extractor(self.llm, self.embedder, self.store, self.active, replace=cfg.extract_replace)
        self.scheduler = Scheduler(on_done=recorder.task)
        self.cache_mode = self._resolve_cache_mode()
        self.prefix_cache = (PrefixCache(self.llm, cfg.cache_ttl_s, delete_replaced=shared is None)
                             if self.cache_mode == "explicit" else None)
        self._last_system: str | None = None
        self._frozen: list[Block] | None = None  # memory rendered until the next epoch (H7)
        self._summary = ""  # compact mode: the running summary of the folded history
        self._compact_from = 0  # compact mode: history rows already folded into the summary
        self._compact_queued = False
        self._tail: list[Block] = []  # blocks paged in this turn, sent after the window (H8)
        self._last_sent_hash: str | None = None  # the prefix the last turn actually sent
        self._sightings: dict[str, int] = {}  # prefix hash -> turns that sent it
        self._pending: list[tuple[str, str]] = []  # exchanges waiting for extraction
        self._extract_queued = False
        self._replicator: asyncio.Task | None = None
        self.turn_index = len(self.store.history(session)) // 2

    def _memory_view(self, route) -> list[Block]:
        """The blocks rendered this turn. With HC_MEMORY_REFRESH=epoch (hypothesis H7) the
        rendered memory is a copy frozen until the next window drop, so the whole prompt is
        append-only between drops; facts learnt meanwhile are still in the growing window. A
        page-in refreshes it at once: the query needs the block it brought."""
        live = self.active.render_order()
        if self.cfg.render_order == "recency":  # ablation of decision 0002: reading a block moves it
            live = sorted(live, key=lambda b: (b.accessed_at, b.id))
        if self.cfg.memory_refresh != "epoch" or self.cfg.window_epoch <= 1:
            return live
        boundary = self.turn_index % self.cfg.window_epoch == 0
        paged = bool(route is not None and route.paged_in)
        if self._frozen is None or boundary or (paged and not self.cfg.page_in_tail):
            self._frozen = [dataclasses.replace(b) for b in live]
        elif paged:  # H8: the block goes to the tail of this turn, the cached head stays as it was
            shown = {b.id for b in self._frozen}
            self._tail = [b for b in (self.active.get(i) for i in route.paged_in) if b and b.id not in shown]
        return self._frozen

    def _resolve_cache_mode(self) -> str:
        """explicit where the provider offers it, implicit (stable order only) where it does not."""
        mode = self.cfg.prefix_cache
        if self.cfg.mode == "naive" or mode == "none":
            return "none"
        explicit_ok = bool(self.model_info.get("explicit_cache", False))
        if mode == "auto":
            return "explicit" if explicit_ok else "implicit"
        if mode == "explicit" and not explicit_ok:
            raise ValueError(f"{self.cfg.model} does not support explicit caching (pricing.toml)")
        return mode

    def manifest_extra(self) -> dict:
        return {"pricing_hash": pricing_hash(self.cfg.pricing_path), "cache_mode": self.cache_mode,
                "model_info": {k: v for k, v in self.model_info.items() if k in (
                    "status", "explicit_cache", "implicit_min_tokens", "read_on")},
                "embedding_space": self.embedder.space}

    async def start(self) -> None:
        self.scheduler.start()
        if self.cfg.online and self.cfg.sync_interval_s > 0:
            self._replicator = asyncio.create_task(self._replicate_forever())

    async def _replicate_forever(self) -> None:
        """Ship the journal on a clock, not when idle. Measured: under back to back turns the
        scheduler was never idle, the sync task never ran, and a crash lost 26 rows per agent."""
        while True:
            await asyncio.sleep(self.cfg.sync_interval_s)
            try:
                out = await self._sync()
                if out["synced"]:
                    self.recorder.task({"task": "replicate", "ok": True, **out})
            except Exception as e:  # noqa: BLE001: the next tick retries; fencing errors are recorded
                self.recorder.task({"task": "replicate", "ok": False, "error": repr(e)[:200]})

    async def close(self) -> None:
        if self._replicator:
            self._replicator.cancel()
        await self.scheduler.stop()
        if self.cfg.online:
            try:  # a clean shutdown ships the journal; a crash is what the RPO measures
                await self._sync()
            except Exception:  # noqa: BLE001
                pass
        if self.prefix_cache is not None:
            await self.prefix_cache.close()
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
        h2_before = self.embedder.counters()  # routing embeds the query: count from the start

        if cfg.mode == "naive":
            blocks: list[Block] = []
            history = self.store.history(self.session)
        elif cfg.mode == "compact":  # baseline: full history, older part folded into a summary
            blocks = [Block("conversation-summary", Kind.PROJET, "Conversation summary", self._summary)] \
                if self._summary else []
            history = self.store.history(self.session)[self._compact_from:]
            if cfg.verbatim_recall:
                route = await self._recall_verbatim(user_text)
        else:
            route = await self.router.route(user_text)
            timings.update({f"route_{k}": v for k, v in route.timings_ms.items()})
            if cfg.mode == "extract":
                blocks = self._memory_view(route)
            else:  # topk: pinned blocks always, then this query's hits by score
                pinned = [b for b in self.active.render_order() if b.pinned]
                hits = [self.active.get(i) for i, _ in route.hits]
                blocks = pinned + [b for b in hits if b is not None and not b.pinned]
            history = self.store.history(self.session, last=2 * window_size(cfg, self.turn_index))

        sent_text = user_text
        if self._tail:  # H8: what the query paged in, in front of the question, at the end of the prompt
            sent_text = ("[memory paged in for this question]\n" + "".join(render_block(b) for b in self._tail)
                         + "\n" + user_text)
            self._tail = []
        t = time.perf_counter()
        req = self.assembler.build(self.session, blocks, history, sent_text)
        timings["assemble"] = (time.perf_counter() - t) * 1000

        phash = prefix_hash(req.system, self.tool_specs)
        cache_entry = self.prefix_cache.lookup(phash) if self.prefix_cache is not None else None
        cache_name, cache_source = (cache_entry.name, "local") if cache_entry else (None, None)
        if cache_name is None and self.prefix_cache is not None and self.shared is not None:
            rem = await _shared(self.shared.get_prefix(phash, self.cfg.model))  # another worker may have it
            if rem and rem["expires_at"] > time.time() + self.prefix_cache.margin_s:
                cache_name, cache_source = rem["name"], "shared"
        self._last_system = req.system
        self._last_sent_hash = phash
        self._sightings[phash] = self._sightings.get(phash, 0) + 1

        out = await self.loop.run(req.system, req.messages, self.tool_specs, cache_name)
        cache_error = out.cache_error
        if cache_error:  # the step fell back without the cache; the turn was not replayed
            cache_name, cache_source = None, None
        timings["llm"] = out.llm_ms
        timings["tools"] = out.tool_ms
        h2_hits = self.embedder.hits - h2_before[0]
        h2_misses = self.embedder.misses - h2_before[1]

        self.store.append_history(self.session, "user", sent_text)  # the window replays what was sent
        self.store.append_history(self.session, "assistant", out.text)
        hot_ms = (time.perf_counter() - t0) * 1000
        self.turn_index += 1

        if cfg.mode == "compact" and not self._compact_queued:
            unfolded = self.store.history(self.session)[self._compact_from:]
            if sum(approx_tokens(m.text) for m in unfolded) > cfg.compact_at:
                self._compact_queued = True
                await self.scheduler.push(f"compact#{self.turn_index}", self._compact, PRIORITY_EXTRACT)
        if cfg.mode == "compact" and cfg.verbatim_recall:
            turn_no = self.turn_index - 1
            await self.scheduler.push(f"verbatim#{self.turn_index}",
                                      lambda: self._store_verbatim(turn_no, user_text), PRIORITY_EXTRACT)
        if cfg.mode in ("extract", "topk"):
            self._pending.append((user_text, out.text))
            due = len(self._pending) >= cfg.extract_every or (
                cfg.extract_policy == "statements" and states_a_fact(user_text))
            if not self._extract_queued and due:  # one call drains them all
                self._extract_queued = True
                await self.scheduler.push(f"extract#{self.turn_index}", self._extract_pending, PRIORITY_EXTRACT)
            if self.prefix_cache is not None:
                await self.scheduler.push(f"cache#{self.turn_index}", self._ensure_cache, PRIORITY_CACHE)
            if cfg.online and cfg.sync_interval_s <= 0:
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
                "blocks": req.blocks_rendered, "block_keys": [b.segment_key for b in blocks],
                "active_size": len(self.active),
                "memory_tokens_est": req.memory_tokens, "window_tokens_est": req.window_tokens,
                "total_tokens_est": req.total_tokens, "stable_prefix_tokens_est": req.stable_prefix_tokens,
            },
            "llm": {"prompt_tokens": out.prompt_tokens, "cached_tokens": out.cached_tokens,
                    "output_tokens": out.output_tokens, "calls": len(out.steps)},
            "steps": out.steps,
            "stopped": out.stopped,
            "model_retries": out.model_retries,
            "tools": out.tools,
            "h2": {"hits": h2_hits, "misses": h2_misses, "enabled": self.embedder.enabled},
            "h3": {"hits": sum(t["outcome"] == "hit" for t in out.tools),
                   "misses": sum(t["outcome"] == "miss" for t in out.tools),
                   "bypass": sum(t["outcome"] == "bypass" for t in out.tools),
                   "errors": sum(t["outcome"] == "error" for t in out.tools),
                   "saved_ms": sum(t["saved_ms"] for t in out.tools), "enabled": self.tool_cache.enabled},
            "web": None if self.web is None else {"live_calls": self.web.live_calls, "replayed": self.web.replayed,
                                                  "semantic": list(self.web.semantic)},
            "prefix_cache": None if self.prefix_cache is None else {
                "hit": cache_name is not None, "source": cache_source, "error": cache_error, "hash": phash[:10],
                "creations": self.prefix_cache.creations,
            },
            "cache_mode": self.cache_mode,
            "budget_spent_usd": self.budget.spent_usd,
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
    async def _extract_pending(self) -> dict:
        batch, self._pending = self._pending[: self.cfg.extract_batch_max], self._pending[self.cfg.extract_batch_max:]
        self._extract_queued = bool(self._pending)
        if self._pending:
            await self.scheduler.push(f"extract#{self.turn_index}+", self._extract_pending, PRIORITY_EXTRACT)
        rep = await self.extractor.run_many(batch)
        return {**self._report(rep), "exchanges": len(batch)}

    async def _compact(self) -> dict:
        """Fold all but the last `compact_keep` exchanges into the running summary (one model call)."""
        try:
            rows = self.store.history(self.session)[self._compact_from:]
            fold = rows[:max(0, len(rows) - 2 * self.cfg.compact_keep)]
            if not fold:
                return {"folded": 0}
            convo = "\n".join(f"{m.role.upper()}: {m.text}" for m in fold)
            prompt = f"PREVIOUS SUMMARY:\n{self._summary or '(none)'}\n\nCONVERSATION TO FOLD:\n{convo}"
            out = await self.llm.generate(COMPACT_SYSTEM, [Message("user", prompt)], temperature=0.0)
            if out.text.strip():
                self._summary = out.text.strip()
                self._compact_from += len(fold)
            return {"folded": len(fold), "summary_tokens_est": approx_tokens(self._summary),
                    "prompt_tokens": out.prompt_tokens, "cached_tokens": out.cached_tokens,
                    "output_tokens": out.output_tokens}
        finally:
            self._compact_queued = False

    async def _recall_verbatim(self, user_text: str):
        """H9: search the stored user messages (apu-urithi's search) and put the ones found, oldest
        first with their turn number, at the end of the prompt, so the head stays the cached
        prefix. Messages still in the unfolded history are already in the prompt and skipped."""
        first_unfolded = self._compact_from // 2
        found, route = await self.router.recall(user_text, skip=lambda bid: _verbatim_turn(bid) >= first_unfolded)
        if found:
            found.sort(key=lambda bs: _verbatim_turn(bs[0].id))
            self._tail = [b for b, _ in found]
        return route

    async def _store_verbatim(self, turn_no: int, user_text: str) -> dict:
        """H9 cold path: the user's message as it was written, embedded locally, stored in L3 only.
        No model call: this replaces extraction."""
        vec = (await self.embedder.embed([user_text], "document"))[0]
        now = time.time()
        b = Block(id=f"msg-{turn_no:06d}", kind=Kind.PROJET, title=f"turn {turn_no}", content=user_text,
                  vector=vec, created_at=now, modified_at=now, accessed_at=now)
        self.store.upsert(b, "verbatim", active=False)
        return {"stored": b.id}

    def _report(self, rep) -> dict:
        return {
            "ops_proposed": rep.ops_proposed, "appended": rep.appended, "created": rep.created,
            "evicted": rep.evicted, "rejected": rep.rejected, "invalid_ops": rep.invalid_ops,
            "parse_error": rep.parse_error, "prompt_tokens": rep.prompt_tokens,
            "cached_tokens": rep.cached_tokens, "replaced": rep.replaced, "replace_missed": rep.replace_missed,
            "output_tokens": rep.output_tokens, "active_size": len(self.active),
        }

    async def _extract(self, user_text: str, assistant_text: str) -> dict:
        rep = await self.extractor.run(user_text, assistant_text)
        return {
            "ops_proposed": rep.ops_proposed, "appended": rep.appended, "created": rep.created,
            "evicted": rep.evicted, "rejected": rep.rejected, "invalid_ops": rep.invalid_ops,
            "parse_error": rep.parse_error, "prompt_tokens": rep.prompt_tokens,
            "cached_tokens": rep.cached_tokens, "replaced": rep.replaced, "replace_missed": rep.replace_missed,
            "output_tokens": rep.output_tokens, "active_size": len(self.active),
        }

    async def _ensure_cache(self) -> dict:
        # extract: the next request will render the current active set, so cache exactly that.
        # topk: the next prefix depends on the next query; cache the last one sent.
        if self.cfg.mode == "extract":
            system = render_system(self.active.render_order())
        else:
            system = self._last_system or ""
        h = prefix_hash(system, self.tool_specs)
        # Admission: a creation pays only if the prefix is sent again. Expected sends = turns that
        # already sent it, plus one when extract's prefix is unchanged since the last turn (the next
        # turn renders the same set). Measured: creating on every change made 200 caches in 200
        # turns, and per-query prefixes (topk) made 198 caches for 2 hits.
        expected = self._sightings.get(h, 0) + (1 if self.cfg.mode == "extract" and h == self._last_sent_hash else 0)
        if self.cfg.cache_admission == "stable" and expected < 2 and not self.prefix_cache.lookup(h):
            return {"action": "skip", "hash": h, "reason": "not yet stable"}
        if self.shared is not None and not self.prefix_cache.lookup(h):
            rem = await _shared(self.shared.get_prefix(h, self.cfg.model))
            if rem and rem["expires_at"] > time.time() + self.prefix_cache.margin_s:
                self.prefix_cache.adopt(h, rem["name"], rem["expires_at"], rem["tokens"])
                return {"action": "adopt", "hash": h, "from": rem["worker"]}
        out = await self.prefix_cache.ensure(system, self.tool_specs)
        if out["action"] == "create" and self.shared is not None:
            c = self.prefix_cache.current
            await self.shared.put_prefix(c.hash, self.cfg.model, c.name, c.expires_at, c.tokens, self.worker)
        return out

    async def _sync(self) -> dict:
        entries = self.store.unsynced()
        if entries:
            await self.archive.push(entries)
            self.store.mark_synced([e["seq"] for e in entries])
        out = {"synced": len(entries)}
        seq = self.store.max_seq()
        # A snapshot must describe replicated rows only: taken while rows are still unshipped, its
        # seq would run ahead of the shared journal and a later recovery would skip a new owner's
        # rows (review finding). It is fenced by the lease, like the journal.
        if (self.shared is not None and seq - self._snapshot_seq >= self.cfg.snapshot_every
                and self.store.unsynced_count() == 0 and self.lease is not None and self.lease() is not None):
            await self.shared.put_snapshot(self.lease(), seq, self.store.snapshot_data())
            self._snapshot_seq = seq
            out["snapshot_seq"] = seq
        return out

    async def drain(self) -> None:
        """Wait for the cold path, then ship whatever the replicator has not shipped yet."""
        await self.scheduler.drain()
        if self.cfg.online:
            await self._sync()
