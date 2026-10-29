"""Cascade router (①): try the cheapest evidence first, touch L3 only on a miss.

  a. lexical:  query-term containment against active blocks. No embedding call.
  b. active:   embed the query once, cosine against the ≤capacity active vectors in RAM.
  c. l3_fault: scan every stored vector; page the best matches into the active set.
  miss:         nothing relevant anywhere; the prompt is unchanged.

The router decides what is *in* the active set and refreshes recency. In `extract`
mode it does not decide what is in the prompt: the whole active set is rendered in
a stable order, so a routing decision never reshuffles the cached prefix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from harness_cache.config import Config
from harness_cache.memory import keywords
from harness_cache.memory.active_set import ActiveSet
from harness_cache.ports import Embedder
from harness_cache.store.sqlite_store import SQLiteStore
from harness_cache.tokens import containment, lexical_set
from harness_cache.types import Block, Kind

URITHI_LIMIT = 12  # candidates per search, as in apu-urithi search_block_index(limit=12)
# apu-urithi apu/config.py CERTAINTY_THRESHOLDS, calibrated on the same 384-dim MiniLM space:
# fundamental 0.40, project and knowledge 0.45, temporary 0.50.
URITHI_THRESHOLDS = {Kind.FONDAMENTAL: 0.40, Kind.PROJET: 0.45, Kind.TEMP: 0.50}
# The same three thresholds in EmbeddingGemma 2's space, carried over by percentile of the cosines
# between the 1500 turn scenario's user messages (in sample). A threshold never crosses embedders.
URITHI_THRESHOLDS_BY_EMBEDDER = {
    "gemma": {Kind.FONDAMENTAL: 0.748, Kind.PROJET: 0.762, Kind.TEMP: 0.775},
}


@dataclass
class RouteResult:
    tier: str = "miss"
    hits: list[tuple[str, float]] = field(default_factory=list)  # (block_id, score), best first
    paged_in: list[str] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    embed_calls: int = 0
    l3_rows_scanned: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


class Router:
    def __init__(self, cfg: Config, store: SQLiteStore, active: ActiveSet, embedder: Embedder):
        self.cfg = cfg
        self.store = store
        self.active = active
        self.embedder = embedder
        self._lex_cache: dict[str, frozenset[str]] = {}  # segment_key -> terms

    def _terms(self, block: Block) -> frozenset[str]:
        key = block.segment_key
        terms = self._lex_cache.get(key)
        if terms is None:
            terms = self._lex_cache[key] = lexical_set(block.text)
        return terms

    async def route(self, query: str) -> RouteResult:
        if self.cfg.route_policy == "urithi":
            return await self._route_urithi(query)
        res = RouteResult()
        now = time.time()

        # a. lexical
        t = time.perf_counter()
        q_terms = lexical_set(query)
        lex = []
        for b in self.active.blocks():
            score, overlap = containment(q_terms, self._terms(b))
            if score >= self.cfg.tau_lexical and overlap >= self.cfg.min_lexical_overlap:
                lex.append((b.id, score))
        res.timings_ms["lexical"] = (time.perf_counter() - t) * 1000
        if lex:
            res.tier = "lexical"
            res.hits = sorted(lex, key=lambda x: -x[1])
            self._touch(res.hits, now)
            return res

        # b. active-set embeddings
        t = time.perf_counter()
        qv = (await self.embedder.embed([query], "query"))[0]
        res.embed_calls = 1
        res.timings_ms["embed_query"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        scored = [
            (b.id, float(np.dot(b.vector, qv))) for b in self.active.blocks() if b.vector is not None
        ]
        act = sorted([s for s in scored if s[1] >= self.cfg.tau_active], key=lambda x: -x[1])
        res.timings_ms["active"] = (time.perf_counter() - t) * 1000
        if act:
            res.tier = "active"
            res.hits = act
            self._touch(act, now)
            return res

        # c. L3 page fault
        t = time.perf_counter()
        found, scanned = self.store.search(qv, k=self.cfg.page_in_k, exclude=self.active.ids())
        res.l3_rows_scanned = scanned
        for block_id, score in found:
            if score < self.cfg.tau_l3:
                continue
            block = self.store.get(block_id)
            if block is None:
                continue
            block.accessed_at = now
            evicted, admitted = self.active.admit(block)
            for v in evicted:
                self.store.set_active(v.id, False)
                res.evicted.append(v.id)
            if admitted:
                self.store.set_active(block.id, True)
                self.store.touch(block.id, now)
                res.paged_in.append(block.id)
                res.hits.append((block.id, score))
            else:
                res.rejected.append(block.id)
        res.timings_ms["l3"] = (time.perf_counter() - t) * 1000
        res.tier = "l3_fault" if res.paged_in else "miss"
        return res

    async def _route_urithi(self, query: str) -> RouteResult:
        """apu-urithi's search (apu/mmu/dll.py search_memory, library/keywords.py), taken as is:
        every question searches the whole store, never stopping at a weak active hit; a block
        counts if its cosine passes the threshold of its kind (calibrated there on this MiniLM
        space) or if it is the best keyword match; meaning and words are fused by rank; blocks
        found outside the active set are paged in."""
        res = RouteResult(tier="miss")
        now = time.time()
        accepted = await self._urithi_select(query, res)
        t = time.perf_counter()
        for block, score in accepted:
            res.hits.append((block.id, score))
            if block.id in self.active.ids():
                self.active.touch(block.id, now)
                self.store.touch(block.id, now)
                continue
            block.accessed_at = now
            evicted, admitted = self.active.admit(block)
            for v in evicted:
                self.store.set_active(v.id, False)
                res.evicted.append(v.id)
            if admitted:
                self.store.set_active(block.id, True)
                self.store.touch(block.id, now)
                res.paged_in.append(block.id)
            else:
                res.rejected.append(block.id)
        res.timings_ms["urithi"] = (time.perf_counter() - t) * 1000
        res.tier = "l3_fault" if res.paged_in else ("active" if res.hits else "miss")
        return res

    async def _urithi_select(self, query: str, res: RouteResult, skip=None) -> list[tuple[Block, float]]:
        """The blocks apu-urithi's search accepts for a query, best first, at most page_in_k.
        `skip(block_id)` drops a candidate before it takes a place (already in the prompt)."""
        t = time.perf_counter()
        qv = (await self.embedder.embed([query], "query"))[0]
        res.embed_calls = 1
        res.timings_ms["embed_query"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        by_meaning, scanned = self.store.search(qv, k=URITHI_LIMIT, exclude=set())
        res.l3_rows_scanned = scanned
        rows = self.store.db.execute("SELECT id, title, content FROM blocks").fetchall()
        if skip is not None:
            by_meaning = [(b, s) for b, s in by_meaning if not skip(b)]
            rows = [r for r in rows if not skip(r["id"])]
        scores = keywords.bm25(query, [f"{r['title']}\n{r['content']}" for r in rows])
        by_words = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -scores[i]) if scores[i] > 0]
        cos = dict(by_meaning)
        thresholds = URITHI_THRESHOLDS_BY_EMBEDDER.get(self.cfg.embedder, URITHI_THRESHOLDS)
        accepted = []
        for bid in keywords.fuse([b for b, _ in by_meaning], by_words[:URITHI_LIMIT]):
            block = self.active.get(bid) or self.store.get(bid)
            if block is None:
                continue
            passes = cos.get(bid, -1.0) >= thresholds[block.kind]
            if passes or (by_words and bid == by_words[0]):
                accepted.append((block, cos.get(bid, 0.0)))
            if len(accepted) >= self.cfg.page_in_k:
                break
        res.timings_ms["urithi_search"] = (time.perf_counter() - t) * 1000
        return accepted

    async def recall(self, query: str, skip=None) -> tuple[list[tuple[Block, float]], RouteResult]:
        """Hypothesis H9 (compact mode): the same search over stored user messages, read only.
        Nothing is paged in: the caller appends what it finds to the end of the prompt."""
        res = RouteResult(tier="miss")
        accepted = await self._urithi_select(query, res, skip)
        res.hits = [(b.id, s) for b, s in accepted]
        res.tier = "recall" if accepted else "miss"
        return accepted, res

    def _touch(self, hits: list[tuple[str, float]], now: float) -> None:
        for block_id, _ in hits:
            self.active.touch(block_id, now)
            self.store.touch(block_id, now)
