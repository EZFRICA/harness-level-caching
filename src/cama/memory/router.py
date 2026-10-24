"""Cascade router (①): try the cheapest evidence first, touch L3 only on a miss.

  a. lexical:  query-term containment against active blocks. No embedding call.
  b. active:   embed the query once, cosine against the ≤capacity active vectors in RAM.
  c. l3_fault: scan every stored vector; page the best matches into the active set.
  miss:         nothing relevant anywhere; the prompt is unchanged.

The router decides what is *in* the active set and refreshes recency. In `cama`
mode it does not decide what is in the prompt: the whole active set is rendered in
a stable order, so a routing decision never reshuffles the cached prefix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cama.config import Config
from cama.memory.active_set import ActiveSet
from cama.ports import Embedder
from cama.store.sqlite_store import SQLiteStore
from cama.tokens import containment, lexical_set
from cama.types import Block


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

    def _touch(self, hits: list[tuple[str, float]], now: float) -> None:
        for block_id, _ in hits:
            self.active.touch(block_id, now)
            self.store.touch(block_id, now)
