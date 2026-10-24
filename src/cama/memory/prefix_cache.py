"""Explicit prefix cache (② online): the stable memory prefix lives in a provider cache.

Providers only cache implicitly above a size threshold (and best-effort), so CAMA
manages an explicit cache instead:

- the hot path only *looks up* a ready cache for the exact prefix it is about to
  send; on a miss it sends the request uncached and never waits for a creation;
- the cold path (after memory extraction, so the prefix is final) creates the
  cache for the current prefix and deletes the previous one.

The number of creations is therefore the number of prefix changes: the more
stable the render order, the fewer paid creations and the more cached reads.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from cama.types import ToolSpec


def prefix_hash(text: str, tools: list[ToolSpec] | None = None) -> str:
    """Identity of the cacheable prefix: the system text and the tool declarations."""
    decl = json.dumps([[t.name, t.description, t.parameters] for t in tools or []], sort_keys=True)
    return hashlib.blake2b(f"{text}\x00{decl}".encode(), digest_size=12).hexdigest()


@dataclass
class CacheEntry:
    hash: str
    name: str
    expires_at: float
    tokens: int


class PrefixCache:
    def __init__(self, llm, ttl_s: int, margin_s: int = 30, delete_replaced: bool = True):
        self.llm = llm
        self.ttl_s = ttl_s
        self.margin_s = margin_s
        self.current: CacheEntry | None = None
        self.unusable: dict[str, str] = {}  # hash -> reason (e.g. below the provider minimum)
        self.creations = 0
        # In a cluster another worker may be using the cache this one replaces: let it expire.
        self.delete_replaced = delete_replaced

    def lookup(self, h: str) -> CacheEntry | None:
        c = self.current
        if c and c.hash == h and time.time() < c.expires_at - self.margin_s:
            return c
        return None

    async def ensure(self, system: str, tools: list[ToolSpec] | None = None) -> dict:
        h = prefix_hash(system, tools)
        if self.lookup(h):
            return {"action": "reuse", "hash": h}
        if h in self.unusable:
            return {"action": "skip", "hash": h, "reason": self.unusable[h]}
        t = time.perf_counter()
        try:
            name, tokens = await self.llm.create_cache(system, tools, self.ttl_s)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "too small" in msg:
                self.unusable[h] = "below provider minimum"
                return {"action": "skip", "hash": h, "reason": self.unusable[h]}
            raise
        create_ms = (time.perf_counter() - t) * 1000
        old = self.current
        self.current = CacheEntry(h, name, time.time() + self.ttl_s, tokens)
        self.creations += 1
        if old is not None and self.delete_replaced:
            try:
                await self.llm.delete_cache(old.name)
            except Exception:  # noqa: BLE001: an orphan cache expires on its own
                pass
        return {"action": "create", "hash": h, "tokens": tokens, "create_ms": create_ms,
                "replaced": old.hash if old else None, "creations": self.creations}

    def adopt(self, h: str, name: str, expires_at: float, tokens: int) -> None:
        """Use a cache another worker created for this exact prefix instead of paying for a new one."""
        self.current = CacheEntry(h, name, expires_at, tokens)

    async def close(self) -> None:
        if self.current is not None and self.delete_replaced:  # in a cluster another worker may use it
            try:
                await self.llm.delete_cache(self.current.name)
            except Exception:  # noqa: BLE001
                pass
            self.current = None
