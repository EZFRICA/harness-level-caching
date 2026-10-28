"""H2: the embedding cache (decision 0006, skill add-a-cache-level).

Key: the embedding space, the kind (query or document) and the exact text. The space is part
of the key so vectors from two geometries can never be confused. Values live in the store's
SQLite file (local first) with a small in-process LRU in front. Pure: a vector never expires;
a different text is a different key.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from typing import Literal

import numpy as np

from harness_cache.ports import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, space TEXT NOT NULL, vector BLOB NOT NULL);
"""


def embed_key(space: str, kind: str, text: str) -> str:
    return hashlib.blake2b(f"{space}\x00{kind}\x00{text}".encode(), digest_size=16).hexdigest()


class CachedEmbedder:
    """Wraps an embedder; same interface, so the router and extractor do not change."""

    def __init__(self, inner: Embedder, db: sqlite3.Connection, enabled: bool = True, ram_entries: int = 2048,
                 shared=None):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.space = f"{inner.name}:{inner.dim}"
        self.db = db
        self.enabled = enabled
        self.ram_entries = ram_entries
        self._ram: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.shared_hits = 0
        self.shared = shared  # coordinator: vectors shared between workers
        self.db.executescript(_SCHEMA)

    def counters(self) -> tuple[int, int]:
        return self.hits, self.misses

    def _get(self, key: str) -> np.ndarray | None:
        v = self._ram.get(key)
        if v is not None:
            self._ram.move_to_end(key)
            return v
        row = self.db.execute("SELECT vector FROM embed_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        v = np.frombuffer(row[0], dtype=np.float32).copy()
        self._remember(key, v)
        return v

    def _remember(self, key: str, v: np.ndarray) -> None:
        self._ram[key] = v
        self._ram.move_to_end(key)
        while len(self._ram) > self.ram_entries:
            self._ram.popitem(last=False)

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        if not self.enabled:
            self.misses += len(texts)
            return await self.inner.embed(texts, kind)
        keys = [embed_key(self.space, kind, t) for t in texts]
        found = [self._get(k) for k in keys]
        todo = [i for i, v in enumerate(found) if v is None]
        if todo and self.shared is not None:
            still = []
            for i in todo:
                try:
                    import asyncio
                    blob = await asyncio.wait_for(self.shared.get_embedding(keys[i]), 0.25)
                except Exception:  # noqa: BLE001: slow or down shared tier: compute locally
                    blob = None
                if blob is None:
                    still.append(i)
                    continue
                found[i] = np.frombuffer(blob, dtype=np.float32).copy()
                self._remember(keys[i], found[i])
                self.shared_hits += 1
            todo = still
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            fresh = await self.inner.embed([texts[i] for i in todo], kind)
            with self.db:
                for i, v in zip(todo, fresh):
                    v = np.asarray(v, dtype=np.float32)
                    found[i] = v
                    self._remember(keys[i], v)
                    self.db.execute("INSERT OR REPLACE INTO embed_cache VALUES (?,?,?)",
                                    (keys[i], self.space, v.tobytes()))
            if self.shared is not None:
                for i in todo:
                    try:
                        await self.shared.put_embedding(keys[i], self.space, found[i].tobytes())
                    except Exception:  # noqa: BLE001
                        break
        return np.vstack(found)
