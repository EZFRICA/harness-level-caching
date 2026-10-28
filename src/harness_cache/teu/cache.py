"""H3: the tool result cache (decision 0006, skill add-a-cache-level).

Key: the tool name and its arguments after the tool's own normalisation, as canonical JSON.
Lookup on the hot path; fill on the hot path, where the result is produced anyway.
Invalidation by class: pure never expires, ttl expires after the tool's TTL, never is never
stored. Every lookup reports its outcome and, on a miss, its cause.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

from harness_cache.teu.tools import Tool

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_cache (
    key TEXT PRIMARY KEY, tool TEXT NOT NULL, args TEXT NOT NULL, value TEXT NOT NULL,
    created_at REAL NOT NULL, expires_at REAL, hits INTEGER NOT NULL DEFAULT 0
);
"""


def tool_key(name: str, normalised_args: dict) -> str:
    canon = json.dumps({"tool": name, "args": normalised_args}, sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(canon.encode(), digest_size=16).hexdigest()


@dataclass
class Lookup:
    outcome: str  # hit | miss | bypass
    cause: str  # hit | cold | expired | side_effect | disabled
    key: str
    value: dict | None = None


class ToolCache:
    def __init__(self, db: sqlite3.Connection, enabled: bool = True, clock: Callable[[], float] = time.time,
                 shared=None, worker: str = "local"):
        self.db = db
        self.enabled = enabled
        self.clock = clock
        self.shared = shared  # coordinator: results shared between workers, local copy in front
        self.worker = worker
        self.db.executescript(_SCHEMA)

    async def alookup(self, tool: Tool, args: dict) -> Lookup:
        look = self.lookup(tool, args)
        if look.outcome != "miss" or self.shared is None:
            return look
        try:
            import asyncio
            remote = await asyncio.wait_for(self.shared.get_tool(look.key), 0.25)
        except Exception:  # noqa: BLE001: slow or down shared tier: a miss is always safe
            return Lookup(look.outcome, f"{look.cause}+shared_down", look.key)
        if remote is None:
            return look
        # Keep the producer's expiry: a shared value must not live longer here than its TTL allows.
        self.store(tool, args, remote["value"], expires_at=remote.get("expires_at"))
        return Lookup("hit", "shared" if remote["worker"] != self.worker else "hit", look.key, remote["value"])

    async def astore(self, tool: Tool, args: dict, value: dict) -> None:
        self.store(tool, args, value)
        if self.shared is None or not self.enabled or tool.cache_class == "never":
            return
        expires = None if tool.cache_class == "pure" else self.clock() + tool.ttl_s
        try:
            await self.shared.put_tool(tool_key(tool.spec.name, tool.normalise(args)), value, expires, self.worker)
        except Exception:  # noqa: BLE001: sharing is best effort
            pass

    def lookup(self, tool: Tool, args: dict) -> Lookup:
        norm = tool.normalise(args)
        key = tool_key(tool.spec.name, norm)
        if tool.cache_class == "never":
            return Lookup("bypass", "side_effect", key)
        if not self.enabled:
            return Lookup("bypass", "disabled", key)
        row = self.db.execute("SELECT value, expires_at FROM tool_cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return Lookup("miss", "cold", key)
        if row[1] is not None and self.clock() >= row[1]:
            return Lookup("miss", "expired", key)
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError):  # a corrupted row is a miss, never a failed turn
            return Lookup("miss", "corrupt", key)
        with self.db:
            self.db.execute("UPDATE tool_cache SET hits = hits + 1 WHERE key=?", (key,))
        return Lookup("hit", "hit", key, value)

    def store(self, tool: Tool, args: dict, value: dict, expires_at: float | None = None) -> None:
        if not self.enabled or tool.cache_class == "never":
            return
        now = self.clock()
        expires = None if tool.cache_class == "pure" else (expires_at if expires_at is not None else now + tool.ttl_s)
        norm = tool.normalise(args)
        with self.db:
            self.db.execute(
                """INSERT INTO tool_cache (key, tool, args, value, created_at, expires_at) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at,
                   expires_at=excluded.expires_at""",
                (tool_key(tool.spec.name, norm), tool.spec.name, json.dumps(norm, sort_keys=True),
                 json.dumps(value, ensure_ascii=False), now, expires),
            )
