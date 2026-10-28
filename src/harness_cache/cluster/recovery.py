"""Rebuild an agent's local store from the shared tier: latest snapshot, then the journal after it.

Replaying the journal gives every block's last state; vectors are recomputed through the
embedder (H2 makes that cheap when the space was seen before). Access times are not
journaled, so recency after a recovery is approximated by the last modification.
"""

from __future__ import annotations

import time

from harness_cache.store.sqlite_store import SQLiteStore

_BLOCK_OPS = ("seed", "create", "update")


def replay(blocks: dict, rows: list[dict]) -> dict:
    state = {k: dict(v) for k, v in blocks.items()}
    for r in rows:
        p = r["payload"]
        if r["op"] in _BLOCK_OPS:
            state[r["block_id"]] = {"kind": p["kind"], "title": p["title"], "content": p["content"],
                                    "pinned": p["pinned"], "version": p["version"],
                                    "active": p.get("active", False), "modified_at": p.get("modified_at", r["ts"])}
        elif r["op"] in ("page_in", "page_out") and r["block_id"] in state:
            state[r["block_id"]]["active"] = r["op"] == "page_in"
    return state


async def recover(coord, agent_id: str, store: SQLiteStore, embedder) -> dict:
    t0 = time.perf_counter()
    local_max = store.max_seq()
    remote_max = await coord.max_seq(agent_id)
    remote = await coord.journal(agent_id, 0)
    unsynced = len(store.unsynced(10_000))
    synced_max = local_max - unsynced if unsynced else local_max
    remote_seqs = {r["seq"] for r in remote}
    local_synced = {r[0] for r in store.db.execute("SELECT seq FROM journal WHERE synced=1").fetchall()}
    diverged = bool(local_synced - remote_seqs)  # rows we think are replicated but the shared tier lacks
    if remote_max <= synced_max and unsynced == 0 and not diverged:
        return {"action": "none", "local_seq": local_max, "remote_seq": remote_max,
                "ms": (time.perf_counter() - t0) * 1000}
    snap = await coord.latest_snapshot(agent_id)
    if snap and snap[0] > remote_max:  # a snapshot ahead of the journal describes rows that were lost
        snap = None
    base_seq, base = (snap[0], snap[1]["blocks"]) if snap else (0, {})
    state = replay(base, [r for r in remote if r["seq"] > base_seq])
    texts = {bid: f"{b['title']}\n{b['content']}" for bid, b in state.items()}
    ids = list(texts)
    vecs = await embedder.embed([texts[i] for i in ids], "document") if ids else []
    store.rebuild(state, dict(zip(ids, vecs)), remote)
    return {"action": "rebuilt", "local_seq": local_max, "remote_seq": remote_max, "from_snapshot": base_seq,
            "rows_replayed": len([r for r in remote if r["seq"] > base_seq]), "blocks": len(state),
            "discarded_unsynced": unsynced, "diverged": diverged, "ms": (time.perf_counter() - t0) * 1000}
