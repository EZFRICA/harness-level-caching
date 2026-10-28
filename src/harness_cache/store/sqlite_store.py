"""Local-first store (⑤): SQLite is the source of truth on the device.

- `blocks`  : every block ever created (L3), with its embedding and active flag.
- `journal` : append-only log of mutations; the L4 replica is fed from it.
- `history` : conversation turns per session.
- `meta`    : the embedding space the vectors belong to.

Vectors are checked for unit norm at this boundary and rejected otherwise: a
non-unit vector means an upstream bug (e.g. truncation without renormalisation)
and silently fixing it would hide a corrupted embedding space.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from harness_cache.types import Block, Kind, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocks (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    modified_at REAL NOT NULL,
    accessed_at REAL NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    vector BLOB,
    active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS journal (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    op TEXT NOT NULL,
    block_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS journal_unsynced ON journal(synced, seq);
CREATE INDEX IF NOT EXISTS history_session ON history(session, id);
"""

NORM_TOLERANCE = 1e-3


class EmbeddingSpaceMismatch(RuntimeError):
    pass


class SQLiteStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []

    # ── embedding space ────────────────────────────────────────────────────
    def bind_embedding_space(self, name: str, dim: int) -> None:
        """Refuse to mix vectors from different embedding spaces in one store."""
        space = f"{name}:{dim}"
        row = self.db.execute("SELECT value FROM meta WHERE key='embedding_space'").fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO meta VALUES ('embedding_space', ?)", (space,))
        elif row["value"] != space:
            raise EmbeddingSpaceMismatch(
                f"store holds vectors from '{row['value']}', embedder is '{space}'. "
                "Use another data volume or re-embed explicitly."
            )

    # ── blocks ─────────────────────────────────────────────────────────────
    def upsert(self, block: Block, op: str, *, active: bool | None = None) -> None:
        if block.vector is not None:
            norm = float(np.linalg.norm(block.vector))
            if abs(norm - 1.0) > NORM_TOLERANCE:
                raise ValueError(f"block {block.id}: vector norm {norm:.6f} is not 1")
        vec = None if block.vector is None else np.asarray(block.vector, dtype=np.float32).tobytes()
        prev = self.db.execute("SELECT active FROM blocks WHERE id=?", (block.id,)).fetchone()
        is_active = int(active) if active is not None else (prev["active"] if prev else 0)
        payload = {
            "kind": block.kind.label,
            "title": block.title,
            "content": block.content,
            "pinned": block.pinned,
            "version": block.version,
            "active": bool(is_active),
            "modified_at": block.modified_at,
        }
        with self.db:
            self.db.execute(
                """INSERT INTO blocks (id, kind, title, content, pinned, created_at, modified_at,
                                       accessed_at, version, vector, active)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, title=excluded.title,
                     content=excluded.content, pinned=excluded.pinned, modified_at=excluded.modified_at,
                     accessed_at=excluded.accessed_at, version=excluded.version,
                     vector=excluded.vector, active=excluded.active""",
                (
                    block.id, block.kind.label, block.title, block.content, int(block.pinned),
                    block.created_at, block.modified_at, block.accessed_at, block.version, vec, is_active,
                ),
            )
            self._journal(op, block.id, payload)
        self._matrix = None

    def set_active(self, block_id: str, active: bool) -> None:
        with self.db:
            self.db.execute("UPDATE blocks SET active=? WHERE id=?", (int(active), block_id))
            self._journal("page_in" if active else "page_out", block_id, {})

    def touch(self, block_id: str, ts: float) -> None:
        # Access times are not journaled: they are local cache metadata, not memory content.
        with self.db:
            self.db.execute("UPDATE blocks SET accessed_at=? WHERE id=?", (ts, block_id))

    def get(self, block_id: str) -> Block | None:
        row = self.db.execute("SELECT * FROM blocks WHERE id=?", (block_id,)).fetchone()
        return self._row_to_block(row) if row else None

    def active_blocks(self) -> list[Block]:
        rows = self.db.execute("SELECT * FROM blocks WHERE active=1 ORDER BY accessed_at").fetchall()
        return [self._row_to_block(r) for r in rows]

    def count(self) -> dict:
        r = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(active),0) a FROM blocks"
        ).fetchone()
        j = self.db.execute("SELECT COUNT(*) n, COALESCE(SUM(synced=0),0) u FROM journal").fetchone()
        return {"blocks": r["n"], "active": r["a"], "journal": j["n"], "journal_unsynced": j["u"]}

    # ── L3 search (in-process brute force) ────────────────────────────────
    def search(self, query: np.ndarray, k: int, exclude: set[str]) -> tuple[list[tuple[str, float]], int]:
        """Cosine top-k over all stored vectors. Returns (results, rows_scanned)."""
        if self._matrix is None:
            rows = self.db.execute("SELECT id, vector FROM blocks WHERE vector IS NOT NULL").fetchall()
            self._matrix_ids = [r["id"] for r in rows]
            self._matrix = (
                np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, query.shape[0]), dtype=np.float32)
            )
        if self._matrix.shape[0] == 0:
            return [], 0
        scores = self._matrix @ query.astype(np.float32)
        order = np.argsort(-scores)
        out = []
        for i in order:
            bid = self._matrix_ids[i]
            if bid in exclude:
                continue
            out.append((bid, float(scores[i])))
            if len(out) >= k:
                break
        return out, int(self._matrix.shape[0])

    # ── journal (L4 feed) ──────────────────────────────────────────────────
    def _journal(self, op: str, block_id: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO journal (ts, op, block_id, payload) VALUES (?,?,?,?)",
            (time.time(), op, block_id, json.dumps(payload, ensure_ascii=False)),
        )

    def unsynced(self, limit: int = 200) -> list[dict]:
        rows = self.db.execute(
            "SELECT seq, ts, op, block_id, payload FROM journal WHERE synced=0 ORDER BY seq LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"seq": r["seq"], "ts": r["ts"], "op": r["op"], "block_id": r["block_id"],
             "payload": json.loads(r["payload"])}
            for r in rows
        ]

    def unsynced_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM journal WHERE synced=0").fetchone()[0]

    def max_seq(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM journal").fetchone()[0]

    def snapshot_data(self) -> dict:
        """The state of every block without vectors: enough to rebuild, with re-embedding."""
        rows = self.db.execute(
            "SELECT id, kind, title, content, pinned, version, active, modified_at FROM blocks").fetchall()
        return {"seq": self.max_seq(), "blocks": {r["id"]: {
            "kind": r["kind"], "title": r["title"], "content": r["content"], "pinned": bool(r["pinned"]),
            "version": r["version"], "active": bool(r["active"]), "modified_at": r["modified_at"]} for r in rows}}

    def rebuild(self, blocks: dict, vectors: dict, journal_rows: list[dict]) -> None:
        """Replace the local state with a recovered one. Nothing here is journaled again: the
        recovered rows are imported as already synced, so the numbering continues after them."""
        with self.db:
            self.db.execute("DELETE FROM blocks")
            self.db.execute("DELETE FROM journal")
            for bid, b in blocks.items():
                vec = vectors.get(bid)
                self.db.execute(
                    "INSERT INTO blocks (id, kind, title, content, pinned, created_at, modified_at, accessed_at, "
                    "version, vector, active) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (bid, b["kind"], b["title"], b["content"], int(b["pinned"]), b["modified_at"], b["modified_at"],
                     b["modified_at"], b["version"], None if vec is None else vec.astype("float32").tobytes(),
                     int(b["active"])))
            self.db.executemany(
                "INSERT INTO journal (seq, ts, op, block_id, payload, synced) VALUES (?,?,?,?,?,1)",
                [(r["seq"], r["ts"], r["op"], r["block_id"], json.dumps(r["payload"], ensure_ascii=False))
                 for r in journal_rows])
        self._matrix = None

    def mark_synced(self, seqs: list[int]) -> None:
        if not seqs:
            return
        with self.db:
            self.db.executemany("UPDATE journal SET synced=1 WHERE seq=?", [(s,) for s in seqs])

    # ── history ────────────────────────────────────────────────────────────
    def append_history(self, session: str, role: str, text: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO history (session, role, text, ts) VALUES (?,?,?,?)",
                (session, role, text, time.time()),
            )

    def history(self, session: str, last: int | None = None) -> list[Message]:
        if last is None:
            rows = self.db.execute(
                "SELECT role, text FROM history WHERE session=? ORDER BY id", (session,)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT role, text FROM (SELECT id, role, text FROM history WHERE session=? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session, last),
            ).fetchall()
        return [Message(r["role"], r["text"]) for r in rows]

    def reset(self) -> None:
        with self.db:
            for t in ("blocks", "journal", "history", "meta"):
                self.db.execute(f"DELETE FROM {t}")
        self._matrix = None

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> Block:
        vec = None if row["vector"] is None else np.frombuffer(row["vector"], dtype=np.float32).copy()
        return Block(
            id=row["id"], kind=Kind.parse(row["kind"]), title=row["title"], content=row["content"],
            pinned=bool(row["pinned"]), created_at=row["created_at"], modified_at=row["modified_at"],
            accessed_at=row["accessed_at"], version=row["version"], vector=vec,
        )
