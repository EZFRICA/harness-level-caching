"""Memory extraction (④): runs on the cold path, never while the user waits.

The LLM proposes operations as JSON; the extractor validates and applies them.
Every applied operation goes through the store, hence into the journal.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from cama.memory.active_set import ActiveSet
from cama.ports import LLM, Embedder
from cama.store.sqlite_store import SQLiteStore
from cama.types import Block, Kind, Message

EXTRACTOR_MARKER = "CAMA_EXTRACTOR_V1"

EXTRACTOR_SYSTEM = f"""{EXTRACTOR_MARKER}
You maintain the long-term memory of an assistant agent.
Given the current memory blocks and the latest exchange, output ONLY a JSON object:
{{"ops": [
  {{"op": "append", "id": "<existing block id>", "text": "<new fact>"}},
  {{"op": "create", "kind": "fondamental|projet|temp", "title": "<short title>", "content": "<facts>"}}
]}}
Rules:
- Record only facts the USER stated or confirmed. Never record assistant suggestions.
- Do not repeat facts already present in a block.
- fondamental = stable facts about the user (identity, preferences, constraints).
  projet = an ongoing goal or piece of work. temp = details only useful for this session.
- Prefer "append" to an existing block on the same topic over "create".
- If nothing new was stated, output {{"ops": []}}."""


# Hypothesis H5 (switch CAMA_EXTRACT_REPLACE): a corrected fact replaces its line instead of being
# appended next to it. Measured on memory_fr_v1, where a value goes 30, 25, then 30 again: with
# append only, the block kept both and the stale one was the latest line.
REPLACE_RULES = """
- When the user corrects or changes a fact, replace the line that holds the old value:
  {"op": "replace", "id": "<existing block id>", "old": "<that line, verbatim>", "text": "<the fact now>"}
  Never append a fact that contradicts a line already in the block."""


@dataclass
class ExtractionReport:
    ops_proposed: int = 0
    appended: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    replace_missed: int = 0  # the old line was not found: the fact was appended instead
    invalid_ops: int = 0
    parse_error: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    exchanges: int = 1


def _slug(title: str, existing: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "block"
    slug, n = base, 2
    while slug in existing:
        slug, n = f"{base}-{n}", n + 1
    return slug


def replace_line(content: str, old: str, text: str) -> tuple[str, bool]:
    """Replace the first line containing `old` (bullet and spaces ignored); append when absent."""
    new = f"- {text.strip().removeprefix('- ')}"
    target = " ".join(old.strip().removeprefix("- ").split())
    lines = content.rstrip().split("\n")
    for i, line in enumerate(lines):
        if target and target in " ".join(line.strip().removeprefix("- ").split()):
            lines[i] = new
            return "\n".join(lines), True
    return f"{content.rstrip()}\n{new}", False


def _render_memory(active: ActiveSet) -> str:
    if not len(active):
        return "(empty)"
    return "\n".join(
        f"- id={b.id} kind={b.kind.label} title={b.title!r}: {b.content.strip()[:300]}"
        for b in active.render_order()
    )


class Extractor:
    def __init__(self, llm: LLM, embedder: Embedder, store: SQLiteStore, active: ActiveSet,
                 replace: bool = False):
        self.replace = replace
        self.system = EXTRACTOR_SYSTEM + (REPLACE_RULES if replace else "")
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.active = active

    async def run(self, user_text: str, assistant_text: str) -> ExtractionReport:
        return await self.run_many([(user_text, assistant_text)])

    async def run_many(self, exchanges: list[tuple[str, str]]) -> ExtractionReport:
        """One extraction call for one or several exchanges, oldest first."""
        rep = ExtractionReport()
        rep.exchanges = len(exchanges)
        convo = "\n\n".join(f"USER: {u}\n\nASSISTANT: {a[:800]}" for u, a in exchanges)
        prompt = f"CURRENT MEMORY BLOCKS:\n{_render_memory(self.active)}\n\n{convo}"
        result = await self.llm.generate(
            self.system, [Message("user", prompt)], temperature=0.0, json_mode=True
        )
        rep.prompt_tokens, rep.output_tokens = result.prompt_tokens, result.output_tokens
        rep.cached_tokens = result.cached_tokens  # the instructions and memory open the prompt: providers cache them
        try:
            raw = result.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            ops = json.loads(raw).get("ops", [])
            if not isinstance(ops, list):
                raise ValueError("ops is not a list")
        except (ValueError, AttributeError):
            rep.parse_error = True
            return rep
        rep.ops_proposed = len(ops)

        now = time.time()
        for op in ops:
            if not isinstance(op, dict):
                rep.invalid_ops += 1
                continue
            if self.replace and op.get("op") == "replace" and op.get("text"):
                block = self.active.get(op.get("id", "")) or self.store.get(op.get("id", ""))
                if block is None:
                    rep.invalid_ops += 1
                    continue
                block.content, found = replace_line(block.content, str(op.get("old", "")), op["text"])
                rep.replace_missed += 0 if found else 1
                block.version += 1
                block.modified_at = now
                block.vector = (await self.embedder.embed([block.text], "document"))[0]
                await self._commit(block, "update", rep)
                rep.replaced.append(block.id)
            elif op.get("op") == "append" and op.get("text"):
                block = self.active.get(op.get("id", "")) or self.store.get(op.get("id", ""))
                if block is None:
                    rep.invalid_ops += 1
                    continue
                block.content = f"{block.content.rstrip()}\n- {op['text'].strip().removeprefix('- ')}"
                block.version += 1
                block.modified_at = now
                block.vector = (await self.embedder.embed([block.text], "document"))[0]
                await self._commit(block, "update", rep)
                rep.appended.append(block.id)
            elif op.get("op") == "create" and op.get("title") and op.get("content"):
                try:
                    kind = Kind.parse(op.get("kind", "temp"))
                except KeyError:
                    kind = Kind.TEMP
                existing = self.active.ids() | {b for b in rep.created}
                block = Block(
                    id=_slug(op["title"], existing | self._all_ids()),
                    kind=kind,
                    title=op["title"].strip()[:80],
                    content=f"- {op['content'].strip()}",
                    created_at=now, modified_at=now, accessed_at=now,
                )
                block.vector = (await self.embedder.embed([block.text], "document"))[0]
                await self._commit(block, "create", rep)
                rep.created.append(block.id)
            else:
                rep.invalid_ops += 1
        return rep

    async def _commit(self, block: Block, op: str, rep: ExtractionReport) -> None:
        evicted, admitted = self.active.admit(block)
        for v in evicted:
            self.store.set_active(v.id, False)
            rep.evicted.append(v.id)
        if not admitted:
            rep.rejected.append(block.id)
        self.store.upsert(block, op, active=admitted)

    def _all_ids(self) -> set[str]:
        return {r[0] for r in self.store.db.execute("SELECT id FROM blocks").fetchall()}
