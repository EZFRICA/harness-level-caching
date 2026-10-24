"""Active set: the bounded working memory that is rendered into every prompt.

Two orders are kept apart on purpose:

- **Recency order** (an OrderedDict, i.e. a doubly linked list): drives eviction.
  Move-to-front on access only touches this order.
- **Render order**: (stability class, last modification). It never depends on
  access recency, so reading a block does not reshuffle the prompt. A modified
  block moves to the end of its class, which keeps every segment rendered before
  it byte-identical and therefore reusable from the KV / context cache (②).

Eviction is cache-aware: among the `victim_window` least recently used unpinned
blocks, the victim is the one rendered last, i.e. the one whose removal
invalidates the fewest cached prefix tokens. `victim_window=1` is plain LRU and
serves as the ablation baseline.
"""

from __future__ import annotations

from collections import OrderedDict

from cama.tokens import approx_tokens
from cama.types import Block


def render_key(block: Block) -> tuple:
    return (int(block.kind), block.modified_at, block.id)


def block_tokens(block: Block) -> int:
    return approx_tokens(render_block(block))


def render_block(block: Block) -> str:
    return f"### [{block.kind.label}] {block.title} (id={block.id})\n{block.content.strip()}\n"


class ActiveSet:
    def __init__(self, capacity: int, max_tokens: int, victim_window: int):
        self.capacity = capacity
        self.max_tokens = max_tokens
        self.victim_window = max(1, victim_window)
        self._lru: OrderedDict[str, Block] = OrderedDict()  # least recent first

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._lru

    def __len__(self) -> int:
        return len(self._lru)

    def get(self, block_id: str) -> Block | None:
        return self._lru.get(block_id)

    def ids(self) -> set[str]:
        return set(self._lru)

    def blocks(self) -> list[Block]:
        return list(self._lru.values())

    def tokens(self) -> int:
        return sum(block_tokens(b) for b in self._lru.values())

    def render_order(self) -> list[Block]:
        return sorted(self._lru.values(), key=render_key)

    def touch(self, block_id: str, ts: float) -> None:
        """Move-to-front in recency order only."""
        block = self._lru[block_id]
        block.accessed_at = ts
        self._lru.move_to_end(block_id)

    def load(self, blocks: list[Block]) -> None:
        """Restore from the store, oldest access first."""
        self._lru.clear()
        for b in sorted(blocks, key=lambda b: b.accessed_at):
            self._lru[b.id] = b

    def admit(self, block: Block) -> tuple[list[Block], bool]:
        """Insert (or replace) a block, evicting until both bounds hold.

        Returns (evicted, admitted). If only pinned blocks are left to evict and the
        bounds still do not hold, an unpinned newcomer is rejected rather than
        breaking a bound. Pinned blocks may exceed the bounds: pinning is an
        explicit operator decision.
        """
        self._lru[block.id] = block
        self._lru.move_to_end(block.id)
        evicted: list[Block] = []
        while len(self._lru) > self.capacity or self.tokens() > self.max_tokens:
            victim = self.choose_victim(protect=block.id)
            if victim is None:
                if not block.pinned:
                    del self._lru[block.id]
                    return evicted, False
                break
            del self._lru[victim.id]
            evicted.append(victim)
        return evicted, True

    def choose_victim(self, protect: str | None = None) -> Block | None:
        candidates = [b for b in self._lru.values() if not b.pinned and b.id != protect][: self.victim_window]
        if not candidates:
            return None
        order = {b.id: i for i, b in enumerate(self.render_order())}
        return max(candidates, key=lambda b: order[b.id])
