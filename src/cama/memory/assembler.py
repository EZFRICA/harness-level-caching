"""Context assembler (②): builds the request and measures prefix stability.

Request layout, most stable first:

    system = BASE_INSTRUCTIONS + memory blocks in render order
    messages = last `window_turns` exchanges + current user message

`stable_prefix_tokens` counts the leading segments that are byte-identical to the
previous request of the same session. It is an upper bound on what a provider
prefix cache (or llama.cpp KV reuse) can serve; provider-reported cached tokens
are recorded next to it as ground truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cama.memory.active_set import render_block
from cama.tokens import approx_tokens
from cama.types import Block, Message

BASE_INSTRUCTIONS = """You are a helpful assistant agent with a long-term memory.
The MEMORY section below holds what you know about the user and their ongoing work.
Use it when relevant, never invent memory content, and answer in the user's language.
Keep answers concise."""


@dataclass
class Assembled:
    system: str
    messages: list[Message]
    segments: list[tuple[str, int]] = field(default_factory=list)  # (hash, approx tokens)
    memory_tokens: int = 0
    window_tokens: int = 0
    total_tokens: int = 0
    stable_prefix_tokens: int = 0
    blocks_rendered: list[str] = field(default_factory=list)


def _h(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


class Assembler:
    def __init__(self) -> None:
        self._previous: dict[str, list[tuple[str, int]]] = {}

    def build(
        self, session: str, blocks: list[Block], history: list[Message], query: str
    ) -> Assembled:
        system_parts = [BASE_INSTRUCTIONS]
        segments = [(_h(BASE_INSTRUCTIONS), approx_tokens(BASE_INSTRUCTIONS))]
        memory_tokens = 0
        if blocks:
            header = "\n\n## MEMORY\n"
            system_parts.append(header)
            segments.append((_h(header), approx_tokens(header)))
            for b in blocks:
                text = render_block(b)
                system_parts.append(text)
                tok = approx_tokens(text)
                memory_tokens += tok
                segments.append((_h(text), tok))
        system = "".join(system_parts)

        messages = list(history) + [Message("user", query)]
        window_tokens = 0
        for m in messages:
            seg = f"{m.role}:{m.text}"
            tok = approx_tokens(m.text)
            window_tokens += tok
            segments.append((_h(seg), tok))

        prev = self._previous.get(session, [])
        stable = 0
        for a, b in zip(segments, prev):
            if a[0] != b[0]:
                break
            stable += a[1]
        self._previous[session] = segments

        return Assembled(
            system=system,
            messages=messages,
            segments=segments,
            memory_tokens=memory_tokens,
            window_tokens=window_tokens,
            total_tokens=sum(t for _, t in segments),
            stable_prefix_tokens=stable,
            blocks_rendered=[b.id for b in blocks],
        )
