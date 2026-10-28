"""Ports: the core depends only on these protocols (⑦). Adapters live in harness_cache.adapters."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from harness_cache.types import LLMResult, Message


class LLM(Protocol):
    name: str

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False,
        cached_content: str | None = None,
    ) -> LLMResult:
        """`cached_content` names a provider cache holding exactly `system`; then `system` is not resent."""
        ...


class CachingLLM(LLM, Protocol):
    async def create_cache(self, system: str, ttl_s: int) -> tuple[str, int]:
        """Create a provider cache for `system`. Returns (cache name, cached tokens)."""
        ...

    async def delete_cache(self, name: str) -> None: ...


class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of unit-norm rows."""
        ...


class Archive(Protocol):
    """L4 replica. Receives journal entries; never on the hot path."""

    name: str

    async def push(self, entries: list[dict]) -> None: ...
