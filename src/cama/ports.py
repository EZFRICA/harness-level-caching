"""Ports: the core depends only on these protocols (⑦). Adapters live in cama.adapters."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from cama.types import LLMResult, Message


class LLM(Protocol):
    name: str

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False
    ) -> LLMResult: ...


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
