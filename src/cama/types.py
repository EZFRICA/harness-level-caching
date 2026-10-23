"""Core data types. No I/O, no network."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Kind(IntEnum):
    """Block stability class. The value is also the render rank:
    lower = more stable = earlier in the prompt (longer reusable prefix)."""

    FONDAMENTAL = 0
    PROJET = 1
    TEMP = 2

    @classmethod
    def parse(cls, value: str | int | Kind) -> Kind:
        if isinstance(value, Kind):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class Block:
    id: str
    kind: Kind
    title: str
    content: str
    pinned: bool = False
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    version: int = 1
    vector: np.ndarray | None = None

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"

    @property
    def segment_key(self) -> str:
        """Identity of the rendered segment: any content change yields a new key."""
        return f"{self.id}@{self.version}"


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
