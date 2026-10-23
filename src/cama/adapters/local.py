"""Network-free adapters for development and plumbing tests.

They make the pipeline runnable at zero cost. They are NOT research baselines:
the echo LLM has no understanding and the hash embedder is a bag of hashed terms.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np

from cama.memory.extractor import EXTRACTOR_MARKER
from cama.tokens import approx_tokens, containment, lexical_set
from cama.types import LLMResult, Message


class HashEmbedder:
    """Feature hashing of terms and character trigrams, L2-normalised."""

    def __init__(self, dim: int = 256):
        self.name = "hash-v1"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        terms = lexical_set(text)
        feats = list(terms) + [t[i : i + 3] for t in terms for i in range(len(t) - 2)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            return v
        return v / n

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class EchoLLM:
    """Deterministic stand-in LLM.

    Chat: echoes the query and lists the memory block ids it was given.
    Extraction: appends the user text to the active block with the highest term
    containment, or creates a temp block when none matches.
    `ms_per_prompt_token` simulates a prefill cost proportional to prompt size.
    """

    def __init__(self, ms_per_prompt_token: float = 0.0):
        self.name = "echo"
        self.ms_per_prompt_token = ms_per_prompt_token

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False
    ) -> LLMResult:
        prompt_tokens = approx_tokens(system) + sum(approx_tokens(m.text) for m in messages)
        if self.ms_per_prompt_token:
            await asyncio.sleep(prompt_tokens * self.ms_per_prompt_token / 1000)
        if EXTRACTOR_MARKER in system:
            text = self._extract(messages[-1].text)
        else:
            ids = re.findall(r"\(id=([^)]+)\)", system)
            text = f"[echo] {messages[-1].text[:120]} | memory={','.join(ids) or 'none'}"
        return LLMResult(text=text, prompt_tokens=prompt_tokens, output_tokens=approx_tokens(text))

    @staticmethod
    def _extract(prompt: str) -> str:
        user = prompt.split("USER:", 1)[-1].split("ASSISTANT:", 1)[0].strip()
        blocks = re.findall(r"- id=(\S+) kind=\S+ title='([^']*)': (.*)", prompt)
        q = lexical_set(user)
        best, best_score = None, 0.0
        for bid, title, content in blocks:
            score, overlap = containment(q, lexical_set(f"{title} {content}"))
            if overlap >= 2 and score > best_score:
                best, best_score = bid, score
        if best and best_score >= 0.3:
            return json.dumps({"ops": [{"op": "append", "id": best, "text": user}]})
        title = " ".join(sorted(q, key=len, reverse=True)[:3]) or "note"
        return json.dumps({"ops": [{"op": "create", "kind": "temp", "title": title, "content": user}]})


class FileArchive:
    """L4 replica simulated as an append-only JSONL file on its own volume."""

    def __init__(self, directory: Path, name: str = "replica"):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{name}.jsonl"
        self.name = f"file:{self.path}"

    async def push(self, entries: list[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
