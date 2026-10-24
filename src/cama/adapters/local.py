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
from cama.types import LLMResult, Message, ToolCall, ToolSpec


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


# A provider's caches are global to the API key, not to one client: workers of a cluster can
# use a cache another worker created. The echo provider reproduces that within the process.
_PROVIDER_CACHES: dict[str, tuple[str, tuple]] = {}
_CACHE_SEQ = __import__("itertools").count(1)


class ModelError(RuntimeError):
    """A transient provider failure (simulated)."""


_CITIES = ("Abidjan", "Nantes", "Paris", "Cotonou", "Dakar")


def _echo_tool_calls(text: str, offered: set[str]) -> list[ToolCall]:
    """Deterministic tool use for the echo model: keyword rules, one round per turn."""
    low = text.lower()
    city = next((c for c in _CITIES if c.lower() in low), "Abidjan")
    calls: list[tuple[str, dict]] = []
    if any(w in low for w in ("météo", "meteo", "weather", "pleuvoir", "pluie")):
        calls.append(("weather", {"city": city}))
    if any(w in low for w in ("taux", "convert", "franc cfa", "xof")):
        calls.append(("fx_rate", {"base": "EUR", "quote": "XOF"}))
    expr = re.search(r"(\d[\d\s.]*[-+*/][\d\s.+*/()-]*\d)", text)
    if any(w in low for w in ("calcule", "combien font", "compute")) and expr:
        calls.append(("calculator", {"expression": expr.group(1)}))
    if any(w in low for w in ("mes notes", "dans les notes", "cherche")):
        calls.append(("search_notes", {"query": " ".join(lexical_set(text))[:60]}))
    if any(w in low for w in ("réserve", "reserve", "book")):
        calls.append(("book_hotel", {"city": city, "nights": 3}))
    return [ToolCall(id=f"echo-{i}", name=n, args=a) for i, (n, a) in enumerate(calls) if n in offered]


class EchoLLM:
    """Deterministic stand-in LLM.

    Chat: echoes the query and lists the memory block ids it was given; calls tools by
    keyword rules when tools are offered, then answers from their results.
    Extraction: appends the user text to the active block with the highest term
    containment, or creates a temp block when none matches.
    `ms_per_prompt_token` simulates a prefill cost proportional to uncached prompt size.
    """

    def __init__(self, ms_per_prompt_token: float = 0.0, base_ms: float = 0.0, ms_per_output_token: float = 0.0,
                 cached_extra_ms: float = 0.0, error_rate: float = 0.0, seed: int = 11):
        """Latency model: base + uncached prompt + output (+ extra when served from a cache).
        Fitted on real gemini-3.5-flash-lite calls (docs/measured, ledger): about 1000 ms base,
        3 ms per output token, prompt size negligible at these sizes, R2 0.19 (a weak fit)."""
        import random
        self.name = "echo"
        self.ms_per_prompt_token = ms_per_prompt_token
        self.base_ms = base_ms
        self.ms_per_output_token = ms_per_output_token
        self.cached_extra_ms = cached_extra_ms
        self.error_rate = error_rate
        self._rng = random.Random(seed)
        self._caches = _PROVIDER_CACHES  # one provider, shared by every client in the process

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False,
        cached_content: str | None = None, tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        cached = 0
        if cached_content is not None:
            system, tool_tuple = self._caches[cached_content]
            tools = list(tool_tuple)
            cached = approx_tokens(system) + _tools_tokens(tools)
        prompt_tokens = (approx_tokens(system) + _tools_tokens(tools)
                         + sum(approx_tokens(m.text or json.dumps(m.result or {})) for m in messages))
        if self.error_rate and self._rng.random() < self.error_rate:
            await asyncio.sleep(self.base_ms / 2000)
            raise ModelError("injected model failure (503)")
        calls: list[ToolCall] = []
        if EXTRACTOR_MARKER in system:
            text = self._extract(messages[-1].text)
        elif messages[-1].role == "tool":
            done = [m for m in messages if m.role == "tool"]
            text = "[echo] tools: " + "; ".join(f"{m.tool_name}={json.dumps(m.result, ensure_ascii=False)[:80]}"
                                                for m in done[-4:])
        else:
            calls = _echo_tool_calls(messages[-1].text, {t.name for t in tools or []})
            ids = re.findall(r"\(id=([^)]+)\)", system)
            text = "" if calls else f"[echo] {messages[-1].text[:120]} | memory={','.join(ids) or 'none'}"
        output_tokens = approx_tokens(text) + 8 * len(calls)
        delay = (self.base_ms + (prompt_tokens - cached) * self.ms_per_prompt_token
                 + output_tokens * self.ms_per_output_token + (self.cached_extra_ms if cached else 0.0))
        if delay:
            await asyncio.sleep(delay / 1000)
        return LLMResult(text=text, prompt_tokens=prompt_tokens, cached_tokens=cached,
                         output_tokens=output_tokens, tool_calls=tuple(calls))

    async def create_cache(self, system: str, tools: list[ToolSpec] | None, ttl_s: int) -> tuple[str, int]:
        tokens = approx_tokens(system) + _tools_tokens(tools)
        if tokens < 256:
            raise ValueError("Cached content is too small (echo minimum 256)")
        name = f"echo-cache/{next(_CACHE_SEQ)}"
        self._caches[name] = (system, tuple(tools or ()))
        return name, tokens

    async def delete_cache(self, name: str) -> None:
        self._caches.pop(name, None)

    @staticmethod
    def _extract(prompt: str) -> str:
        user = prompt.rsplit("USER:", 1)[-1].split("ASSISTANT:", 1)[0].strip()
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


def _tools_tokens(tools: list[ToolSpec] | None) -> int:
    return sum(approx_tokens(json.dumps([t.name, t.description, t.parameters])) for t in tools or [])


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
