"""Gemini adapters (online profile)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from google import genai
from google.genai import types

from cama.types import LLMResult, Message


class GeminiLLM:
    def __init__(self, model: str, api_key: str):
        self.name = f"gemini:{model}"
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False
    ) -> LLMResult:
        contents = [
            types.Content(role="user" if m.role == "user" else "model", parts=[types.Part(text=m.text)])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        r = await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)
        u = r.usage_metadata
        return LLMResult(
            text=r.text or "",
            prompt_tokens=(u.prompt_token_count or 0) if u else 0,
            cached_tokens=(u.cached_content_token_count or 0) if u else 0,
            output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
        )


class GeminiEmbedder:
    """gemini-embedding-001 with Matryoshka truncation.

    Truncated vectors are NOT unit norm; they are renormalised here, explicitly,
    because this adapter is the one that truncates. The store still rejects any
    non-unit vector that reaches it by another route.
    """

    def __init__(self, model: str, dim: int, api_key: str):
        self.name = f"gemini:{model}"
        self.dim = dim
        self.model = model
        self.client = genai.Client(api_key=api_key)

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        r = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dim,
                task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT",
            ),
        )
        m = np.asarray([e.values for e in r.embeddings], dtype=np.float32)
        if m.shape != (len(texts), self.dim):
            raise ValueError(f"unexpected embedding shape {m.shape}, wanted ({len(texts)}, {self.dim})")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero embedding vector returned")
        return m / norms
