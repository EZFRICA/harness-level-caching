"""Gemini adapters (online profile).

Tool declarations are part of the cacheable prefix: Gemini refuses a request that sets
`system_instruction` or `tools` next to `cached_content`, so both go into the cache and the
request carries only the cache name and the turn's contents.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from google import genai
from google.genai import types

from harness_cache.types import LLMResult, Message, ToolCall, ToolSpec


def _tools(specs: list[ToolSpec] | None) -> list[types.Tool] | None:
    if not specs:
        return None
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=s.name, description=s.description, parameters_json_schema=s.parameters)
        for s in specs
    ])]


def _contents(messages: list[Message]) -> list[types.Content]:
    out: list[types.Content] = []
    pending: list[types.Part] = []
    for m in messages:
        if m.role == "tool":
            pending.append(types.Part(function_response=types.FunctionResponse(
                id=m.tool_call_id or None, name=m.tool_name, response=m.result or {})))
            continue
        if pending:
            out.append(types.Content(role="user", parts=pending))
            pending = []
        if m.role == "assistant" and m.raw is not None:
            out.append(m.raw)  # replays thought signatures and function calls exactly
        elif m.role == "assistant":
            parts = [types.Part(text=m.text)] if m.text else []
            parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                      for c in m.tool_calls]
            # A model turn with no text and no call (Argon once billed 275 tokens of reasoning and
            # returned no answer) is replayed as a placeholder: Vertex refuses a Content without parts.
            out.append(types.Content(role="model", parts=parts or [types.Part(text="(no answer)")]))
        else:
            out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
    if pending:
        out.append(types.Content(role="user", parts=pending))
    return out


class GeminiLLM:
    def __init__(self, model: str, api_key: str, vertex_project: str = "", token_file: str = ""):
        self.name = f"gemini:{model}" if not vertex_project else f"vertex:{model}"
        self.model = model
        self.vertex_project = vertex_project
        self.token_file = token_file
        self._token = None
        self._client = None if vertex_project else genai.Client(api_key=api_key)

    @property
    def client(self) -> genai.Client:
        """Vertex AI: a client per access token, re-read from its file (tokens last an hour)."""
        if not self.vertex_project:
            return self._client
        with open(self.token_file) as f:
            token = f.read().strip()
        if token != self._token:
            from google.oauth2.credentials import Credentials
            self._client = genai.Client(vertexai=True, project=self.vertex_project, location="global",
                                        credentials=Credentials(token, quota_project_id=self.vertex_project))
            self._token = token
        return self._client

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False,
        cached_content: str | None = None, tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        config = types.GenerateContentConfig(
            system_instruction=None if cached_content else system,
            tools=None if cached_content else _tools(tools),
            cached_content=cached_content,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        r = await self.client.aio.models.generate_content(
            model=self.model, contents=_contents(messages), config=config)
        u = r.usage_metadata
        calls = tuple(ToolCall(id=c.id or f"call-{i}", name=c.name, args=dict(c.args or {}))
                      for i, c in enumerate(r.function_calls or []))
        raw = r.candidates[0].content if r.candidates else None
        text = "".join(p.text for p in (raw.parts or []) if p.text and not p.thought) if raw else ""
        return LLMResult(
            text=text,
            prompt_tokens=(u.prompt_token_count or 0) if u else 0,
            cached_tokens=(u.cached_content_token_count or 0) if u else 0,
            output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
            tool_calls=calls,
            raw=raw,
        )

    async def create_cache(self, system: str, tools: list[ToolSpec] | None, ttl_s: int) -> tuple[str, int]:
        c = await self.client.aio.caches.create(
            model=self.model,
            config=types.CreateCachedContentConfig(system_instruction=system, tools=_tools(tools), ttl=f"{ttl_s}s"),
        )
        tokens = (c.usage_metadata.total_token_count or 0) if c.usage_metadata else 0
        return c.name, tokens

    async def delete_cache(self, name: str) -> None:
        await self.client.aio.caches.delete(name=name)


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
