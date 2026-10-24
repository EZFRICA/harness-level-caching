"""OpenAI-compatible chat completions (Nebius Token Factory, a self-hosted vLLM or SGLang).

These servers cache prompt prefixes implicitly: there is no cache object to create, so this
adapter has no `create_cache` and the engine runs it in implicit mode (order kept stable,
provider counts recorded). The cached share comes back in
`usage.prompt_tokens_details.cached_tokens`.

Two things keep the measured prefix honest:
- serialisation is deterministic (tool results with sorted keys, assistant turns replayed as
  the server returned them), since one changed byte early in a prompt ends the cached prefix;
- an optional salt opens the system prompt. A shared serverless cache serves every customer
  of a model (Nebius docs, "Cache scope"), so without it a hit may come from someone else's
  identical prefix. The salt is constant within a run, so it never costs a hit the run earned.
"""

from __future__ import annotations

import json

import httpx

from cama.types import LLMResult, Message, ToolCall, ToolSpec


class ProviderError(RuntimeError):
    """A refused or failed request. Client errors other than 408 and 429 are not retried."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.retryable = status in (408, 429) or status >= 500


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tools_payload(specs: list[ToolSpec] | None) -> list[dict] | None:
    if not specs:
        return None
    return [{"type": "function", "function": {"name": s.name, "description": s.description,
                                              "parameters": s.parameters}} for s in specs]


def messages_payload(system: str, messages: list[Message], salt: str = "") -> list[dict]:
    head = f"[run {salt}]\n{system}" if salt else system
    out: list[dict] = [{"role": "system", "content": head}]
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": _dump(m.result or {})})
        elif m.role == "assistant" and isinstance(m.raw, dict):
            out.append(m.raw)  # exactly what the server returned, reasoning stripped
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": _dump(c.args)}}
                                     for c in m.tool_calls]
            out.append(msg)
        else:
            out.append({"role": "user", "content": m.text})
    return out


def parse_response(body: dict) -> LLMResult:
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": fn.get("arguments")}
        calls.append(ToolCall(id=c.get("id") or f"call-{i}", name=fn.get("name", ""),
                              args=args if isinstance(args, dict) else {"value": args}))
    raw = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        raw["tool_calls"] = msg["tool_calls"]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return LLMResult(
        text=msg.get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),  # reasoning included, as billed
        tool_calls=tuple(calls),
        raw=raw,
    )


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str, api_key: str = "", salt: str = "",
                 timeout_s: float = 120.0, transport: httpx.AsyncBaseTransport | None = None):
        if not base_url:
            raise ValueError("CAMA_OAI_BASE_URL is required for CAMA_LLM=openai_compat")
        self.name = f"openai_compat:{model}"
        self.model = model
        self.salt = salt
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s,
                                        transport=transport)

    async def generate(
        self, system: str, messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False,
        cached_content: str | None = None, tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        if cached_content is not None:
            raise ProviderError(400, "this provider has no explicit cache")
        payload: dict = {"model": self.model, "messages": messages_payload(system, messages, self.salt),
                         "temperature": temperature}
        spec = tools_payload(tools)
        if spec:
            payload["tools"] = spec
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = await self.client.post("/chat/completions", json=payload)
        if r.status_code != 200:
            raise ProviderError(r.status_code, r.text)
        return parse_response(r.json())
