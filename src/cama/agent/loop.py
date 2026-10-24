"""The agent loop: model, tools, model again, until an answer or the step limit.

Every step sends the same cacheable prefix (system and tool declarations) and appends only
the step's own messages at the end, so an explicit prefix cache serves every step of the
turn. Tool calls of one step run concurrently; each goes through the H3 cache first.
A failing tool is retried a bounded number of times, then its error is handed to the model
as the tool's result: the turn completes either way.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from cama.ports import LLM
from cama.teu.cache import ToolCache
from cama.teu.tools import ToolError, ToolRunner
from cama.types import Message, ToolCall, ToolSpec


@dataclass
class LoopResult:
    text: str
    steps: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    stopped: str = "answer"  # answer | max_steps
    model_retries: int = 0
    cache_error: str | None = None  # the cache name failed and the turn continued without it


class AgentLoop:
    def __init__(self, llm: LLM, runner: ToolRunner, cache: ToolCache, max_steps: int = 4,
                 max_attempts: int = 2, temperature: float = 0.7, llm_attempts: int = 2):
        self.llm = llm
        self.runner = runner
        self.cache = cache
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.llm_attempts = llm_attempts
        self.model_retries = 0

    async def run(self, system: str, messages: list[Message], tools: list[ToolSpec] | None,
                  cached_content: str | None) -> LoopResult:
        res = LoopResult(text="")
        msgs = list(messages)
        self.cache_name = cached_content
        for step in range(1, self.max_steps + 1):
            t = time.perf_counter()
            out = await self._generate(system, msgs, tools, cached_content, res)
            ms = (time.perf_counter() - t) * 1000
            res.llm_ms += ms
            res.prompt_tokens += out.prompt_tokens
            res.cached_tokens += out.cached_tokens
            res.output_tokens += out.output_tokens
            res.steps.append({"step": step, "llm_ms": ms, "prompt_tokens": out.prompt_tokens,
                              "cached_tokens": out.cached_tokens, "output_tokens": out.output_tokens,
                              "tool_calls": [c.name for c in out.tool_calls]})
            res.text = out.text or res.text
            if not out.tool_calls:
                return res
            msgs.append(Message("assistant", text=out.text, tool_calls=out.tool_calls, raw=out.raw))
            t = time.perf_counter()
            results = await asyncio.gather(*(self._call(c, step) for c in out.tool_calls))
            res.tool_ms += (time.perf_counter() - t) * 1000
            for call, (value, rec) in zip(out.tool_calls, results):
                res.tools.append(rec)
                msgs.append(Message("tool", tool_name=call.name, tool_call_id=call.id, result=value))
        res.stopped = "max_steps"
        return res

    async def _generate(self, system, msgs, tools, cached_content, res: LoopResult):
        """One model call of one step, retried in place: the step's messages and the tool results
        already obtained are kept, so a retry never runs a tool again. A failure while sending a
        cache name falls back to the same call without it (the cache may have expired or been
        dropped by another worker). Review finding: the former fallback replayed the whole loop
        and booked the hotel twice."""
        for attempt in range(1, self.llm_attempts + 1):
            try:
                return await self.llm.generate(system, msgs, temperature=self.temperature,
                                               cached_content=self.cache_name, tools=tools)
            except Exception as e:
                if getattr(e, "retryable", True) is False:
                    raise
                if self.cache_name is not None:
                    res.cache_error = repr(e)[:200]
                    self.cache_name = None
                    continue
                if attempt == self.llm_attempts:
                    raise
                res.model_retries += 1
                await asyncio.sleep(0.2 * attempt)
        return await self.llm.generate(system, msgs, temperature=self.temperature, cached_content=None, tools=tools)

    async def _call(self, call: ToolCall, step: int) -> tuple[dict, dict]:
        rec = {"step": step, "tool": call.name, "outcome": "miss", "cause": "cold", "ms": 0.0,
               "saved_ms": 0.0, "attempts": 0, "error": None}
        tool = self.runner.tools.get(call.name)
        if tool is None:
            rec.update(outcome="error", cause="unknown_tool", error=f"unknown tool {call.name}")
            return {"error": rec["error"]}, rec
        t = time.perf_counter()
        look = await self.cache.alookup(tool, call.args)
        rec.update(outcome=look.outcome, cause=look.cause)
        if look.outcome == "hit":  # local or shared
            rec["ms"] = (time.perf_counter() - t) * 1000
            rec["saved_ms"] = tool.latency_ms * self.runner.faults.latency_scale
            return look.value, rec
        last_error = None
        # A tool with a side effect is never retried: it may have acted before failing.
        attempts = 1 if tool.cache_class == "never" else self.max_attempts
        for attempt in range(1, attempts + 1):
            rec["attempts"] = attempt
            try:
                value = await self.runner.run(call.name, call.args)
                await self.cache.astore(tool, call.args, value)
                rec["ms"] = (time.perf_counter() - t) * 1000
                return value, rec
            except (ToolError, KeyError, TypeError, ValueError, SyntaxError) as e:
                last_error = f"{type(e).__name__}: {e}"
                await asyncio.sleep(0.05 * attempt)
        rec.update(ms=(time.perf_counter() - t) * 1000, outcome="error", error=last_error)
        return {"error": last_error}, rec
