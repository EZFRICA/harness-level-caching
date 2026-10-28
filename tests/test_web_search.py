"""The real web search tool (Tavily) behind a tape: offline, through a fake HTTP transport."""

import json
import sqlite3
import time

import httpx
import pytest

from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.metrics import RunRecorder
from harness_cache.teu.cache import ToolCache
from harness_cache.teu.tools import FaultConfig, ToolError, ToolRunner
from harness_cache.teu.web import WebSearch, WebTape, normalise, web_search_tool


async def _nosleep(_s):
    return None


def _tavily(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["query"])
        return httpx.Response(200, json={"results": [
            {"title": "T", "url": "https://example.org", "content": "x" * 900, "score": 0.9, "raw_content": None}]})
    return httpx.MockTransport(handler)


def _search(tmp_path, calls, mode="tape", cap=300, key="k"):
    return WebSearch(WebTape(tmp_path / "tape.jsonl"), mode, key, cap, transport=_tavily(calls), sleep=_nosleep)


def test_queries_differing_only_in_case_and_spaces_share_one_key():
    assert normalise({"query": "  DevFest   Abidjan 2026 "}) == normalise({"query": "devfest abidjan 2026"})


async def test_a_recorded_query_is_replayed_identically_without_a_live_call(tmp_path):
    calls = []
    s = _search(tmp_path, calls)
    first = await s({"query": "devfest abidjan"}, 0)
    again = await _search(tmp_path, calls)({"query": "devfest abidjan"}, 0)  # a new run, same tape
    assert again == first and calls == ["devfest abidjan"]
    assert len(first["results"][0]["content"]) == 500  # trimmed, so prompts stay comparable


async def test_replay_mode_never_calls_live(tmp_path):
    calls = []
    with pytest.raises(ToolError, match="not on the tape"):
        await _search(tmp_path, calls, mode="replay")({"query": "unseen"}, 0)
    assert calls == []


async def test_live_calls_stop_at_the_cap(tmp_path):
    calls = []
    s = _search(tmp_path, calls, cap=2)
    await s({"query": "a"}, 0)
    await s({"query": "b"}, 0)
    with pytest.raises(ToolError, match="cap 2"):
        await s({"query": "c"}, 0)
    assert calls == ["a", "b"]


def test_the_first_record_of_a_query_wins_and_survives_a_reload(tmp_path):
    tape = WebTape(tmp_path / "t.jsonl")
    tape.put("q", {"v": 1}, 10)
    tape.put("q", {"v": 2}, 20)
    assert WebTape(tmp_path / "t.jsonl").get("q")["value"] == {"v": 1}


async def test_a_real_tool_is_not_slept_by_the_runner(tmp_path):
    calls = []
    tool = web_search_tool(_search(tmp_path, calls))
    slow = tool.__class__(**{**tool.__dict__, "latency_ms": 60_000})  # a nominal latency, never slept
    t = time.perf_counter()
    await ToolRunner({"web_search": slow}, FaultConfig()).run("web_search", {"query": "x"})
    assert time.perf_counter() - t < 5


async def test_an_h3_hit_returns_exactly_what_the_miss_returned(tmp_path):
    calls = []
    tool = web_search_tool(_search(tmp_path, calls))
    cache = ToolCache(sqlite3.connect(":memory:"), clock=lambda: 1000.0)
    runner = ToolRunner({"web_search": tool}, FaultConfig())
    assert (await cache.alookup(tool, {"query": "Lomé"})).outcome == "miss"
    value = await runner.run("web_search", {"query": "Lomé"})
    await cache.astore(tool, {"query": "Lomé"}, value)
    hit = await cache.alookup(tool, {"query": " lomé "})
    assert hit.outcome == "hit" and hit.value == value and calls == ["lomé"]


def test_the_web_tool_is_offered_only_when_switched_on(tmp_path):
    def names(**kw):
        cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                     web_tape=tmp_path / "tape.jsonl", web_mode="replay", **kw)
        return [s.name for s in Engine(cfg, RunRecorder(tmp_path / "r", "x")).tool_specs]
    assert "web_search" not in names()  # existing scenarios keep the same tool prefix
    assert names(web_search=True)[-1] == "web_search"


def _fake_embed(table):
    import numpy as np

    async def embed(texts, kind):
        return np.asarray([table[t] for t in texts], dtype=np.float32)
    return embed


async def test_a_close_query_is_served_from_the_one_already_answered_and_recorded(tmp_path):
    calls = []
    table = {"devfest abidjan dates": [1.0, 0.0], "dates du devfest abidjan": [0.96, 0.28], "visa": [0.0, 1.0]}
    s = WebSearch(WebTape(tmp_path / "t.jsonl"), "tape", "k", 300, transport=_tavily(calls), sleep=_nosleep,
                  embed=_fake_embed(table), tau=0.9)
    first = await s({"query": "devfest abidjan dates"}, 0)
    again = await s({"query": "dates du devfest abidjan"}, 10)
    await s({"query": "visa"}, 20)
    assert again == first and calls == ["devfest abidjan dates", "visa"]
    assert s.semantic == [{"query": "dates du devfest abidjan", "matched": "devfest abidjan dates", "sim": 0.96}]


async def test_without_a_threshold_no_query_is_embedded(tmp_path):
    async def boom(texts, kind):
        raise AssertionError("embedded")
    calls = []
    s = WebSearch(WebTape(tmp_path / "t.jsonl"), "tape", "k", 300, transport=_tavily(calls), sleep=_nosleep,
                  embed=boom, tau=0.0)
    await s({"query": "a"}, 0)
    assert calls == ["a"]


async def test_a_semantic_match_expires_with_the_tool_ttl(tmp_path):
    calls = []
    table = {"q1": [1.0, 0.0], "q2": [1.0, 0.0]}
    s = WebSearch(WebTape(tmp_path / "t.jsonl"), "tape", "k", 300, transport=_tavily(calls), sleep=_nosleep,
                  embed=_fake_embed(table), tau=0.9)
    await s({"query": "q1"}, 0)
    await s({"query": "q2"}, 3600)
    assert calls == ["q1", "q2"] and s.semantic == []
