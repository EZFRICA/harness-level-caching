"""The OpenAI-compatible adapter (Nebius, vLLM): offline, through a fake HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest

from cama.adapters import make_llm
from cama.adapters.openai_compat import OpenAICompatLLM, ProviderError, messages_payload, parse_response
from cama.config import Config
from cama.cost import load_pricing, price_for
from cama.types import Message, ToolCall, ToolSpec

ROOT = Path(__file__).resolve().parent.parent
SPEC = ToolSpec("weather", "Weather in a city", {"type": "object", "properties": {"city": {"type": "string"}}})


def _reply(content="ok", cached=None, tool_calls=None, prompt=100, completion=7):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    msg = {"role": "assistant", "content": content, "reasoning_content": "hidden"}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}], "usage": usage}


def _llm(handler, salt=""):
    return OpenAICompatLLM("m", "https://fake/v1", "k", salt, transport=httpx.MockTransport(handler))


def test_the_cached_share_is_read_from_the_provider_usage():
    out = parse_response(_reply(cached=64))
    assert (out.prompt_tokens, out.cached_tokens, out.output_tokens) == (100, 64, 7)
    assert parse_response(_reply()).cached_tokens == 0  # no details means no hit, not an error


def test_tool_results_serialise_the_same_bytes_whatever_the_key_order():
    a = messages_payload("s", [Message("tool", tool_call_id="c1", result={"b": 1, "a": "é"})])
    b = messages_payload("s", [Message("tool", tool_call_id="c1", result={"a": "é", "b": 1})])
    assert a == b and a[1]["content"] == '{"a":"é","b":1}'


def test_the_salt_opens_the_system_prompt_and_is_constant_within_a_run():
    first = messages_payload("SYS", [Message("user", text="hi")], salt="run-7")
    second = messages_payload("SYS", [Message("user", text="again")], salt="run-7")
    assert first[0]["content"].startswith("[run run-7]\n") and first[0] == second[0]
    assert messages_payload("SYS", [], salt="")[0]["content"] == "SYS"


def test_an_assistant_turn_is_replayed_as_returned_without_its_reasoning():
    calls = [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Lomé"}'}}]
    out = parse_response(_reply(content=None, tool_calls=calls))
    assert out.tool_calls == (ToolCall("c1", "weather", {"city": "Lomé"}),)
    assert "reasoning_content" not in out.raw and out.raw["tool_calls"] == calls
    sent = messages_payload("s", [Message("assistant", tool_calls=out.tool_calls, raw=out.raw)])
    assert sent[1] is out.raw  # the exact bytes the server produced, not a re-serialisation


async def test_a_request_carries_tools_and_json_mode_and_returns_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers.get("authorization")
        calls = [{"id": "c9", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]
        return httpx.Response(200, json=_reply(content="", tool_calls=calls, cached=32))

    out = await _llm(handler, salt="s1").generate("SYS", [Message("user", text="q")], tools=[SPEC], json_mode=True)
    assert seen["tools"][0]["function"]["name"] == "weather"
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["messages"][0]["content"].startswith("[run s1]")
    assert seen["auth"] == "Bearer k"
    assert out.tool_calls[0].name == "weather" and out.cached_tokens == 32


async def _nosleep(_s):
    return None


async def test_rate_limits_and_server_errors_are_retryable_and_client_errors_are_not():
    for status, retryable in ((429, True), (503, True), (400, False), (401, False)):
        llm = _llm(lambda r, s=status: httpx.Response(s, text="nope"))
        llm.sleep = _nosleep
        with pytest.raises(ProviderError) as e:
            await llm.generate("SYS", [Message("user", text="q")])
        assert e.value.retryable is retryable


async def test_an_explicit_cache_name_is_refused_since_the_provider_has_none():
    llm = _llm(lambda r: httpx.Response(200, json=_reply()))
    with pytest.raises(ProviderError):
        await llm.generate("SYS", [], cached_content="caches/x")
    assert not hasattr(llm, "create_cache")  # so the engine runs this model in implicit mode


def test_a_missing_key_variable_is_refused_by_name(monkeypatch):
    monkeypatch.delenv("CAMA_TEST_ABSENT_KEY", raising=False)
    cfg = Config(llm="openai_compat", model="m", oai_base_url="https://fake/v1", oai_key_env="CAMA_TEST_ABSENT_KEY")
    with pytest.raises(RuntimeError, match="CAMA_TEST_ABSENT_KEY is not set"):
        make_llm(cfg)


def test_the_nebius_model_is_priced_and_implicit_only():
    p = load_pricing(ROOT / "pricing.toml")["deepseek-ai/DeepSeek-V4-Flash-0731"]
    assert price_for({"x": p}, "x") is not None
    assert p["explicit_cache"] is False and p["cache_read"] < p["input"]


async def test_a_short_outage_is_ridden_out_and_counted():
    answers = iter([503, 503, 200])

    def handler(request):
        code = next(answers)
        return httpx.Response(code, json=_reply()) if code == 200 else httpx.Response(code, text="down")
    llm = _llm(handler)
    waits = []

    async def sleep(s):
        waits.append(s)
    llm.sleep = sleep
    out = await llm.generate("SYS", [Message("user", text="q")])
    assert out.text == "ok" and llm.retried == 2 and waits == [2.0, 4.0]


async def test_a_client_error_is_never_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad")
    llm = _llm(handler)
    llm.sleep = _nosleep
    with pytest.raises(ProviderError):
        await llm.generate("SYS", [Message("user", text="q")])
    assert len(calls) == 1


async def test_a_short_lived_token_is_read_from_its_file_before_every_request(tmp_path):
    key = tmp_path / "token"
    key.write_text("first\n")
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_reply())
    llm = OpenAICompatLLM("m", "https://fake/v1", "", "", transport=httpx.MockTransport(handler), key_file=str(key))
    await llm.generate("SYS", [Message("user", text="q")])
    key.write_text("second")
    await llm.generate("SYS", [Message("user", text="q")])
    assert seen == ["Bearer first", "Bearer second"]


def test_gemini_on_vertex_builds_a_new_client_when_the_token_changes(tmp_path):
    from cama.adapters.gemini import GeminiLLM
    tok = tmp_path / "token"
    tok.write_text("t1")
    llm = GeminiLLM("gemini-3.5-flash-lite", "", vertex_project="p", token_file=str(tok))
    first = llm.client
    assert llm.client is first and llm.name == "vertex:gemini-3.5-flash-lite"
    tok.write_text("t2")
    assert llm.client is not first


def test_an_empty_model_turn_is_replayed_with_a_placeholder_part():
    from cama.adapters.gemini import _contents
    out = _contents([Message("user", text="q"), Message("assistant", text=""), Message("user", text="again")])
    assert out[1].role == "model" and out[1].parts and out[1].parts[0].text == "(no answer)"
