"""Cost model, budget guard, and the prefix cache mode chosen from the model's capability."""

import asyncio
from pathlib import Path

import pytest

from harness_cache.config import Config
from harness_cache.cost import Budget, BudgetExceeded, MeteredLLM, load_pricing, run_cost
from harness_cache.engine import Engine
from harness_cache.metrics import RunRecorder
from harness_cache.scheduler import Scheduler
from harness_cache.types import LLMResult, Message

ROOT = Path(__file__).resolve().parent.parent
PRICE = {"read_on": "d", "source": "s", "status": "official", "input": 0.30, "output": 2.50,
         "cache_read": 0.03, "cache_storage_hour": 1.00, "cache_write": 0.30, "cache_write_status": "assumption"}


def _turn(i, prompt, cached, output):
    return {"turn": i, "ts": 1000.0 + i, "llm": {"prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": output}}


def test_a_run_is_priced_from_provider_tokens_hot_and_cold():
    manifest = {"config": {"llm": "gemini", "model": "m", "embedder": "onnx"}}
    turns = [_turn(1, 1_000_000, 0, 100_000), _turn(2, 1_000_000, 800_000, 0)]
    tasks = [
        {"task": "extract#1", "ok": True, "prompt_tokens": 1_000_000, "output_tokens": 0, "ts": 1001.5},
        {"task": "cache#1", "ok": True, "action": "create", "tokens": 1_000_000, "ts": 1002.0},
        {"task": "cache#2", "ok": True, "action": "create", "tokens": 1_000_000, "ts": 1002.0 + 3600},
    ]
    c = run_cost(manifest, turns, tasks, {"m": PRICE})
    assert c["priced"]
    assert c["hot"] == pytest.approx({"input": 0.30 + 0.06, "cache_read": 0.024, "output": 0.25})
    assert c["cold"]["extraction"] == pytest.approx(0.30)
    assert c["cold"]["cache_write"] == pytest.approx(0.60)
    assert c["cold"]["cache_storage"] == pytest.approx(1.00), "first cache alive one hour, second until the run ends"
    assert c["assumptions"] == ["cache creation priced as input (not stated by the provider)"]


def test_an_unpriced_model_has_no_cost_rather_than_a_zero_cost():
    c = run_cost({"config": {"llm": "gemini", "model": "x"}}, [_turn(1, 10, 0, 1)], [],
                 {"x": {"status": "unpriced"}})
    assert c == {"priced": False, "model": "x", "reason": "model not priced in pricing.toml"}


class _Paid:
    name = "paid"

    def __init__(self):
        self.calls = 0

    async def generate(self, system, messages, **kw):
        self.calls += 1
        return LLMResult("ok", prompt_tokens=1000, cached_tokens=0, output_tokens=100)


async def test_the_budget_guard_refuses_a_call_before_it_is_made():
    inner = _Paid()
    llm = MeteredLLM(inner, PRICE, Budget(max_usd=0.002), cache_ttl_s=60)
    await llm.generate("s", [Message("user", "hello")])
    assert inner.calls == 1 and llm.budget.spent_usd == pytest.approx((1000 * 0.30 + 100 * 2.50) / 1e6), \
        "the reservation is settled to the actual cost"
    with pytest.raises(BudgetExceeded):
        await llm.generate("s", [Message("user", "hello")])
    assert inner.calls == 1, "the refused call never reached the provider"


async def test_a_budget_stop_is_not_retried_by_the_cold_path():
    done = []
    s = Scheduler(on_done=done.append)
    s.start()

    async def over():
        raise BudgetExceeded("cap")

    await s.push("extract#1", over, 2)
    await asyncio.wait_for(s.drain(), 5)
    await s.stop()
    assert [d["attempt"] for d in done] == [1] and done[0]["ok"] is False


def _cfg(tmp_path, **kw):
    return Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a", **kw)


def test_a_paid_model_without_a_price_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        Engine(_cfg(tmp_path, llm="gemini", model="gemini-3.8-flash"), RunRecorder(tmp_path / "r", "x"))  # still unpriced


def test_the_prefix_cache_mode_follows_the_model_capability(tmp_path):
    pricing = load_pricing(ROOT / "pricing.toml")
    assert pricing["gemini-4-argon"]["explicit_cache"] is False
    eng = Engine(_cfg(tmp_path), RunRecorder(tmp_path / "r", "echo"))
    assert eng.cache_mode == "explicit"
    eng.model_info = pricing["gemini-4-argon"]
    assert eng._resolve_cache_mode() == "implicit"
    eng.cfg.prefix_cache = "explicit"
    with pytest.raises(ValueError, match="does not support explicit"):
        eng._resolve_cache_mode()
    eng.cfg.mode = "naive"
    assert eng._resolve_cache_mode() == "none"
