"""The compact baseline: full history until a threshold, then the oldest part folded into a summary."""

import asyncio

from cama.config import Config
from cama.engine import Engine
from cama.metrics import RunRecorder


def _engine(tmp_path, **kw):
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 mode="compact", tools=False, **kw)
    return Engine(cfg, RunRecorder(tmp_path / "r", "x"))


async def test_compact_folds_old_exchanges_into_a_summary_and_keeps_the_recent_ones(tmp_path):
    eng = _engine(tmp_path, compact_at=60, compact_keep=1)
    await eng.start()
    try:
        for i in range(6):
            await eng.turn(f"Fact {i}: my locker code is {1000 + i}. " + "pad " * 30)
            await eng.drain()
        assert eng._summary  # a fold happened
        assert eng._compact_from > 0
        kept = eng.store.history(eng.session)[eng._compact_from:]
        assert len(kept) < 12 and kept[-1].role == "assistant"
    finally:
        await eng.close()


async def test_compact_does_not_extract_memory(tmp_path):
    eng = _engine(tmp_path, compact_at=10_000)
    await eng.start()
    try:
        await eng.turn("My cat is named Biscotte.")
        await eng.drain()
        assert eng._pending == [] and not eng._summary
    finally:
        await eng.close()


def test_compaction_calls_are_priced_on_the_cold_path():
    from cama.cost import run_cost
    p = {"read_on": "d", "source": "s", "status": "official", "input": 1.0, "output": 0.0, "cache_read": 0.1,
         "cache_storage_hour": 0.0, "cache_write": 0.0, "cache_write_status": "none"}
    tasks = [{"task": "compact#9", "ok": True, "prompt_tokens": 1_000_000, "cached_tokens": 0, "output_tokens": 0, "ts": 1.0}]
    cost = run_cost({"config": {"llm": "x", "model": "m", "embedder": "onnx"}}, [], tasks, {"m": p})
    assert round(cost["cold"]["compaction"], 6) == 1.0
