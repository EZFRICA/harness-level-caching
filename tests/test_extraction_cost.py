"""Extraction calls are priced with their cached tokens, as the budget guard already counts them."""

from harness_cache.cost import run_cost

PRICE = {"read_on": "d", "source": "s", "status": "official", "input": 1.0, "output": 0.0,
         "cache_read": 0.1, "cache_storage_hour": 0.0, "cache_write": 0.0, "cache_write_status": "none"}
MANIFEST = {"config": {"llm": "openai_compat", "model": "m", "embedder": "onnx"}}


def _extract(prompt, cached=None):
    x = {"task": "extract#1", "ok": True, "prompt_tokens": prompt, "output_tokens": 0, "ts": 1.0}
    if cached is not None:
        x["cached_tokens"] = cached
    return x


def test_cached_extraction_tokens_are_priced_at_the_cache_read_rate():
    cost = run_cost(MANIFEST, [], [_extract(1_000_000, cached=800_000)], {"m": PRICE})
    assert round(cost["cold"]["extraction"], 6) == round(0.2 + 0.08, 6)


def test_a_run_recorded_before_the_field_existed_is_priced_as_before():
    cost = run_cost(MANIFEST, [], [_extract(1_000_000)], {"m": PRICE})
    assert round(cost["cold"]["extraction"], 6) == 1.0
