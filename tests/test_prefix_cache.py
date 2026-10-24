"""H4: the hot path only looks up; the cold path creates, replaces and gives up below the minimum."""

import time

import pytest

from cama.memory.prefix_cache import PrefixCache, prefix_hash


class FakeProvider:
    def __init__(self, minimum=10):
        self.minimum = minimum
        self.live: dict[str, str] = {}
        self.created = 0
        self.deleted: list[str] = []

    async def create_cache(self, system, tools, ttl_s):
        if len(system) < self.minimum:
            raise ValueError("Cached content is too small. total_token_count=3, min_total_token_count=1024")
        self.created += 1
        name = f"c/{self.created}"
        self.live[name] = system
        return name, len(system)

    async def delete_cache(self, name):
        self.deleted.append(name)
        self.live.pop(name)


async def test_lookup_finds_only_the_exact_prefix_that_was_cached():
    pc = PrefixCache(FakeProvider(), ttl_s=600)
    out = await pc.ensure("stable prefix A")
    assert out["action"] == "create" and out["tokens"] == len("stable prefix A")
    assert pc.lookup(prefix_hash("stable prefix A")).name == "c/1"
    assert pc.lookup(prefix_hash("stable prefix B")) is None


async def test_an_unchanged_prefix_is_reused_not_recreated():
    prov = FakeProvider()
    pc = PrefixCache(prov, ttl_s=600)
    await pc.ensure("stable prefix A")
    again = await pc.ensure("stable prefix A")
    assert again["action"] == "reuse" and prov.created == 1 and pc.creations == 1


async def test_a_new_prefix_replaces_and_deletes_the_previous_cache():
    prov = FakeProvider()
    pc = PrefixCache(prov, ttl_s=600)
    await pc.ensure("stable prefix A")
    out = await pc.ensure("stable prefix B")
    assert out["action"] == "create" and out["replaced"] == prefix_hash("stable prefix A")
    assert prov.deleted == ["c/1"] and list(prov.live) == ["c/2"]


async def test_a_prefix_below_the_provider_minimum_is_skipped_once_and_remembered():
    prov = FakeProvider(minimum=100)
    pc = PrefixCache(prov, ttl_s=600)
    first = await pc.ensure("short")
    second = await pc.ensure("short")
    assert first["action"] == second["action"] == "skip"
    assert first["reason"] == "below provider minimum" and prov.created == 0


async def test_a_cache_close_to_expiry_is_not_served():
    """Serving a cache that expires mid-request would fail the turn; the margin prevents it."""
    pc = PrefixCache(FakeProvider(), ttl_s=600, margin_s=30)
    await pc.ensure("stable prefix A")
    pc.current.expires_at = time.time() + 29
    assert pc.lookup(prefix_hash("stable prefix A")) is None
    pc.current.expires_at = time.time() + 31
    assert pc.lookup(prefix_hash("stable prefix A")) is not None


async def test_other_provider_errors_are_not_mistaken_for_too_small():
    class Broken(FakeProvider):
        async def create_cache(self, system, tools, ttl_s):
            raise RuntimeError("503 unavailable")
    pc = PrefixCache(Broken(), ttl_s=600)
    with pytest.raises(RuntimeError):
        await pc.ensure("stable prefix A")
    assert pc.unusable == {}


async def test_close_deletes_the_live_cache():
    prov = FakeProvider()
    pc = PrefixCache(prov, ttl_s=600)
    await pc.ensure("stable prefix A")
    await pc.close()
    assert prov.live == {} and pc.current is None


async def test_the_creation_record_carries_what_the_dashboard_and_cost_model_read():
    seen = {}

    class Recording(FakeProvider):
        async def create_cache(self, system, tools, ttl_s):
            seen["ttl"] = ttl_s
            return await super().create_cache(system, tools, ttl_s)

    pc = PrefixCache(Recording(), ttl_s=600)
    first = await pc.ensure("stable prefix A")
    second = await pc.ensure("stable prefix B")
    assert seen["ttl"] == 600
    assert set(first) == {"action", "hash", "tokens", "create_ms", "replaced", "creations"}
    assert first["hash"] == prefix_hash("stable prefix A") and first["replaced"] is None
    assert 0 <= first["create_ms"] < 1000 and first["creations"] == 1 and second["creations"] == 2
    assert pc.current.tokens == len("stable prefix B")
    reuse, skip = await pc.ensure("stable prefix B"), None
    assert reuse == {"action": "reuse", "hash": prefix_hash("stable prefix B")}
    pc2 = PrefixCache(FakeProvider(minimum=100), ttl_s=600)
    skip = await pc2.ensure("short")
    assert skip == {"action": "skip", "hash": prefix_hash("short"), "reason": "below provider minimum"}


async def test_the_same_text_with_other_tools_is_another_prefix():
    """Gemini caches system and tools together; a tool change must not reuse the old cache."""
    from cama.types import ToolSpec
    a = [ToolSpec("weather", "Weather.", {"type": "object"})]
    b = [ToolSpec("weather", "Weather forecast.", {"type": "object"})]
    assert prefix_hash("same", a) != prefix_hash("same", b) != prefix_hash("same")
    prov = FakeProvider()
    pc = PrefixCache(prov, ttl_s=600)
    await pc.ensure("stable prefix A", a)
    assert pc.lookup(prefix_hash("stable prefix A", a)) and not pc.lookup(prefix_hash("stable prefix A", b))
