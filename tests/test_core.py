import json
from pathlib import Path

import numpy as np
import pytest

from harness_cache.adapters.local import HashEmbedder
from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.memory.active_set import ActiveSet
from harness_cache.memory.assembler import Assembler
from harness_cache.metrics import RunRecorder
from harness_cache.store.sqlite_store import EmbeddingSpaceMismatch, SQLiteStore
from harness_cache.types import Block, Kind

SCENARIO = Path(__file__).parent.parent / "scenarios" / "assistant_fr_v1.jsonl"


def blk(i, kind=Kind.TEMP, pinned=False, content="x", t=None):
    t = float(i) if t is None else t
    return Block(id=f"b{i}", kind=kind, title=f"t{i}", content=content, pinned=pinned,
                 created_at=t, modified_at=t, accessed_at=t)


def test_render_order_ignores_access_recency():
    s = ActiveSet(capacity=10, max_tokens=10_000, victim_window=3)
    for b in [blk(1, Kind.TEMP), blk(2, Kind.FONDAMENTAL), blk(3, Kind.PROJET)]:
        s.admit(b)
    before = [b.id for b in s.render_order()]
    s.touch("b2", 99.0)
    assert [b.id for b in s.render_order()] == before == ["b2", "b3", "b1"]


def test_modified_block_moves_to_end_of_its_class():
    s = ActiveSet(capacity=10, max_tokens=10_000, victim_window=3)
    for b in [blk(1, Kind.PROJET), blk(2, Kind.PROJET), blk(3, Kind.TEMP)]:
        s.admit(b)
    b1 = s.get("b1")
    b1.modified_at, b1.version = 50.0, 2
    s.admit(b1)
    assert [b.id for b in s.render_order()] == ["b2", "b1", "b3"]


def test_cache_aware_eviction_prefers_latest_rendered_among_lru_window():
    s = ActiveSet(capacity=3, max_tokens=10_000, victim_window=2)
    s.admit(blk(1, Kind.TEMP))         # rendered last
    s.admit(blk(2, Kind.FONDAMENTAL))  # rendered first; least recently used together with b1
    s.admit(blk(3, Kind.PROJET))
    evicted, ok = s.admit(blk(4, Kind.PROJET))
    assert ok and [b.id for b in evicted] == ["b1"]


def test_pure_lru_when_window_is_one():
    s = ActiveSet(capacity=2, max_tokens=10_000, victim_window=1)
    s.admit(blk(1, Kind.FONDAMENTAL))
    s.admit(blk(2, Kind.TEMP))
    evicted, _ = s.admit(blk(3, Kind.TEMP))
    assert [b.id for b in evicted] == ["b1"]


def test_pinned_blocks_are_never_evicted_and_newcomer_rejected():
    s = ActiveSet(capacity=2, max_tokens=10_000, victim_window=3)
    s.admit(blk(1, pinned=True))
    s.admit(blk(2, pinned=True))
    evicted, ok = s.admit(blk(3))
    assert not ok and evicted == [] and s.ids() == {"b1", "b2"}


def _rendered_tokens(s):
    """Independent oracle: the tokens of what would actually be rendered."""
    from harness_cache.memory.assembler import render_system
    from harness_cache.tokens import approx_tokens
    return approx_tokens(render_system(s.render_order())) - approx_tokens(render_system([]))


def test_token_bound_is_enforced_on_what_is_actually_rendered():
    s = ActiveSet(capacity=100, max_tokens=60, victim_window=3)
    for i in range(5):
        s.admit(blk(i, content="mot " * 40))
    assert 0 < s.tokens() <= 60
    assert _rendered_tokens(s) <= 60 + 8  # header and rounding


def test_a_set_exactly_at_its_token_bound_keeps_every_block():
    one = ActiveSet(capacity=10, max_tokens=10_000, victim_window=3)
    one.admit(blk(1, content="a b c"))
    one.admit(blk(2, content="d e f"))
    exact = one.tokens()
    s = ActiveSet(capacity=10, max_tokens=exact, victim_window=3)
    s.admit(blk(1, content="a b c"))
    evicted, ok = s.admit(blk(2, content="d e f"))
    assert ok and evicted == [] and len(s) == 2


def test_a_pinned_newcomer_is_admitted_even_past_the_bounds():
    s = ActiveSet(capacity=1, max_tokens=10_000, victim_window=3)
    s.admit(blk(1, pinned=True))
    evicted, ok = s.admit(blk(2, pinned=True))
    assert ok and evicted == [] and s.ids() == {"b1", "b2"}


def test_touch_records_the_access_time_used_after_a_restart():
    s = ActiveSet(capacity=10, max_tokens=10_000, victim_window=1)
    s.admit(blk(1))
    s.admit(blk(2))
    s.touch("b1", 99.0)
    assert s.get("b1").accessed_at == 99.0
    restored = ActiveSet(capacity=2, max_tokens=10_000, victim_window=1)
    restored.load([s.get("b1"), s.get("b2")])
    evicted, _ = restored.admit(blk(3, t=100.0))
    assert [b.id for b in evicted] == ["b2"], "reload must restore recency from access times"
    assert restored.get("b1") is not None


def test_stable_prefix_measured_across_turns():
    a = Assembler()
    blocks = [blk(1, Kind.FONDAMENTAL, content="stable"), blk(2, Kind.TEMP, content="v1")]
    first = a.build("s", blocks, [], "q1")
    assert first.stable_prefix_tokens == 0
    second = a.build("s", blocks, [], "q2")
    assert 0 < second.stable_prefix_tokens < second.total_tokens


def test_store_rejects_non_unit_vectors_and_foreign_spaces(tmp_path):
    st = SQLiteStore(tmp_path / "s.sqlite")
    st.bind_embedding_space("hash-v1", 256)
    b = blk(1)
    b.vector = np.full(256, 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        st.upsert(b, "create")
    with pytest.raises(EmbeddingSpaceMismatch):
        st.bind_embedding_space("gemini:gemini-embedding-001", 768)


async def test_hash_embedder_unit_norm():
    m = await HashEmbedder().embed(["bonjour le monde", ""], "document")
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


@pytest.mark.parametrize("mode", ["extract", "topk", "naive"])
async def test_end_to_end_scenario_echo(tmp_path, mode):
    cfg = Config(mode=mode, data_dir=tmp_path / "data", runs_dir=tmp_path / "runs",
                 archive_dir=tmp_path / "archive", cache_admission="always")
    rec = RunRecorder(cfg.runs_dir, f"t-{mode}")
    rec.manifest(cfg.as_dict())
    eng = Engine(cfg, rec, session="t")
    await eng.start()
    steps = [json.loads(line) for line in SCENARIO.read_text().splitlines() if line.strip()]
    await eng.seed(steps[0]["seed"])
    tiers = []
    for step in steps[1:]:
        res = await eng.turn(step["user"])
        await eng.drain()
        if res.record["route"]:
            tiers.append(res.record["route"]["tier"])
        assert len(eng.active) <= cfg.capacity
    counts = eng.store.count()
    await eng.close()

    turns = [json.loads(x) for x in (cfg.runs_dir / f"t-{mode}" / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == len(steps) - 1
    if mode == "naive":
        assert turns[-1]["context"]["total_tokens_est"] > turns[0]["context"]["total_tokens_est"]
    else:
        assert "lexical" in tiers and "l3_fault" in tiers
        if mode == "extract":  # stable prefix => later turns are served from the prefix cache
            assert sum(t["prefix_cache"]["hit"] for t in turns) > 0
            assert sum(t["llm"]["cached_tokens"] for t in turns) > 0
        assert counts["journal_unsynced"] == 0  # cold path synced everything to L4
        replica = (cfg.archive_dir / f"{cfg.store_name}.jsonl").read_text().splitlines()
        assert len(replica) == counts["journal"]
