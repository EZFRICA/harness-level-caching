"""Invariants of the active set over random sequences of operations (decisions 0002, 0003)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from cama.memory.active_set import ActiveSet, render_key
from cama.types import Block, Kind

ops = st.lists(
    st.tuples(
        st.sampled_from(["admit", "touch", "modify"]),
        st.integers(min_value=0, max_value=15),             # block id
        st.sampled_from(list(Kind)),
        st.booleans(),                                      # pinned
        st.integers(min_value=1, max_value=60),             # content words
    ),
    min_size=1, max_size=60,
)


def _block(i: int, kind: Kind, pinned: bool, words: int, t: float) -> Block:
    return Block(id=f"b{i}", kind=kind, title=f"t{i}", content="w " * words, pinned=pinned,
                 created_at=t, modified_at=t, accessed_at=t)


def _replay(capacity, max_tokens, window, seq, check):
    s = ActiveSet(capacity, max_tokens, window)
    for step, (op, i, kind, pinned, words) in enumerate(seq):
        t = float(step + 1)
        bid = f"b{i}"
        if op == "admit" and bid not in s:
            s.admit(_block(i, kind, pinned, words, t))
        elif op == "touch" and bid in s:
            before = [b.id for b in s.render_order()]
            s.touch(bid, t)
            assert [b.id for b in s.render_order()] == before, "a read reordered the prompt"
        elif op == "modify" and bid in s:
            b = s.get(bid)
            b.content, b.modified_at, b.version = b.content + " x", t, b.version + 1
            s.admit(b)
        check(s)
    return s


@settings(max_examples=300, deadline=None)
@given(seq=ops, capacity=st.integers(2, 8), max_tokens=st.integers(30, 400), window=st.integers(1, 4))
def test_unpinned_blocks_never_push_the_set_past_its_bounds(seq, capacity, max_tokens, window):
    """Pinned blocks may exceed the bounds by operator decision; nothing else may."""
    from cama.memory.active_set import render_block
    from cama.tokens import approx_tokens

    def tok(b):  # independent of ActiveSet.tokens, so a broken count cannot pass its own check
        return approx_tokens(render_block(b))

    def check(s):
        pinned = [b for b in s.blocks() if b.pinned]
        assert len(s) <= max(capacity, len(pinned))
        assert sum(tok(b) for b in s.blocks()) <= max(max_tokens, sum(tok(b) for b in pinned))
    _replay(capacity, max_tokens, window, seq, check)


@settings(max_examples=300, deadline=None)
@given(seq=ops, window=st.integers(1, 4))
def test_render_order_is_always_kind_then_last_modification(seq, window):
    def check(s):
        rendered = s.render_order()
        assert rendered == sorted(rendered, key=render_key)
        kinds = [int(b.kind) for b in rendered]
        assert kinds == sorted(kinds), "a less stable kind was rendered before a more stable one"
    _replay(6, 10_000, window, seq, check)


@settings(max_examples=300, deadline=None)
@given(seq=ops, capacity=st.integers(1, 6))
def test_a_pinned_block_once_admitted_is_never_evicted(seq, capacity):
    admitted_pinned: set[str] = set()

    def check(s):
        for b in s.blocks():
            if b.pinned:
                admitted_pinned.add(b.id)
        assert admitted_pinned <= s.ids()
    _replay(capacity, 10_000, 2, seq, check)


@settings(max_examples=200, deadline=None)
@given(seq=ops)
def test_with_a_window_of_one_the_victim_is_the_least_recently_used(seq):
    """victim_window=1 must stay plain LRU: it is the ablation every cache-aware claim is read against."""
    s = ActiveSet(4, 10_000, 1)
    for step, (_, i, kind, _, words) in enumerate(seq):
        bid = f"b{i}"
        if bid in s:
            s.touch(bid, float(step + 1))
            continue
        lru = next((b.id for b in s.blocks() if not b.pinned), None)
        evicted, _ = s.admit(_block(i, kind, False, words, float(step + 1)))
        if evicted:
            assert evicted[0].id == lru
