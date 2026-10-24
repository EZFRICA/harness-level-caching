"""Switches under test (window epoch, extraction every N turns): defaults keep the old behaviour."""

from cama.config import Config
from cama.engine import window_size


def test_by_default_the_window_slides_every_turn():
    assert [window_size(Config(window_turns=2), t) for t in range(5)] == [2, 2, 2, 2, 2]


def test_with_an_epoch_the_window_grows_then_drops_at_once():
    cfg = Config(window_turns=2, window_epoch=3)
    assert [window_size(cfg, t) for t in range(7)] == [2, 3, 4, 2, 3, 4, 2]


def test_an_epoch_turns_the_window_into_an_append_only_suffix_between_drops():
    cfg = Config(window_turns=2, window_epoch=3)
    starts = [t - window_size(cfg, t) for t in range(3, 9)]  # first exchange sent at turn t
    assert starts == [1, 1, 1, 4, 4, 4]


def test_defaults_extract_after_every_exchange():
    assert Config().extract_every == 1 and Config().window_epoch == 1


def _engine(tmp_path, **kw):
    from types import SimpleNamespace

    from cama.metrics import RunRecorder
    from cama.engine import Engine
    from cama.types import Block, Kind
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a", **kw)
    eng = Engine(cfg, RunRecorder(tmp_path / "r", "x"))
    eng.active.admit(Block("b", Kind.PROJET, "B", "- one"))
    return eng, SimpleNamespace(paged_in=[]), SimpleNamespace(paged_in=["c"])


def test_with_epoch_refresh_memory_changes_wait_for_the_window_drop(tmp_path):
    eng, quiet, _ = _engine(tmp_path, window_epoch=3, memory_refresh="epoch")
    eng.turn_index = 3
    assert eng._memory_view(quiet)[0].content == "- one"
    eng.active.get("b").content = "- one\n- two"  # an extraction between drops
    eng.turn_index = 4
    assert eng._memory_view(quiet)[0].content == "- one"  # the prompt stays append-only
    eng.turn_index = 6
    assert eng._memory_view(quiet)[0].content == "- one\n- two"  # picked up at the drop


def test_a_page_in_refreshes_the_frozen_memory_at_once(tmp_path):
    eng, quiet, paged = _engine(tmp_path, window_epoch=3, memory_refresh="epoch")
    eng.turn_index = 3
    eng._memory_view(quiet)
    eng.active.get("b").content = "- changed"
    eng.turn_index = 4
    assert eng._memory_view(paged)[0].content == "- changed"


def test_by_default_memory_is_rendered_live_every_turn(tmp_path):
    eng, quiet, _ = _engine(tmp_path, window_epoch=3)
    eng._memory_view(quiet)
    eng.active.get("b").content = "- live"
    assert eng._memory_view(quiet)[0].content == "- live"


def test_with_tail_page_ins_the_cached_head_stays_and_the_block_goes_last(tmp_path):
    from cama.types import Block, Kind
    eng, quiet, paged = _engine(tmp_path, window_epoch=3, memory_refresh="epoch", page_in_tail=True)
    eng.turn_index = 3
    head = eng._memory_view(quiet)
    eng.active.admit(Block("c", Kind.PROJET, "C", "- paged"))
    eng.turn_index = 4
    assert [b.id for b in eng._memory_view(paged)] == [b.id for b in head]  # head unchanged
    assert [b.id for b in eng._tail] == ["c"]
    eng._tail = []
    eng.turn_index = 6
    assert "c" in [b.id for b in eng._memory_view(quiet)]  # joins the head at the drop


def test_a_statement_is_told_apart_from_a_question_or_a_request():
    from cama.engine import states_a_fact
    assert states_a_fact("Mon vol retour est prévu le 18 décembre.")
    assert states_a_fact("Changement de programme : mon talk passe à 30 minutes.")
    assert not states_a_fact("Quand est mon vol retour ?")
    assert not states_a_fact("Calcule 12 * 75 pour l'hôtel.")
    assert not states_a_fact("What is the next step for the trip?")
    assert not states_a_fact("")


def test_recency_order_moves_a_block_when_it_is_read(tmp_path):
    import time as _t
    from cama.types import Block, Kind
    eng, quiet, _ = _engine(tmp_path, render_order="recency")
    eng.active.admit(Block("a", Kind.FONDAMENTAL, "A", "- a"))
    eng.active.get("a").accessed_at = _t.time() - 100
    eng.active.get("b").accessed_at = _t.time()
    assert [b.id for b in eng._memory_view(quiet)] == ["a", "b"]
    eng.active.get("a").accessed_at = _t.time() + 1  # a read
    assert [b.id for b in eng._memory_view(quiet)] == ["b", "a"]
    stable = _engine(tmp_path / "s")[0]
    stable.active.admit(Block("a", Kind.FONDAMENTAL, "A", "- a"))
    assert [b.id for b in stable._memory_view(quiet)][0] == "a"  # stable: kind first, whatever was read
