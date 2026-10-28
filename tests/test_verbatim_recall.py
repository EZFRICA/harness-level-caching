"""Hypothesis H9: compact plus apu-urithi's search over the user's own messages, recalled verbatim at the tail."""

from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.metrics import RunRecorder


def _engine(tmp_path, **kw):
    kw = {"compact_at": 60, "compact_keep": 1, **kw}
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 mode="compact", tools=False, **kw)
    return Engine(cfg, RunRecorder(tmp_path / "r", "x"))


async def _say(eng, texts):
    out = None
    for t in texts:
        out = await eng.turn(t)
        await eng.drain()
    return out


PAD = ["Tell me something about clouds and rain please. " + "pad " * 30] * 4


async def test_a_folded_statement_and_its_correction_come_back_verbatim_oldest_first(tmp_path):
    eng = _engine(tmp_path, verbatim_recall=True)
    await eng.start()
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD
                   + ["Correction: my gym locker code is 7290."] + PAD)
        assert eng._compact_from > 0
        await _say(eng, ["What is my gym locker code?"])
        sent = eng.store.history(eng.session)[-2].text
        assert sent.endswith("What is my gym locker code?")
        assert "4831" in sent and "7290" in sent
        assert sent.index("4831") < sent.index("7290")  # the latest value is read last
        assert not eng.active.ids()  # nothing paged into a rendered memory
        assert eng.store.count() == {**eng.store.count(), "blocks": 11, "active": 0}  # one per turn, L3 only
    finally:
        await eng.close()


async def test_a_message_still_in_the_unfolded_history_is_not_sent_twice(tmp_path):
    eng = _engine(tmp_path, verbatim_recall=True, compact_at=100_000)
    await eng.start()
    try:
        await _say(eng, ["My gym locker code is 4831.", "What is my gym locker code?"])
        assert "[memory paged in" not in eng.store.history(eng.session)[-2].text
    finally:
        await eng.close()


async def test_compact_without_the_switch_stores_no_message(tmp_path):
    eng = _engine(tmp_path)
    await eng.start()
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD + ["What is my gym locker code?"])
        assert eng.store.count()["blocks"] == 0
        assert "[memory paged in" not in eng.store.history(eng.session)[-2].text
    finally:
        await eng.close()
