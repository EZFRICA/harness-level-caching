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


async def test_statements_only_stores_what_states_a_fact_and_still_finds_it(tmp_path):
    eng = _engine(tmp_path, verbatim_recall=True, verbatim_store="statements")
    await eng.start()
    try:
        await _say(eng, ["My gym locker code is 4831.", "What is the weather like on Mars?"] + PAD
                   + ["What is my gym locker code?"])
        stored = [r["content"] for r in eng.store.db.execute("SELECT content FROM blocks").fetchall()]
        assert "My gym locker code is 4831." in stored
        assert not any("?" in c for c in stored)  # questions are not stored
        assert "4831" in eng.store.history(eng.session)[-2].text
    finally:
        await eng.close()


async def test_quotes_sent_once_reach_the_model_but_not_the_history(tmp_path):
    eng = _engine(tmp_path, verbatim_recall=True, verbatim_history="once")
    await eng.start()
    sent = []
    run = eng.loop.run

    async def spy(system, messages, *a, **kw):
        sent.append(messages[-1])
        return await run(system, messages, *a, **kw)

    eng.loop.run = spy
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD + ["What is my gym locker code?"])
        assert "4831" in str(sent[-1])  # the model saw the quote this turn
        assert eng.store.history(eng.session)[-2].text == "What is my gym locker code?"
    finally:
        await eng.close()


def test_an_unknown_h9_switch_is_refused(tmp_path):
    import pytest
    for kw in ({"verbatim_store": "some"}, {"verbatim_history": "never"}):
        with pytest.raises(ValueError):
            _engine(tmp_path, verbatim_recall=True, **kw)


def test_retrieval_counts_the_latest_statement_of_each_probed_value():
    from harness_cache.report import verbatim_retrieval
    messages = ["My code is 4831.", "hello", "Correction: my code is 7290.", "What is my code?", "What is my code?"]
    manifest = {"probes": {"items": [{"turn": 4, "expect": ["7290"]}, {"turn": 5, "expect": ["7290"]}]}}
    turns = [{"turn": 1, "route": None}, {"turn": 2, "route": {"hits": []}}, {"turn": 3, "route": {"hits": []}},
             {"turn": 4, "route": {"hits": [["msg-000002", 0.9]]}},
             {"turn": 5, "route": {"hits": [["msg-000000", 0.9], ["msg-000001", 0.5]]}}]  # only the stale value
    assert verbatim_retrieval(manifest, turns, messages) == {
        "found": 1, "needed": 2, "turns_with_recall": 2, "lines": 3, "turns": 5}
