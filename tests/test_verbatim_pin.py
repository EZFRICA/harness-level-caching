"""Hypothesis H10: compact keeps the statements it folds word for word in a block beside the summary."""

import pytest

from harness_cache.config import Config
from harness_cache.engine import Engine, pin_statements
from harness_cache.metrics import RunRecorder
from harness_cache.types import Message


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


async def test_a_folded_statement_and_its_correction_stay_in_the_prompt_word_for_word(tmp_path):
    eng = _engine(tmp_path, verbatim_pin=True)
    await eng.start()
    systems = []
    run = eng.loop.run

    async def spy(system, messages, *a, **kw):
        systems.append(system)
        return await run(system, messages, *a, **kw)

    eng.loop.run = spy
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD
                   + ["Correction: my gym locker code is 7290."] + PAD)
        out = await _say(eng, ["What is my gym locker code?"])
        head = systems[-1]
        assert "- turn 1: My gym locker code is 4831." in head
        assert "- turn 6: Correction: my gym locker code is 7290." in head
        assert head.index("4831") < head.index("7290")  # the latest value is read last
        assert "Tell me something" not in head.split("What the user stated")[1]  # questions are not pinned
        assert out.record["context"]["pinned"] == 2
        assert out.record["context"]["folded_turns"] >= 6
        assert eng.store.count()["blocks"] == 0  # nothing stored, nothing searched
        assert out.record["route"] is None
    finally:
        await eng.close()


async def test_the_head_changes_only_when_a_fold_happens(tmp_path):
    eng = _engine(tmp_path, verbatim_pin=True)
    await eng.start()
    systems = []
    run = eng.loop.run

    async def spy(system, messages, *a, **kw):
        systems.append((eng._compact_from, system))
        return await run(system, messages, *a, **kw)

    eng.loop.run = spy
    try:
        await _say(eng, ["My cat is named Biscotte."] + PAD * 3)
        by_fold = {}
        for folded, system in systems:
            by_fold.setdefault(folded, set()).add(system)
        assert len(by_fold) > 1 and all(len(s) == 1 for s in by_fold.values())
    finally:
        await eng.close()


def test_a_statement_said_again_moves_to_its_latest_turn():
    rows = [Message("user", "My code is 4831."), Message("assistant", "ok"),
            Message("user", "My code is 7290."), Message("assistant", "ok"),
            Message("user", "What is my code?"), Message("assistant", "7290"),
            Message("user", "My code is 4831."), Message("assistant", "ok")]
    pinned = pin_statements({"My cat is Biscotte.": 1}, rows, first_turn=10)
    assert list(pinned.items()) == [("My cat is Biscotte.", 1), ("My code is 7290.", 11), ("My code is 4831.", 13)]


async def test_without_the_switch_nothing_is_pinned(tmp_path):
    eng = _engine(tmp_path)
    await eng.start()
    try:
        out = await _say(eng, ["My gym locker code is 4831."] + PAD + ["What is my gym locker code?"])
        assert eng._compact_from > 0 and out.record["context"]["pinned"] == 0
    finally:
        await eng.close()


def test_pinning_needs_compact_and_replaces_the_search(tmp_path):
    with pytest.raises(ValueError):
        _engine(tmp_path, verbatim_pin=True, verbatim_recall=True)
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 mode="naive", tools=False, verbatim_pin=True)
    with pytest.raises(ValueError):
        Engine(cfg, RunRecorder(tmp_path / "r", "x"))


def test_presence_counts_a_statement_folded_but_pinned_or_still_unfolded():
    from harness_cache.report import statement_in_prompt
    messages = ["My code is 4831.", "hello there", "Correction: my code is 7290.", "What is my code?",
                "What is my code?", "What is my code?"]
    manifest = {"config": {"verbatim_pin": True},
                "probes": {"items": [{"turn": 4, "expect": ["7290"]}, {"turn": 5, "expect": ["7290"]},
                                     {"turn": 6, "expect": ["7290"]}]}}
    turns = [{"turn": 4, "context": {"folded_turns": 2, "pinned": 1}},   # statement unfolded
             {"turn": 5, "context": {"folded_turns": 4, "pinned": 2}},   # folded, pinned
             {"turn": 6, "context": {"folded_turns": 4, "pinned": 2}}]
    assert statement_in_prompt(manifest, turns, messages) == {
        "found": 3, "needed": 3, "unfolded": 1, "pinned_last": 2}
    manifest["config"]["verbatim_pin"] = False  # a plain summary: only the unfolded one is word for word
    assert statement_in_prompt(manifest, turns, messages)["found"] == 1
    asked = {**manifest, "probes": {"items": [{"turn": 5, "expect": ["code"]}]}}  # last said in a question
    asked["config"] = {"verbatim_pin": True}
    assert statement_in_prompt(asked, turns, messages)["found"] == 0  # folded questions are not pinned
    searched = {"config": {"verbatim_recall": True}, "probes": manifest["probes"]}
    turns[1]["route"] = {"hits": [["msg-000002", 0.9]]}  # H9 recalled the statement at turn 5 only
    assert statement_in_prompt(searched, turns, messages)["found"] == 2  # unfolded at 4, recalled at 5


def test_the_pinning_key_of_a_compact_run_names_its_memory_switches():
    from harness_cache.report import pin
    cfg = {"mode": "compact", "llm": "echo", "embedder": "hash", "capacity": 8, "max_memory_tokens": 1,
           "victim_window": 4, "compact_at": 6000, "verbatim_recall": False, "verbatim_pin": True}
    m = {"run_id": "r", "started_at": "2026-11-11T00:00:00", "config": cfg}
    assert pin(m)["verbatim_pin"] is True and pin(m)["compact_at"] == 6000
    assert "verbatim_pin" not in pin({**m, "config": {**cfg, "mode": "naive"}})


async def test_with_rest_the_summary_never_reads_a_pinned_statement(tmp_path):
    from harness_cache.engine import COMPACT_REST
    eng = _engine(tmp_path, verbatim_pin=True, pin_summary="rest")
    await eng.start()
    calls = []
    gen = eng.llm.generate

    async def spy(system, messages, **kw):
        calls.append((system, messages[-1].text))
        return await gen(system, messages, **kw)

    eng.llm.generate = spy
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD + ["What is my gym locker code?"])
        folds = [(s, p) for s, p in calls if p.startswith("PREVIOUS SUMMARY")]
        assert folds and all(s.endswith(COMPACT_REST) for s, _ in folds)
        assert not any("USER: My gym locker code is 4831." in p for _, p in folds)  # pinned, not folded
        assert any("USER: Tell me something" in p for _, p in folds)  # the rest still is
        assert any("ASSISTANT: [echo] My gym locker code" in p for _, p in folds)  # replies are kept
        assert "My gym locker code is 4831." in eng._pinned
    finally:
        await eng.close()


async def test_by_default_the_summary_still_reads_everything(tmp_path):
    eng = _engine(tmp_path, verbatim_pin=True)
    await eng.start()
    calls = []
    gen = eng.llm.generate

    async def spy(system, messages, **kw):
        calls.append((system, messages[-1].text))
        return await gen(system, messages, **kw)

    eng.llm.generate = spy
    try:
        await _say(eng, ["My gym locker code is 4831."] + PAD)
        folds = [(s, p) for s, p in calls if p.startswith("PREVIOUS SUMMARY")]
        assert folds and any("USER: My gym locker code is 4831." in p for _, p in folds)
        assert all("kept word for word" not in s for s, _ in folds)
    finally:
        await eng.close()


def test_rest_is_refused_without_pinning(tmp_path):
    for kw in ({"pin_summary": "rest"}, {"verbatim_pin": True, "pin_summary": "none"}):
        with pytest.raises(ValueError):
            _engine(tmp_path, **kw)
