"""HC_PIN_JUDGE: the Decisions API on top of the rule, with frozen probes and a fallback. No network:
the API is replaced by a function returning (probability, tokens) per text."""

import pytest

from harness_cache.config import Config
from harness_cache.cost import run_cost
from harness_cache.engine import Engine, states_a_fact
from harness_cache.memory.pin_judge import DecisionsJudge, load_probes
from harness_cache.metrics import RunRecorder

PROBES = [("probe fact", 0.9), ("probe noise", 0.05)]


def fake(answers: dict[str, float], asked: list, fail: bool = False, probe_shift: float = 0.0):
    async def ask(texts):
        asked.append(list(texts))
        if fail:
            raise ConnectionError("down")
        out = {}
        for t in texts:
            p = dict(PROBES).get(t)
            out[t] = ((p + probe_shift) if p is not None else answers.get(t, 0.0), 10)
        return out
    return ask


async def test_keeps_what_the_api_accepts_and_never_sends_a_question():
    asked = []
    j = DecisionsJudge(fake({"My code is 4831.": 0.9, "Note: it moved by one week.": 0.2}, asked),
                       states_a_fact, PROBES)
    v = await j.judge(["My code is 4831.", "Note: it moved by one week.", "What is my code?"])
    assert v.keep == {"My code is 4831."}
    assert v.fallback is None and v.calls == 4 and v.tokens == 40  # 2 probes + 2 candidates
    assert all("What is my code?" not in batch for batch in asked)


async def test_a_probe_that_moved_sends_the_run_back_to_the_rule_for_good():
    asked = []
    j = DecisionsJudge(fake({"Note: it moved by one week.": 0.2}, asked, probe_shift=0.11), states_a_fact, PROBES)
    v = await j.judge(["Note: it moved by one week.", "What now?"])
    assert v.keep == {"Note: it moved by one week."}  # the rule decides
    assert v.fallback.startswith("probe drift")
    before = len(asked)
    v2 = await j.judge(["My code is 4831."])
    assert v2.keep == {"My code is 4831."} and v2.fallback == v.fallback and len(asked) == before  # no more calls


async def test_a_probe_within_tolerance_is_not_drift():
    j = DecisionsJudge(fake({"My code is 4831.": 0.9}, [], probe_shift=0.09), states_a_fact, PROBES)
    assert (await j.judge(["My code is 4831."])).fallback is None


async def test_a_failed_call_falls_back_for_that_fold_only():
    asked = []
    j = DecisionsJudge(fake({}, asked, fail=True), states_a_fact, PROBES)
    v = await j.judge(["My code is 4831."])
    assert v.keep == {"My code is 4831."} and v.fallback == "api error: ConnectionError"
    j.ask = fake({"My code is 4831.": 0.9}, asked)
    assert (await j.judge(["My code is 4831."])).fallback is None


def test_the_frozen_probes_are_shipped():
    probes = load_probes()
    assert len(probes) == 8 and all(0 <= p <= 1 for _, p in probes)


PAD = ["Tell me something about clouds and rain please. " + "pad " * 30] * 4


async def test_the_fold_pins_only_what_the_judge_kept(tmp_path):
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a", mode="compact",
                 tools=False, compact_at=60, compact_keep=1, verbatim_pin=True)
    eng = Engine(cfg, RunRecorder(tmp_path / "r", "x"))
    eng.pin_judge = DecisionsJudge(fake({"My gym locker code is 4831.": 0.9, "Note for the trip: it moved.": 0.2}, []),
                                   states_a_fact, PROBES)
    await eng.start()
    try:
        for t in ["My gym locker code is 4831.", "Note for the trip: it moved."] + PAD:
            await eng.turn(t)
            await eng.drain()
        assert "My gym locker code is 4831." in eng._pinned
        assert "Note for the trip: it moved." not in eng._pinned
        tasks = [line for line in (tmp_path / "r").rglob("tasks.jsonl")][0].read_text()
        assert '"judge_calls"' in tasks and '"judge_fallback": null' in tasks
    finally:
        await eng.close()


def test_decisions_needs_pinning_and_an_unknown_judge_is_refused(tmp_path):
    for kw in ({"pin_judge": "decisions"}, {"verbatim_pin": True, "pin_judge": "llm"}):
        cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a", mode="compact",
                     tools=False, **kw)
        with pytest.raises(ValueError):
            Engine(cfg, RunRecorder(tmp_path / "r", "x"))


def test_the_judge_tokens_are_priced_on_the_cold_path():
    pricing = {"echo": {"status": "local", "read_on": "x", "source": "x", "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0,
                        "cache_storage_hour": 0.0},
               "gemini-decision-1": {"status": "assumption", "input": 0.10, "output": 0.0, "cache_read": 0.10,
                                     "cache_write": 0.0, "cache_storage_hour": 0.0}}
    manifest = {"config": {"llm": "echo", "model": "echo"}}
    tasks = [{"task": "compact#9", "ok": True, "ts": 1.0, "judge_tokens": 2_000_000}]
    cost = run_cost(manifest, [], tasks, pricing)
    assert cost["cold"]["pin_judge"] == pytest.approx(0.2)
