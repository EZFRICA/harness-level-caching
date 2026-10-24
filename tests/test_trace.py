"""The trace holds: the ledger chain, generated pages, generated figures, evidence-linked decisions,
and the owner's writing rules on every Markdown file of the repository."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cama.trace import decisions, figures
from cama.trace.ledger import Ledger, LedgerError, render, write_pages

ROOT = Path(__file__).resolve().parent.parent
DASH = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")


def _event(**kw):
    base = {"who": "test", "kind": "action", "title": "t", "why": "w", "result": "done"}
    return {**base, **kw}


def test_an_edited_ledger_event_breaks_the_chain(tmp_path):
    """The ledger is append-only by construction: rewriting history must be detectable."""
    led = Ledger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.append(_event(title=f"step {i}"))
    assert led.check() == []
    lines = led.path.read_text().splitlines()
    lines[1] = lines[1].replace("step 1", "step one")
    led.path.write_text("\n".join(lines) + "\n")
    assert any("chain broken" in p for p in led.check())


def test_a_deleted_ledger_event_breaks_the_chain(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.append(_event(title=f"step {i}"))
    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert led.check()


@pytest.mark.parametrize("bad", [
    {"title": "a " + chr(0x2014) + " b"},
    {"why": "key AIzaSyA1234567890abcdefghijklmnop"},
    {"kind": "chat"},
    {"result": "maybe"},
    {"why": ""},
])
def test_the_ledger_refuses_dashes_secrets_and_unknown_values(tmp_path, bad):
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "l.jsonl").append(_event(**bad))


def test_journal_pages_are_rendered_under_the_line_limit(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    for i in range(40):
        led.append(_event(title=f"step {i}", files=["a.py", "b.py"], verified="ok", live="run x"), now=now)
    pages = render(led.events())
    assert len(pages) > 1 and all(t.count("\n") <= 100 for t in pages.values())


def test_the_repository_ledger_chain_holds_and_its_pages_are_current():
    led = Ledger(ROOT / "docs" / "ledger.jsonl")
    assert led.check() == []
    for name, text in render(led.events()).items():
        assert (ROOT / "docs" / "journal" / name).read_text() == text, f"journal {name} is stale"


def test_published_figures_equal_their_regeneration():
    """A number in docs/measured is generated from a frozen run, never typed."""
    measured = ROOT / "docs" / "measured"
    for name, text in figures.generate(measured).items():
        assert (measured / name).read_text() == text, f"{name} differs from its regeneration"


def test_every_decision_cites_frozen_runs_and_existing_tests():
    assert decisions.check(ROOT) == []


def _markdown_files():
    skip = {".venv", "runs", "node_modules", ".pytest_cache", "mutants"}
    return [p for p in ROOT.rglob("*.md") if not skip & set(p.relative_to(ROOT).parts)]


def test_no_markdown_file_is_longer_than_100_lines():
    long = [f"{p.relative_to(ROOT)}: {p.read_text().count(chr(10))}" for p in _markdown_files()
            if p.read_text().count("\n") > 100]
    assert not long, long


def test_no_em_or_en_dash_in_repository_prose():
    found = [str(p.relative_to(ROOT)) for p in _markdown_files() if DASH.search(p.read_text())]
    found += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*") if p.suffix in (".py", ".html", ".js", ".css")
              and DASH.search(p.read_text())]
    assert not found, found


def test_a_journal_entry_renders_every_recorded_field_in_order(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    e = led.append(_event(title="Add H3", why="tool calls repeat", files=["a.py"], verified="39 passed",
                          live="run r1, 12 hits", runs=["r1"], decisions=["0006"], data="store run-r1",
                          result="partial", detail="TTL classes left"),
                   now=datetime(2026, 10, 23, 9, 5, tzinfo=timezone.utc))
    page = render([e])["2026-10-23.md"]
    assert page.splitlines()[0] == "# Journal 2026-10-23"
    body = page.split("Do not edit.\n\n", 1)[1]
    assert body == (
        "## 09:05 #1 action, test: Add H3\n\n"
        "- **Why:** tool calls repeat\n- **Files:** `a.py`\n- **Verified:** 39 passed\n"
        "- **Live:** run r1, 12 hits\n- **Runs:** `r1`\n- **Decisions:** 0006\n"
        "- **Data:** store run-r1\n- **Result:** partial, TTL classes left\n"
    )


def test_the_first_event_starts_the_chain_and_numbers_increase(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    a, b = led.append(_event()), led.append(_event())
    assert a["prev"] == "0" * 16 and a["seq"] == 1 and b["seq"] == 2 and b["prev"] != a["prev"]


@pytest.mark.parametrize("field", ["who", "kind", "title", "why", "result"])
def test_each_required_field_is_enforced(tmp_path, field):
    with pytest.raises(LedgerError):
        Ledger(tmp_path / "l.jsonl").append({k: v for k, v in _event().items() if k != field})


def test_reordered_events_are_reported(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.append(_event(title=f"s{i}"))
    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    problems = led.check()
    assert any("out of order" in p for p in problems) and any("chain broken" in p for p in problems)


def test_rendering_removes_pages_of_days_no_longer_in_the_ledger(tmp_path):
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "2026-01-01.md").write_text("stale")
    (journal / "README.md").write_text("kept")
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append(_event(), now=datetime(2026, 10, 23, tzinfo=timezone.utc))
    write_pages(led, journal)
    assert sorted(p.name for p in journal.iterdir()) == ["2026-10-23.md", "README.md"]


def test_events_are_stamped_in_utc_when_no_time_is_given(tmp_path):
    e = Ledger(tmp_path / "l.jsonl").append(_event())
    stamped = datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 5


def test_an_empty_ledger_file_starts_a_new_chain(tmp_path):
    p = tmp_path / "deep" / "dir" / "l.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("")
    e = Ledger(p).append(_event())
    assert e["seq"] == 1 and e["prev"] == "0" * 16
    assert Ledger(tmp_path / "new" / "sub" / "l.jsonl").append(_event())["seq"] == 1
