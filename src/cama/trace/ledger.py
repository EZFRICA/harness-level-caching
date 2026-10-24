"""The ledger: CAMA's trace of every action, as typed events in one append-only file.

`docs/ledger.jsonl` holds one event per line. Each event carries the hash of the line
before it, so an edited or deleted event breaks the chain and `cama ledger check` says
where. The daily journal pages under `docs/journal/` are rendered from the ledger and are
never edited by hand: the ledger is the only source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("action", "run", "decision", "measurement", "fault", "review")
RESULTS = ("done", "partial", "failed")
GENESIS = "0" * 16
MAX_PAGE_LINES = 100

# Text the house rules forbid in the repository.
_DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
_SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z]{20,}|AQ\.[0-9A-Za-z_\-]{20,}")


class LedgerError(ValueError):
    pass


def _hash(line: str) -> str:
    return hashlib.blake2b(line.encode(), digest_size=8).hexdigest()


@dataclass
class Ledger:
    path: Path

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def _last_line(self) -> str | None:
        if not self.path.exists():
            return None
        lines = [x for x in self.path.read_text(encoding="utf-8").splitlines() if x]
        return lines[-1] if lines else None

    def append(self, event: dict, now: datetime | None = None) -> dict:
        validate(event)
        last = self._last_line()
        prev = _hash(last) if last else GENESIS
        seq = (json.loads(last)["seq"] + 1) if last else 1
        ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        full = {"seq": seq, "ts": event.get("ts", ts), "prev": prev, **{k: v for k, v in event.items() if k != "ts"}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full, ensure_ascii=False, sort_keys=True) + "\n")
        return full

    def check(self) -> list[str]:
        """Return every integrity problem; an empty list means the chain holds."""
        problems = []
        prev_line = None
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        for n, line in enumerate(x for x in lines if x):
            event = json.loads(line)
            want = _hash(prev_line) if prev_line else GENESIS
            if event.get("prev") != want:
                problems.append(f"seq {event.get('seq')}: chain broken (prev {event.get('prev')} != {want})")
            if event.get("seq") != n + 1:
                problems.append(f"line {n + 1}: seq {event.get('seq')} out of order")
            try:
                validate(event)
            except LedgerError as e:
                problems.append(f"seq {event.get('seq')}: {e}")
            prev_line = line
        return problems


def validate(event: dict) -> None:
    for key in ("who", "kind", "title", "why", "result"):
        if not event.get(key):
            raise LedgerError(f"missing {key!r}")
    if event["kind"] not in KINDS:
        raise LedgerError(f"kind {event['kind']!r} not in {KINDS}")
    if event["result"] not in RESULTS:
        raise LedgerError(f"result {event['result']!r} not in {RESULTS}")
    text = json.dumps(event, ensure_ascii=False)
    if _DASHES.search(text):
        raise LedgerError("em or en dash in prose (house rules)")
    if _SECRET.search(text):
        raise LedgerError("looks like a secret; never record keys or tokens")


# ── rendering ───────────────────────────────────────────────────────────────

def _entry(e: dict) -> list[str]:
    out = [f"## {e['ts'][11:16]} #{e['seq']} {e['kind']}, {e['who']}: {e['title']}", ""]
    out.append(f"- **Why:** {e['why']}")
    if e.get("files"):
        out.append(f"- **Files:** {', '.join(f'`{f}`' for f in e['files'])}")
    if e.get("verified"):
        out.append(f"- **Verified:** {e['verified']}")
    if e.get("live"):
        out.append(f"- **Live:** {e['live']}")
    if e.get("runs"):
        out.append(f"- **Runs:** {', '.join(f'`{r}`' for r in e['runs'])}")
    if e.get("decisions"):
        out.append(f"- **Decisions:** {', '.join(e['decisions'])}")
    if e.get("data"):
        out.append(f"- **Data:** {e['data']}")
    out.append(f"- **Result:** {e['result']}" + (f", {e['detail']}" if e.get("detail") else ""))
    out.append("")
    return out


def render(events: list[dict]) -> dict[str, str]:
    """Render the ledger as daily pages of at most MAX_PAGE_LINES lines each."""
    by_day: dict[str, list[dict]] = {}
    for e in events:
        by_day.setdefault(e["ts"][:10], []).append(e)
    pages: dict[str, str] = {}
    for day, evs in sorted(by_day.items()):
        part, lines = 1, None
        for e in evs:
            entry = _entry(e)
            if lines is None or len(lines) + len(entry) > MAX_PAGE_LINES:
                if lines is not None:
                    pages[_page_name(day, part)] = "\n".join(lines).rstrip() + "\n"
                    part += 1
                lines = _header(day, part)
            lines += entry
        pages[_page_name(day, part)] = "\n".join(lines).rstrip() + "\n"
    return pages


def _page_name(day: str, part: int) -> str:
    return f"{day}.md" if part == 1 else f"{day}-{part}.md"


def _header(day: str, part: int) -> list[str]:
    suffix = "" if part == 1 else f" (part {part})"
    return [f"# Journal {day}{suffix}", "",
            "Generated from `docs/ledger.jsonl` by `cama ledger render`. Do not edit.", ""]


def write_pages(ledger: Ledger, journal_dir: Path) -> list[Path]:
    journal_dir.mkdir(parents=True, exist_ok=True)
    pages = render(ledger.events())
    written = []
    for name, text in pages.items():
        p = journal_dir / name
        p.write_text(text, encoding="utf-8")
        written.append(p)
    for stale in journal_dir.glob("*.md"):
        if stale.name not in pages and stale.name != "README.md":
            stale.unlink()
    return written
