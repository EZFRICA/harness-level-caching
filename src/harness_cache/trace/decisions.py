"""Decisions carry their evidence.

Each `docs/decisions/NNNN-*.md` starts with a front matter block:

    ---
    id: 0005
    status: accepted | replaced by NNNN
    evidence: run-id-a, run-id-b      (frozen runs in docs/measured/runs/)
    held_by: tests/test_x.py::test_name, ...
    ---

`check()` verifies that every cited run is frozen and every cited test exists, so a decision
cannot outlive the proof it rests on without someone noticing.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def parse(path: Path) -> dict:
    m = _FRONT.match(path.read_text())
    if not m:
        raise ValueError(f"{path.name}: missing front matter")
    meta: dict = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        meta[key.strip()] = [v.strip() for v in value.split(",") if v.strip()] if key.strip() in (
            "evidence", "held_by") else value
    return meta


def check(root: Path) -> list[str]:
    problems = []
    runs = {p.stem for p in (root / "docs/measured/runs").glob("*.json")}
    for path in sorted((root / "docs/decisions").glob("[0-9][0-9][0-9][0-9]-*.md")):
        try:
            meta = parse(path)
        except ValueError as e:
            problems.append(str(e))
            continue
        if meta.get("id") != path.name[:4]:
            problems.append(f"{path.name}: id {meta.get('id')} does not match the file name")
        for run in meta.get("evidence", []):
            if run not in runs:
                problems.append(f"{path.name}: evidence run {run} is not frozen in docs/measured/runs")
        for ref in meta.get("held_by", []):
            file, _, name = ref.partition("::")
            f = root / file
            if not f.exists() or (name and f"def {name}(" not in f.read_text()):
                problems.append(f"{path.name}: held_by {ref} not found")
    return problems
