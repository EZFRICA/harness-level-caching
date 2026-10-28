"""Trace: the ledger of actions, generated figures, and evidence-linked decisions."""

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """The directory holding pyproject.toml, searched upward from `start` (default: cwd)."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists() and (d / "src" / "harness_cache").exists():
            return d
    raise FileNotFoundError("not inside the CAMA repository")
