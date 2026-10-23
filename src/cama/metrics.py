"""Run recording. One directory per run:

    manifest.json  configuration, resource limits, versions — written before the first turn
    turns.jsonl    one record per hot-path turn
    tasks.jsonl    one record per cold-path task
    cama.log       human-readable log
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cama import __version__


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def resource_limits() -> dict:
    """cgroup v2 limits as seen from inside the container."""
    cpu_max = _read("/sys/fs/cgroup/cpu.max")  # "<quota> <period>" or "max <period>"
    cpus = None
    if cpu_max and not cpu_max.startswith("max"):
        quota, period = cpu_max.split()
        cpus = int(quota) / int(period)
    mem_max = _read("/sys/fs/cgroup/memory.max")
    return {
        "cpu_max_raw": cpu_max,
        "cpus_limit": cpus,
        "memory_max_bytes": int(mem_max) if mem_max and mem_max.isdigit() else mem_max,
        "host_cpu_count": os.cpu_count(),
    }


def rss_mb() -> float | None:
    statm = _read("/proc/self/statm")
    if statm:
        return int(statm.split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    return None


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024


class RunRecorder:
    def __init__(self, runs_dir: Path, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._turns = (self.dir / "turns.jsonl").open("a", encoding="utf-8")
        self._tasks = (self.dir / "tasks.jsonl").open("a", encoding="utf-8")

    def manifest(self, config: dict, extra: dict | None = None) -> None:
        data = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cama_version": __version__,
            "code_sha": os.environ.get("CAMA_CODE_SHA", "unknown"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "limits": resource_limits(),
            "config": config,
            **(extra or {}),
        }
        (self.dir / "manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def turn(self, record: dict) -> None:
        self._write(self._turns, record)

    def task(self, record: dict) -> None:
        self._write(self._tasks, record)

    @staticmethod
    def _write(f, record: dict) -> None:
        f.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False) + "\n")
        f.flush()

    def close(self) -> None:
        self._turns.close()
        self._tasks.close()
