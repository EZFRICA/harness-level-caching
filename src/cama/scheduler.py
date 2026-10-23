"""Cold-path scheduler: a priority queue whose worker only runs while the hot path is idle.

Priorities (lower runs first): 2 = memory extraction, 3 = L4 sync.
Priorities 0 and 1 are the hot path itself and never go through this queue.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from cama.logging import get_logger

log = get_logger(__name__)

TaskFn = Callable[[], Awaitable[dict | None]]


@dataclass(order=True)
class _Task:
    priority: int
    seq: int
    name: str = field(compare=False)
    fn: TaskFn = field(compare=False)
    enqueued_at: float = field(compare=False)
    attempt: int = field(default=1, compare=False)


class Scheduler:
    def __init__(self, on_done: Callable[[dict], None], max_attempts: int = 3):
        self.queue: asyncio.PriorityQueue[_Task] = asyncio.PriorityQueue()
        self.idle = asyncio.Event()
        self.idle.set()
        self._seq = itertools.count()
        self._on_done = on_done
        self._max_attempts = max_attempts
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="cama-scheduler")

    async def push(self, name: str, fn: TaskFn, priority: int) -> None:
        await self.queue.put(_Task(priority, next(self._seq), name, fn, time.perf_counter()))

    def backlog(self) -> int:
        return self.queue.qsize()

    async def drain(self) -> None:
        await self.queue.join()

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            task = await self.queue.get()
            try:
                await self.idle.wait()  # never compete with a turn in progress
                started = time.perf_counter()
                record = {"task": task.name, "priority": task.priority, "attempt": task.attempt,
                          "queued_ms": (started - task.enqueued_at) * 1000}
                try:
                    detail = await task.fn()
                    record.update(ok=True, run_ms=(time.perf_counter() - started) * 1000, **(detail or {}))
                except Exception as e:  # noqa: BLE001 — cold path must survive any task failure
                    record.update(ok=False, run_ms=(time.perf_counter() - started) * 1000, error=repr(e)[:300])
                    log.warning("cold task %s failed (attempt %d): %r", task.name, task.attempt, e)
                    if task.attempt < self._max_attempts:
                        await asyncio.sleep(0.5 * 2 ** (task.attempt - 1))
                        await self.queue.put(
                            _Task(task.priority, next(self._seq), task.name, task.fn, time.perf_counter(),
                                  task.attempt + 1)
                        )
                self._on_done(record)
            finally:
                self.queue.task_done()
