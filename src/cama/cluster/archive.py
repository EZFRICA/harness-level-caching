"""L4 through the coordinator: journal rows shipped with the writer's lease, fenced by epoch."""

from __future__ import annotations


class CoordinatorArchive:
    name = "coordinator"

    def __init__(self, coord, lease):
        self.coord = coord
        self.lease = lease  # callable: the current Lease of this agent on this worker

    async def push(self, entries: list[dict]) -> None:
        lease = self.lease()
        if lease is None:
            raise RuntimeError("no lease: this worker does not own the agent any more")
        await self.coord.append(lease, entries)
