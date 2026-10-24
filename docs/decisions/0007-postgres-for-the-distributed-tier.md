---
id: 0007
status: accepted
evidence:
held_by:
---
# 0007: Postgres, in a container, is the shared tier of the distributed system

Decided: 2026-10-23, by the owner.

## Context
Scale and fault tolerance are required now. Workers must be replaceable, an agent must have
one writer, and a crash must lose a bounded, measured amount of memory.

## Decision
One Postgres container holds: agent leases with an epoch (fencing of stale writers), the
journal replicated idempotently by (agent, seq), snapshots for recovery, and the shared
registries of H2, H3 and H4.

## Consequences
- RPO is the unshipped journal at crash time; RTO is snapshot load plus replay. Both are to be
  measured by the stress profiles S5 and S6.
- Not built yet: this record holds no test and no evidence until it is.

## Alternatives set aside
- Redis with an object store: two services to operate for the same guarantees.
- A shared SQLite file: does not exercise real network faults.
