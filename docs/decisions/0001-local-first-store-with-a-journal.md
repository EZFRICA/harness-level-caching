---
id: 0001
status: accepted
evidence: echo-cama-v1
held_by: tests/test_core.py::test_end_to_end_scenario_echo, tests/test_core.py::test_store_rejects_non_unit_vectors_and_foreign_spaces
---
# 0001: Memory is local first: SQLite on the device, a journal feeds every replica

Decided: 2026-10-23, proposed by Claude, accepted by the owner.

## Context
The APU kept memory in Weaviate and Letta Cloud. Both are network services: the agent could
not run offline, and every turn paid network round trips. The owner's target includes a phone
(Pixel 4 XL) with intermittent connectivity.

## Decision
- The source of truth is a SQLite file on the device: blocks, vectors, conversation history.
- Every block mutation writes a journal row in the same transaction.
- Remote stores (L4) are replicas fed from the journal by the cold path, never read on the
  hot path.

## Consequences
- Online and offline share one memory; only the model changes.
- A replica can lag; its lag is the journal's unsynced count, shown in the dashboard.
- A store is bound to one embedding space and refuses vectors that are not unit norm.

## Alternatives set aside
- Keeping Weaviate as L3: a server on the hot path, unavailable offline.
- A remote database as source of truth: every turn would depend on the network.
