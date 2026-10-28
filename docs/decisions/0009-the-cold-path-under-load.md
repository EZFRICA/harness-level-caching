---
id: 0009
status: accepted
evidence: cluster-s5-kill-w2-v2, cluster-s5-kill-w2-v3, cluster-s5-pg-down-v1
held_by: tests/test_cluster.py::test_replication_keeps_up_under_back_to_back_turns, tests/test_cluster.py::test_extraction_coalesces_when_the_cold_path_falls_behind
---
# 0009: Replication runs on a clock, and extraction coalesces when the cold path falls behind

Decided: 2026-10-23, proposed by Claude after the first kill test, within the owner's plan.

## Context
The cold path ran only while no turn was in progress. Under back to back turns that never
happened: the first `kill -9` of a worker lost 26 journal rows per agent, and the cheaper
latency of that run came from a cold path that did no work (ledger, fault events).

## Decision
- Replication ships the journal every `HC_SYNC_INTERVAL_S` (1 s), whatever the hot path
  does; `drain` and a clean shutdown flush it. The RPO is bounded by that interval.
- One queued extraction drains every pending exchange, up to `HC_EXTRACT_BATCH_MAX` (8), in
  one model call.

## Consequences
- Measured after the change: RPO 0 rows at a kill, journal continuous across epochs.
- Throughput fell from 4.2 to 2.9 requests per second on three workers: the cold path now does
  its work. The earlier figure was not a capacity, it was starvation.
- Batched extraction reads several exchanges at once; its effect on extraction quality is not
  measured.

## Alternatives set aside
- A higher cold priority without a clock: still starved by a continuous stream of turns.
- Replicating inside the turn: the user would pay for it.
