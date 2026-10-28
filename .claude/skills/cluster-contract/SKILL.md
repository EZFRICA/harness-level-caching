---
name: cluster-contract
description: The rules the distributed tier must keep (one writer per agent through a lease and an epoch, fencing of stale writers, idempotent journal, recovery where the shared tier is the authority, replication on a clock, degradation without blocking, shared cache entries) and the tests that hold each. Use when touching src/harness_cache/cluster, the coordinator schema, replication, recovery, or anything a second worker could observe.
when_to_use: Changing the coordinator, the worker service, the load generator, recovery or a shared cache; reviewing split brain, data loss or recovery time.
---

# The cluster contract

Both coordinators, `PostgresCoordinator` and `MemoryCoordinator`, obey every rule below; the
contract tests in `tests/test_cluster.py` run on both (Postgres with `HC_LIVE_PG=<dsn>` from
a container on the compose network).

| rule | held by |
|---|---|
| One writer per agent; a takeover raises the epoch | `test_one_writer_per_agent_and_a_takeover_increments_the_epoch` |
| A stale epoch cannot append; appends are idempotent on (agent, seq) | `test_a_stale_writer_is_fenced_and_appends_are_idempotent` |
| A lease is lost only if someone else took it | `test_an_expired_lease_nobody_took_can_be_renewed` |
| A dead worker's agent is rebuilt elsewhere, same block contents | `test_a_dead_worker_is_taken_over_and_its_memory_rebuilt_from_the_shared_tier` |
| The shared tier is the authority on replicated rows | `test_a_local_store_unknown_to_the_shared_tier_is_rebuilt_from_it` |
| Replication keeps up under continuous turns (RPO bounded by the tick) | `test_replication_keeps_up_under_back_to_back_turns` |
| One model per process, loaded before serving | `test_engines_of_one_process_share_one_embedding_model` |
| Tool results and prefix caches are shared | `test_tool_results_and_prefix_caches_are_shared_between_workers` |

## Things that look harmless and are not

- Anything CPU bound in a request handler (a model load, a big embedding batch without a
  thread) stalls every coordinator call of the worker and costs it its agents.
- A shared value copied locally keeps its producer's expiry, not a fresh one.
- A replaced prefix cache is not deleted in a cluster: another worker may be using it.
- Clients need a short connect timeout: a killed container is silent, it does not refuse.

## Degradation

Postgres down: turns continue locally, renewals fail quietly (nobody can take over either),
shared lookups are skipped, the journal queues and catches up. Measured in
`docs/measured/stress-cluster.md`.
