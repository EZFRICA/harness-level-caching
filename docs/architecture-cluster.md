# Architecture: the distributed tier

```
 load generator ──(rendezvous hash, failover)──> worker w1 | w2 | w3   (FastAPI, port 8797 each)
                                                   │ local SQLite per agent, one ONNX model per process
                                                   ▼
                     Postgres: leases(epoch) · journal(agent, seq) · snapshots
                               prefix_caches · tool_cache · embed_cache · workers · events
```

- **Ownership.** A worker serves an agent while it holds the lease; a takeover after expiry
  raises the epoch; the journal refuses appends with a lower epoch (fencing).
- **Replication.** Every second, idle or not; idempotent on (agent, seq). RPO is bounded by the
  tick; measured 0 rows at a `kill -9`.
- **Recovery.** Latest snapshot, then the journal after it; vectors recomputed through H2. The
  shared tier is the authority: a local store it does not know is rebuilt.
- **Shared caches.** Tool results, vectors and prefix cache names, each behind a local copy;
  a shared tool result keeps its producer's expiry.
- **Degradation.** Postgres down: turns continue, renewals and shared lookups are skipped, the
  journal catches up later (measured on a 30 s outage).
- **Limits measured.** `docs/measured/stress-cluster.md`: RTO about 10 s (lease 6 s plus client
  retries), throughput and tail latency under faults.

Run it: `./scripts/cluster_fresh.sh <run id>`, then `harness-cache load` from the `hc` container.
