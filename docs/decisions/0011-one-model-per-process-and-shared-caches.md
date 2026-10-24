---
id: 0011
status: accepted
evidence: cluster-s6-3w-v1
held_by: tests/test_cluster.py::test_engines_of_one_process_share_one_embedding_model, tests/test_cluster.py::test_tool_results_and_prefix_caches_are_shared_between_workers
---
# 0011: A worker loads its model once, before serving, and shares cache entries with the others

Decided: 2026-10-23, proposed by Claude after the first cluster runs, within the owner's plan.

## Context
Three faults of the first cluster runs came from where the embedding model lived: one ONNX
model per agent put 1 GB workers out of memory with three agents; loading it inside the first
request blocked the event loop long enough for every coordinator call to time out.

## Decision
- One embedder per process and embedding space, created and warmed before the worker accepts
  requests, off the event loop.
- Tool results, vectors and prefix cache names are shared through the coordinator, each with a
  local copy in front; a shared tool result keeps its producer's expiry. In a cluster a
  replaced prefix cache is left to expire rather than deleted, since another worker may use it.

## Consequences
- Workers hold about 700 MB with the model loaded; agents add little on top.
- A worker that misses locally may read the shared tier on the hot path: one Postgres round
  trip, skipped when the tier is down.

## Alternatives set aside
- A separate embedding service: one more network hop on every routed turn.
