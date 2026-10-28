---
name: add-a-cache-level
description: The contract every level of the harness cache must satisfy (content-addressed key, where it is looked up and where it is filled, invalidation, what may never be cached, the per-turn record, its cost line, its ablation switch, its property test and fault, its decision). Use when adding or changing a cache level (routing, embeddings, tool results, prefix, any future one) or a cache key, TTL or invalidation rule.
when_to_use: Building H2 (embeddings) or H3 (tool results); changing the H4 prefix cache; adding a TTL or a class; reviewing whether something may be cached at all.
---

# Adding a level to the harness cache

The harness cache is one hierarchy (decision 0006). A level earns its place only if it can be measured
apart, switched off, and trusted under faults.

## The contract, one line each

| item | requirement |
|---|---|
| key | content addressed: a hash of everything the value depends on (text and embedding space; tool name and normalised arguments; rendered prefix and model). Nothing implicit such as wall time or a worker id. |
| lookup | on the hot path, O(1) or bounded, never a network call it cannot skip |
| fill | where the value is produced anyway (hot path) or on the cold path when producing it costs the user (as for H4) |
| invalidation | stated: never (pure), time to live, or an event (a block version, a prefix hash) |
| never cached | anything with a side effect (a booking, a payment, a write) and anything whose answer depends on who asks, unless the key carries it |
| record | per turn: hits, misses by cause, bytes or tokens saved, in `turns.jsonl` |
| cost | a line in the cost model: what a hit saves, what the cache itself costs (storage, creation) |
| switch | an ablation flag (`HC_<LEVEL>=off`) that keeps everything else identical |
| test | a property test over random sequences (a hit returns exactly what a miss would compute; a side effect is never served from cache) |
| fault | what happens when the cache is wrong, missing or corrupted: the turn still completes, uncached |
| decision | a record citing the run that measured it, and the test that holds it |

## Order of work

1. Write the decision draft (key, invalidation, never cached) and check `prior-art` for who
   already caches this and how they invalidate it.
2. Implement behind the switch; record hits and misses by cause.
3. Property test, then a mutant that breaks the key or the invalidation and must be caught.
4. Run the experiment with the level on and off (`run-an-experiment`).
5. Add the dashboard field and the cost line, then accept the decision.

## Traps already met

- H4: a prefix under the provider minimum (1024 tokens on Gemini) cannot be cached; record
  it as a miss cause, not as an error.
- H4: a page fault changes the prefix inside the turn; the miss is structural.
- Embeddings: two spaces under one key would serve vectors from the wrong geometry; the space
  is part of the key.
