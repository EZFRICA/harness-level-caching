# Architecture

What the code does today. Why: `docs/decisions/`. The distributed tier:
`docs/architecture-cluster.md`.

## One turn

```
HOT PATH (the user waits)
  query -> H1 routing -> assembler -> agent loop (model, tools, model ...) -> answer
           H1: words over active blocks, then active block vectors (H2), then L3 (page fault)
           assembler: instructions + active set in render order + recent window + question
           H4: if the provider holds this exact prefix (system + tool declarations), send its name
           tools: H3 first (pure, ttl, never for side effects), bounded retries, errors to the model
COLD PATH (never awaited by a turn)
  extraction   pending exchanges in one model call (0009), runs while no turn is in progress
  H4 creation  after extraction, only for a prefix expected to be sent again (0010)
  replication  the journal shipped every second, idle or not (0009)
```

## Modules (`src/harness_cache/`)

| module | role |
|---|---|
| `memory/active_set.py` | bounded working memory, recency apart from render order, cache-aware eviction (0002, 0003) |
| `memory/router.py`, `memory/keywords.py` | H1: the cascade, or apu-urithi's whole-store search with BM25 fused to vectors (`HC_ROUTE_POLICY=urithi`); page faults; read-only recall for compact (H9) |
| `memory/embed_cache.py` | H2, content addressed by text, kind and embedding space |
| `teu/tools.py`, `teu/cache.py` | simulated tools, H3 with cache classes |
| `agent/loop.py` | model and tools within a turn, bounded retries |
| `memory/assembler.py`, `memory/prefix_cache.py` | the request, H4 lookup and creation (0005, 0008) |
| `memory/extractor.py` | memory extraction, batched |
| `store/sqlite_store.py` | local source of truth, journal, snapshots, rebuild (0001) |
| `cost.py` | price table, run cost, budget guard (0008) |
| `engine.py`, `scheduler.py` | one turn; modes `extract`, `topk`, `naive`, `compact`; the cold path |
| `cluster/` | coordinator, worker service, recovery, load generator (0007, 0011) |
| `dash/` | read only dashboard on port 8797 |
| `trace/` | ledger, generated figures, evidence linked decisions |
| `stress/gen.py` | long generated scenarios |

## Invariants held by tests

- The active set never exceeds its bounds; pinned blocks stay; a read never reorders the prompt.
- The hot path never creates a provider cache and never awaits a cold task.
- A side effect tool is never served from cache; a cache hit returns what a fresh call computed
  when it was stored, within its TTL.
- Vectors are unit norm and of one embedding space per store and per cache key.
- A paid call is refused before it is made when it would cross the budget.
- The ledger is a hash chain; journal and figure pages equal their regeneration.
- Cluster rules: `.claude/skills/cluster-contract/SKILL.md`.

## Modes for comparison

`extract` renders the whole active set in stable order; `topk` renders pinned blocks plus this
query's hits (the APU); `naive` sends the full conversation; `compact` sends the history and
folds its older part into a running summary past `HC_COMPACT_AT` tokens, and with
`HC_VERBATIM_RECALL=on` (H9) also appends the user's own past messages found by the urithi
search, verbatim, at the end of the prompt. Ablation switches:
`HC_PREFIX_CACHE=none`, `HC_TOOL_CACHE=off`, `HC_EMBED_CACHE=off`,
`HC_CACHE_ADMISSION=always`, `HC_VICTIM_WINDOW=1`.
