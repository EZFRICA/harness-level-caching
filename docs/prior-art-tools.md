# Prior art: tool, workflow, plan and response caching

Read on 2026-10-23 (URLs in `prior-art-sources.md`). These act in the harness, like CAMA's
H3, and are the direct competitors of its tool-result cache.

## Hierarchical Caching for Agentic Workflows (Begum et al., MAKE 2026, 8(2), 30)

- **Caches**: whole workflows (execution sequences) above individual tool results.
- **Invalidation**: dependency graph (a write invalidates the cached reads it affects,
  Section 3.3, Algorithm 2) plus TTLs per data category (300 s gave a 59.4% hit rate,
  Section 3.1); session isolation through scoped keys (Section 4.6).
- **Distributed**: Redis 7.0.8 backend; runs single-threaded on one Windows 11 server
  (Section 4.2). No replication or failover is evaluated.
- **Numbers**: synthetic Zipfian workload, 15 runs. Efficiency 76.5% (95% CI 76.4 to 76.7),
  query time 0.31 s to 0.023 s (13.3x) **excluding LLM inference**, estimated tool cost -73.3%
  (Section 5.1, Table 4). Plain LRU-512: efficiency 76.9%, 12.1x; unbounded memoization 16.1x
  (Section 5.1). 15 concurrent tenants: 74.1% efficiency (abstract).
- **CAMA must show**: its H3 beats a plain LRU tool cache on a real agent trace (this paper's
  own LRU baseline matches its efficiency), and that invalidation is correct under writes.

## Cortex (arXiv 2509.17360v2; v1 titled "Asteria")

- **Caches**: results of remote knowledge access (web search, RAG services) across regions.
  A Semantic Element stores the query embedding with latency, cost and "staticity"; lookup is
  approximate nearest neighbour then a small LLM judge (abstract, Section 4).
- **Invalidation**: staticity estimates validity; TTL purge; cost-aware eviction (Section 4).
- **Distributed**: cross-region by design (asymmetric partitioning, Section 4).
- **Numbers**: up to 3.6x throughput with hit rates above 85% on search workloads versus
  exact-match caching (abstract, Section 1).
- **CAMA must show**: if H3 ever matches by similarity rather than exact key, compare with
  Cortex's judge; with exact keys only, say so and accept the lower hit rate.

## Workload-Aware Caching for Multi-Agent Systems (arXiv 2607.20495v1)

- **Caches**: intermediate agent results in a DAG, keyed by (task description, subject id)
  (Section II); eviction scores recomputation cost, DAG dependents and invocation frequency.
- **Invalidation and distribution**: none described (text search for stale, expire,
  invalidate and TTL found no mechanism).
- **Numbers**: latency up to 64.7% lower than uncached and 31.1% lower on average than the
  next best finite-capacity policy (abstract, Section I, results in Section IV-B).
- **CAMA must show**: cost-aware eviction in H3 is this paper's idea; cite it.

## Agentic Plan Caching, APC (arXiv 2506.14852v2)

- **Caches**: plan templates extracted from finished runs, matched by keywords, adapted by a
  small model (Section 3). Eviction LRU in the size study (Section 4.4, Table 4).
- **Invalidation**: none described. **Distributed**: not addressed.
- **Numbers**: cost -50.31% and latency -27.28% on average, keeping 96.61% of optimal accuracy
  (Sections 4.2, 4.3); the latency figure comes from one workload where 46 of 100 queries hit.
- **CAMA must show**: nothing, unless CAMA caches plans; it is complementary.

## GPTCache (repository zilliztech/GPTCache), cited as out of scope

- **Caches**: whole LLM answers matched by embedding similarity in a vector store.
- **Invalidation**: eviction by number of entries, LRU supported; Redis for distributed
  eviction (README). No freshness mechanism described.
- **Unverified**: the README banner "Slash Your LLM API Costs by 10x, Boost Speed by 100x" has
  no measurement attached. Not cited as a result.
- **Why out of scope**: CAMA never returns a cached answer; an agent's answer depends on tool
  results and state, which is the reason APC and Cortex give for not using response caches.
