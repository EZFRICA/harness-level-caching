# Prior art: agent memory

Read on 2026-10-23. Sources and URLs in `prior-art-sources.md`. A text search of the four
papers below (MemGPT, Mem0, Zep, A-MEM) for "prompt cach", "prefix cach", "KV cache",
"context cach" and "cache hit" found **no occurrence**: none measures provider cache reuse.

## MemGPT / Letta (arXiv 2310.08560v2)

- **Caches**: nothing in the cache sense. It pages text between main context (system
  instructions, working context, FIFO queue) and external context (recall and archival
  storage), like an OS virtual memory (Section 2, Figure 3).
- **Layer**: harness. **Invalidation**: a queue manager warns the LLM at a warning token
  count (example 70% of the window) and flushes at 100%, evicting messages (example 50%) and
  rewriting a recursive summary (Section 2.2). The summary rewrite changes the prompt prefix.
- **Distributed**: archival storage on PostgreSQL in the document QA experiment (Section
  3.2.1); no fault-tolerance story in the paper. Letta (f.k.a. MemGPT) now lives in
  `letta-ai/letta-code` per its README.
- **Numbers**: DMR accuracy 93.4% with GPT-4 Turbo + MemGPT versus 35.3% without (Table 2).
- **The study must show**: that paging into a bounded, stably rendered active set keeps provider
  cache hits where MemGPT's summary rewrite and FIFO eviction would break them, with a DMR or
  equivalent recall metric beside the cost.

## Mem0 (arXiv 2504.19413v1)

- **Caches**: extracted facts (and a graph variant, Mem0g). An LLM chooses ADD, UPDATE,
  DELETE or NOOP per candidate fact (Section 2).
- **Layer**: harness, retrieval per query. **Invalidation**: LLM-driven DELETE of contradicted
  memories. **Distributed**: not described in the paper.
- **Numbers** (Table 2, LOCOMO): p95 total latency 1.440 s for Mem0 versus 17.117 s full
  context (the abstract's "91% lower"); memory tokens 1,764 versus 26,031 full context (the
  abstract's "more than 90% token cost"); J score 66.88% versus 72.90% full context. Memory
  footprint 7k tokens per conversation, Mem0g 14k, Zep above 600k (Section 4.5).
- **Unverified**: the README benchmark table (LoCoMo 92.5, 7.0K tokens) reports Mem0's managed
  platform with "proprietary optimizations"; it is not in the paper.
- **The study must show**: lower cost per turn than per-query top-k retrieval once prompt caching
  is counted (Mem0's baseline is full context without caching), at a comparable judge score.

## Zep / Graphiti (arXiv 2501.13956v1; repository getzep/graphiti)

- **Caches**: a temporal knowledge graph of episodes, entities, facts and communities.
- **Layer**: harness service. **Invalidation**: bi-temporal edges (t_valid, t_invalid,
  t'_created, t'_expired); an LLM detects contradictions and sets t_invalid, new information
  wins (Section 2.2.3). The repository says facts are "invalidated, not deleted".
- **Distributed**: graph database backend (Neo4j in the paper, Section 3.1; the repository
  lists Neo4j, FalkorDB, Amazon Neptune). No fencing or replay story in the paper.
- **Numbers**: DMR 94.8% versus MemGPT 93.4% with gpt-4-turbo (Table 1); LongMemEval with
  gpt-4o: 71.2% versus 60.2% full context, latency 2.58 s versus 28.9 s, context 1.6k versus
  115k tokens (Table 2). The text says "18.5%" improvement; Table 2 gives 18.3% relative.
- **The study must show**: a validity-time story for its own blocks at least as explicit as Zep's,
  or say it has none; and the cost of graph construction (cold path) against its own.

## LangMem (LangChain docs)

- **Caches**: semantic, episodic and procedural memories in LangGraph's BaseStore.
- **Layer**: harness, two modes: memory tools "in the hot path" and a background memory
  manager that extracts and consolidates. **Invalidation**: consolidation by the manager.
- **Distributed**: inherits the LangGraph store; the docs show `InMemoryStore`.
- **Numbers**: none in the docs. Mem0's Table 2 measured LangMem search at p50 17.99 s, p95
  59.82 s (a competitor's measurement, cited as such).
- **The study must show**: nothing specific; its hot path versus cold path split is the same idea
  as LangMem's background manager and must be presented as such, not as new.

## A-MEM (arXiv 2502.12110v1)

- **Caches**: Zettelkasten-style notes (content, keywords, tags, context, links); new notes
  trigger "memory evolution" that rewrites older notes' attributes (Section 3.3).
- **Layer**: harness. **Invalidation**: evolution by the LLM. **Distributed**: not described.
- **Numbers**: average 2,520 tokens per answer with GPT-4o-mini versus 16,910 for the LoCoMo
  baseline and 16,977 for MemGPT (Table 1).
- **The study must show**: that rewriting old memory (A-MEM evolution) and a stable prefix can
  coexist: an edited block either moves to the tail or costs a measured cache miss.
