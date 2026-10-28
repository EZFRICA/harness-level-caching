---
name: prior-art
description: How the study positions itself against existing systems (agent memory, inference-side prefix and KV caching, tool and workflow caching, semantic caches), what may be cited and how, and what the study must show beyond each. Use before designing a mechanism, before claiming novelty, and when writing docs/prior-art or a comparison.
when_to_use: Designing a new cache level or policy; writing "unlike X"; citing a paper's number; choosing a baseline.
---

# Prior art

The owner's rule: always take what exists into account. A mechanism is designed after
checking who already did it, and a claim of novelty names what it is new against.

## The map (details and sources in `docs/prior-art*.md`)

| family | examples | where it acts |
|---|---|---|
| agent memory | MemGPT/Letta, Mem0, Zep/Graphiti, LangMem | what to remember |
| prefix and KV cache | provider prompt caching, vLLM prefix caching, SGLang RadixAttention, LMCache, KVFlow, Continuum | inference server |
| tool and workflow cache | hierarchical agentic caching, workload-aware multi-agent caching, agentic plan caching | tool calls, plans |
| semantic response cache | GPTCache | whole answers |

The study's claimed place: **the harness**, upstream of the inference server, making the prefix
stable by construction (render order apart from recency, cache-aware eviction, cold-path cache
creation) and caching the harness's own work (routing, embeddings, tools) in one hierarchy
with one invalidation story. That claim stands only as long as the map above says nobody
does it; recheck it when the map changes.

## Citing

- A number from a paper is cited only after reading it in the paper (section, table), with
  the URL and the access date. A summary site, an aggregator or a search snippet is a lead,
  not a source: label it "unverified".
- Name the baseline a paper used and whether its setting matches ours (server-side KV vs
  provider API, GPU vs CPU, single vs multi-agent).
- When an existing system can be used as a baseline in our harness, prefer measuring it to
  quoting it.
