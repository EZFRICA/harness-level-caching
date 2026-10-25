# Prior art: where CAMA stands

Read on 2026-10-23. Every number below was read in its original source (paper PDF, official
documentation or repository) at the location given in the detail files. A number that could
not be read in a primary source is labelled **unverified** and is never used as a fact.

| file | content |
|---|---|
| `prior-art-memory.md` | agent memory: MemGPT/Letta, Mem0, Zep/Graphiti, LangMem, A-MEM |
| `prior-art-providers.md` | provider prompt caching (Gemini, Anthropic, OpenAI), Gemini prices, Manus, "Don't Break the Cache" |
| `prior-art-serving.md` | inference-server KV caching: vLLM, SGLang, Prompt Cache, LMCache, CacheBlend, KVFlow, Continuum, CacheScout, Leyline |
| `prior-art-tools.md` | tool, workflow and plan caching (hierarchical agentic caching, Cortex, workload-aware, APC) and GPTCache |
| `prior-art-gateways.md` | gateways that route with the cache in mind (OpenRouter sticky routing, LiteLLM affinity and auto router, Portkey) |
| `prior-art-compaction.md` | compaction and memory for long sessions (ARC, CWL, LightMem, TokenPilot, Anthropic and ADK compaction) against cama's two measured defects |
| `prior-art-sources.md` | every source with its URL, version and access date |

## The map

| family | systems | what is cached | layer | invalidation |
|---|---|---|---|---|
| agent memory | MemGPT/Letta, Mem0, Zep, LangMem, A-MEM | facts, summaries, notes | harness | LLM decides (update, delete, edge expiry) |
| provider prompt cache | Gemini, Anthropic, OpenAI | KV of an exact prefix | provider | TTL; any change before the hit point |
| serving KV cache | vLLM, SGLang, LMCache, Prompt Cache, CacheBlend | KV blocks | inference server | LRU, hash mismatch, recompute |
| agent-aware KV | KVFlow, Continuum, CacheScout, Leyline | KV of agent prompts | inference server | workflow or learned eviction, TTL pinning, directives |
| tool and plan cache | hierarchical caching, Cortex, workload-aware, APC | tool results, workflows, plans | harness | TTL, dependency graph, staticity, none |
| semantic response cache | GPTCache | whole answers | harness | count-based eviction |

## What the map says about CAMA's claim

1. **Nobody found in this review stabilises the provider prefix by construction from a memory
   system.** The memory systems (MemGPT, Mem0, Zep, A-MEM) select per query or rewrite the
   context, which changes the prefix; none of their papers measures prompt cache hits. The
   provider docs and the practitioner advice (Manus, "Don't Break the Cache") say *keep the
   prefix stable* but give no mechanism that keeps it stable while memory changes.
2. **Recency-blind, reuse-aware eviction already exists on the server side** (KVFlow,
   CacheScout, Continuum, the workload-aware policy). CAMA's cache-aware eviction is new only
   at the harness level and against a provider API it does not control. It must not be
   presented as a new eviction idea.
3. **Reordering or editing cached content is solved server-side** by Prompt Cache (schema
   modules), CacheBlend (non-prefix reuse) and Leyline (span directives). None of these is
   reachable through a provider API, which only reuses an exact prefix. That is the gap CAMA's
   stable render order targets.
4. **Tool result caching with TTLs, dependency invalidation and tenant isolation exists**
   (hierarchical caching, MDPI 2026). CAMA's H3 is not novel alone; a single hierarchy and
   one invalidation story across H1 to H4 is the claim, and it is untested until measured.
5. **Distribution and fault tolerance**: the memory and tool caches reviewed describe no
   lease, epoch fencing or replay story; LMCache shares KV across engines; Anthropic and
   OpenAI isolate caches per workspace or organization. CAMA's journal, leases and fencing
   are an engineering contribution, to be shown by its stress profiles, not by citation.

## What CAMA must demonstrate (summary; details per system in the files)

- Prompt cache hit share and cost per turn **against a Mem0-style or A-MEM-style top-k
  baseline** on the same scenario, model and price table (`topk` mode is that proxy).
- That memory quality does not drop: a retrieval or QA metric next to the cost figure, since
  every memory paper reports one (DMR, LoCoMo, LongMemEval).
- The cold-path cost: explicit cache creation and storage at the official storage price, and
  extraction calls, as `cama-cost` requires. A hot-path gain paid back cold is not a gain.
- The ablations `--no-prefix-cache`, `--no-tool-cache`, `--no-embed-cache`, and an LRU tool
  cache baseline, because the MDPI paper shows a plain LRU-512 reaching the same efficiency
  as its full system (76.9% versus 76.5%).
- Behaviour when the provider removes explicit caching: Gemini docs say explicit caching is
  not supported from Gemini 4 Argon onward. CAMA must report its gain with implicit caching
  only, or say the H4 design depends on a feature being withdrawn.

## Flags raised by this review

- **Unverified, kept out of the argument**: GPTCache's README banner ("10x" cost, "100x"
  speed: no measurement attached); Mem0 README benchmark table (managed platform, not the
  paper); any number seen only in search snippets.
- **Inconsistent within the source**: Prompt Cache abstract (8x GPU, 60x CPU) versus Section
  5.2 (5x to 10x GPU memory, up to 70x CPU); Continuum abstract ("over 8x") versus Section 1
  (1.12x to 3.66x across setups, 8.18x on one internal testbed); Zep "18.5%" versus its Table
  2 values (60.2% to 71.2%, which is 18.3% relative).
- **Gemini 3.5 Flash-Lite is absent from the implicit minimum-token table, and the explicit
  caching section gives no minimum at all**: CAMA's own observations on these thresholds are
  measurements, not documented facts, and must cite `docs/measured`.
