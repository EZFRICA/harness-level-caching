# Prior art: KV and prefix caching in the inference server

Read on 2026-10-23 (URLs in `prior-art-sources.md`). All systems here run **inside a
self-hosted inference server on GPUs**. CAMA calls a provider API and cannot touch KV
tensors, so none of these numbers transfers to CAMA's setting; they show which ideas exist.

## vLLM automatic prefix caching (design doc, docs.vllm.ai)

- **Caches**: KV blocks keyed by a hash of the parent block hash, the block tokens and extra
  keys (LoRA id, image hash, optional `cache_salt` for tenant isolation).
- **Invalidation**: none needed (content addressed); eviction is LRU over the free queue.
- **Distributed**: single engine; sharing across engines is LMCache's job. **Numbers**: none.

## SGLang RadixAttention (arXiv 2312.07104v2)

- **Caches**: KV in a radix tree over token prefixes; LRU eviction of leaves and a
  cache-aware scheduler (longest shared prefix first) (Section 3).
- **Numbers**: up to 6.4x throughput and 3.7x lower latency (Section 6.2); one month in
  Chatbot Arena, hit rate 52.4% (LLaVA-Next-34B) and 74.1% (Vicuna-33B), one worker per model.
- **CAMA must show**: nothing against it directly; it is the server behind many providers.

## Prompt Cache (Gim et al., arXiv 2311.04934v2)

- **Caches**: attention states of "prompt modules" declared in a schema (PML), reusable at
  non-prefix positions with their own position ids (Section 3).
- **Numbers**: abstract says 8x (GPU) to 60x (CPU) TTFT; Section 5.2 says 1.5x to 3x (modules
  in CPU memory) and 5x to 10x (GPU memory) on GPUs, up to 70x and 20x on two CPUs (Llama 7B).
- **CAMA must show**: Prompt Cache is the closest idea to CAMA's blocks rendered in a fixed
  order. CAMA's difference is that it must work with exact-prefix provider caches, so it pays
  for order changes that Prompt Cache avoids. Name Prompt Cache when claiming the block idea.

## LMCache (arXiv 2510.09665v2)

- **Caches**: KV extracted from vLLM or SGLang, offloaded to CPU, disk, remote stores (Redis
  and others), shared across engines and queries; a controller API for admit and evict.
- **Distributed**: yes, the only system here with a cross-engine KV story (abstract, Section 1).
- **Numbers**: up to 15x throughput with vLLM (abstract). Real traces from one company:
  context truncation drops prefix hit ratio from about 85% to 45% (Section 9).
- **CAMA must show**: Section 9 is independent evidence for CAMA's premise (editing the start
  of the context destroys reuse); cite it as motivation, not as CAMA's result.

## CacheBlend (arXiv 2405.16444v3)

- **Caches**: KV of RAG chunks reused when not a prefix, recomputing about 15% of tokens
  (Sections 4.2, 5). **Numbers**: TTFT 2.2x to 3.3x lower, throughput 2.8x to 5x versus
  full recompute (abstract, Section 7, Figure 12).

## KVFlow (arXiv 2507.07400v1)

- **Caches**: KV of agents' fixed prompts; evicts by "steps to execution" on an Agent Step
  Graph instead of LRU, and prefetches from CPU (Section 3). Built on SGLang v0.4.4 (Section 3.3).
- **Numbers**: up to 1.83x (single workflow, A10G) and 2.19x (many concurrent workflows)
  versus SGLang with hierarchical radix cache (Sections 4.1, 4.2).

## Continuum (arXiv 2511.02230v7; written "Continnum" in the text)

- **Caches**: pins a finished turn's KV in GPU during a tool call with a TTL set from reload
  cost and queueing delay (Sections 4, 5.2).
- **Numbers**: abstract says average job completion time improves "by over 8x"; Section 1 says
  1.12x to 3.66x lower delay across three setups and up to 8.18x on one internal testbed.

## CacheScout (arXiv 2608.14624v1, "Learning Agent Execution for KV-Cache Management")

- **Caches**: KV of agent fixed contexts; learns agent transitions online to drive eviction
  and prefetch, on vLLM. Fixed contexts are 53% to 62% of prompt tokens in its four
  workloads (Section 1, Figure 2).
- **Numbers**: hit rate +10 to 18 points, mean TTFT 18% to 45% lower, per-turn latency 29% to
  38% lower, peak throughput up to 57% higher (abstract; Section 5, Figure 10).

## Leyline (arXiv 2606.01065v1)

- **Caches**: none new; a serving primitive letting a policy splice or trim a cached span
  without re-prefilling what follows, with a RoPE rotation correction (Section 3).
- **Numbers**: replay cache hit 49.6% to 60.8% (+11.2 points) on SGLang, DeepSeek-V2-Lite,
  about 17K-token prompts; p50 latency up to 241 ms (4.4%) lower at C=8 (Section 5). Solve
  rate 10/32 to 15/33 (+14.3 points) on debug-gym: small samples (Table 2).
- **CAMA must show**: Leyline states that production harnesses "fall back to re-prefill on
  every edit". CAMA's answer through a provider API is to avoid edits in the cached prefix;
  it must measure what that constraint costs in memory freshness.
