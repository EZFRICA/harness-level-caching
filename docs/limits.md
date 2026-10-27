# Limits and open problems

What CAMA does not do, or does badly, as of the last ledger event. Each line names where it
was measured or found. A line leaves this page only through a ledger event that fixes it.

## Measured limits

| limit | where | consequence |
|---|---|---|
| The memory extraction is the largest cold cost; over 24 agentic turns CAMA cost more than naive | `docs/measured/agentic.md` | the cold path, not the prompt, decides the bill on short sessions |
| Explicit prefix caching made turns slower on gemini-3.5-flash-lite (about 2.5 s against 1.8 s) | `docs/measured/agentic.md` | cost against latency is a deployment choice |
| A memory under 1024 tokens cannot be cached explicitly; over 200 turns cama never reached it | `docs/measured/stress-sessions-gemini.md` | the saving came from the bounded prompt alone |
| Gemini 4 Argon has no explicit caching and caches implicitly from 4096 tokens | `docs/prior-art-providers.md` | on the production model only a large stable prefix can be cached |
| The ONNX embedding model needs more than 512 MB | `docs/measured/stress-device.md` | a 2 GB device needs a smaller embedder |
| After a worker kill, an agent answers again after about 10 s | `docs/measured/stress-cluster.md` | bounded by the 6 s lease and client retries |
| Doing the cold path's work costs throughput (4.2 to 2.9 requests per second on three workers) | ledger, fault events | the earlier figure was a starved cold path |
| Many agents opening at once on one worker queue longer than client retries (28 of 384 lost at 32 agents) | `docs/measured/stress-scale.md` | a cold start limit; steady state holds |
| With append-only extraction a corrected fact stays next to its stale value; cama recalled the stale one in 1 of 2 runs (fixed by `CAMA_EXTRACT_REPLACE=on`, 13/13 in 4 runs) | `docs/measured/recall.md` | replace is a switch, not yet the default |
| An exact-key tool cache served 1 of 15 to 18 real web searches: the model rephrases its queries | `docs/measured/web.md` | H3 pays on deterministic tools, rarely on model-written queries |
| A semantic key (cosine 0.90) served 2 more searches per run but 1 of 4 matches crossed a year (2024 got 2025) | `docs/measured/web.md` | a semantic tool key needs dates in the key or a judge |
| On a 32 turn session naive recalled every fact and cost less than any cama variant | `docs/measured/recall.md` | cama's case rests on long sessions |
| Over 200 turns on Nebius naive (97.7% cached) cost 0.0219 USD; the best cama (decision 0013, proposed) 0.0236, level only with batched extraction | `docs/measured/long-session.md` | with cheap cached reads an append-only history is hard to beat; cama's edge is a bounded prompt |
| Extraction batched over 4 or 10 exchanges lost a correction in 3 of 6 runs; every 2 lost none in 6 | `docs/measured/recall-batching.md` | the cheapest extraction schedule is not safe yet |
| On an implicit-only provider the stable order raised the cached share (75 to 78% against 55 to 63% for topk) but sent about 50% more prompt tokens, so the hot cost did not fall | `docs/measured/nebius.md` | a higher cached share is not a lower bill; extraction still dominates |

| Implicit cache life, one sample per delay: Gemini 3.5 Flash-Lite hit at 15 s and 2 min, never from 5 min; Gemini 4 Argon never hit, even at 15 s on an identical 10k token prefix; Nebius hit at 15 s, 5, 15, 25 min and missed at 2, 10, 20, 30, 45, 60 min | `docs/measured/probes/cache-lifetime-2026-10-24.jsonl` | Flash-Lite's cache lasts minutes, not the 15 to 20 claimed by secondary sources; on Nebius the replica a request reaches matters as much as idle time |
| Self-hosted vLLM: naive stopped at turn 198 when its history passed the 16384 token context | `docs/measured/vllm-l4.md` | naive's cheapness ends at the context window; cama's prompt stays bounded |

| Extracting two exchanges per call (decision 0013, `CAMA_EXTRACT_EVERY=2`) lost facts with GLM-5.2 (8 of 13) and GLM-5.3 (11 of 13) where every-turn extraction kept them | `docs/measured/providers.md` | 0013's batching setting is model dependent; extract every turn unless a model is shown to tolerate batches |
| A small extractor (DeepSeek-V2-Lite, 16B) timed out on 19 of 47 extraction calls and evicted planted facts | `docs/measured/providers.md` | cama needs an extractor that returns clean JSON |

| **Decisive test, 500 turns: a history folded into a summary (compact) recalled 24 of 24 for about 0.05 USD; cama with decision 0013 recalled 21 and 22 for about 0.074 USD** | `docs/measured/decisive.md` | on this test cama loses to the simple alternative in both recall and cost; per-turn extraction is the cost. The three missed facts were stored and paged out to L3, and the page fault did not bring them back: a routing failure, not a loss (checked in both runs' stores) |

| Gemini implicit cache on 2026-10-27, ~13.3k token prompts with a ~10.1k token identical prefix: 0 hits in 54 of 54 valid arms on the Gemini API and 41 of 41 on Vertex, every idle delay from 15 s to 30 min, and 0 in 20 sequential calls 2 s apart; on 2026-10-24 Flash-Lite had hit at 15 s and 2 min | `docs/measured/probes/cache-lifetime-*-2026-10-27.jsonl` | no cache life can be stated for Gemini; implicit caching did not reproduce across days on this account |

## Open problems

- **Nebius was unstable on 2026-10-24 afternoon** (503 on 8 of 12 calls over a minute): the
  replicates of the 200 turn rows, statement-triggered extraction and the render order ablation
  are not measured yet (only one ablation pair completed).
- **On Gemini 4 Argon naive was the cheapest** over 32 turns (`docs/measured/argon.md`): a large
  memory is re-billed when calls add under 2050 tokens. cama's case on Argon is unproven.

- **The stable render order has no ablation.** With an implicit-only provider
  `CAMA_PREFIX_CACHE=none` changes the label, not the request; decision 0002 needs a switch that
  renders by recency to be tested directly.
- **Run manifests carry the last commit's sha**, so runs on uncommitted code are not told apart
  by code version.

- **Conversation history is not replicated.** After a takeover the new owner serves an empty or
  stale recent window; only memory blocks and the journal are recovered.
- **Recovery reads the whole journal** when no usable snapshot exists: recovery time grows with
  an agent's age.
- **Answer quality is not measured** anywhere. Every cost or latency gain is shown without the
  evidence that answers stayed as good; a quality metric (a judged QA set, LongMemEval or
  LoCoMo style) is the next measurement.
- **Routing thresholds are in sample** for the MiniLM space; no holdout calibration yet.
- **Cache-aware eviction (0003) has no measured effect**: no scenario forced enough evictions.
- **The H2 cache rarely hits within one session**; its value across workers and restarts is
  designed but not yet measured on its own.
- **The echo provider's caches live in one process**: in cluster runs with the echo model a
  cache name shared by another container fails once, then the step falls back uncached.
- **The dashboard stream rereads frozen runs every second**: fine at a hundred runs, not at
  thousands.
