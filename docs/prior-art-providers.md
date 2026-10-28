# Prior art: provider prompt caching

Read on 2026-10-23 in the official documentation (URLs in `prior-art-sources.md`). Provider
docs change often; every figure here holds for that date only.

## Gemini API (generateContent caching page, "Last updated 2026-10-23 UTC")

- **Implicit caching**: on by default for Gemini 2.5 and newer, "no cost saving guarantee".
  Minimum input tokens listed: 4,096 (Gemini 4 Argon, 3.8 Flash, 3.6 Flash, 3.1 Pro Preview),
  2,048 (2.5 Flash, 2.5 Pro). **Gemini 3.5 Flash-Lite is not listed.** On Gemini 4 agent
  loops, the full previous request is cached only if the new one adds at least 2,050 tokens.
- **Explicit caching**: Beta, v1beta, "cost saving guarantee", on Gemini 3 and 2.5 models;
  **not supported for Gemini 4 Argon and onward**. TTL defaults to 1 hour, no minimum or
  maximum bound. Billing: cached tokens at a reduced rate plus storage per hour of TTL. The
  minimum token count "varies by model" and no figure is given. "Cached content is a prefix
  to the prompt." No cache creation price is stated.
- **Layer**: provider. **Invalidation**: TTL expiry or deletion; implicit hits need an
  identical prefix. **Distributed**: not the user's concern; no isolation scope is stated.

### Gemini 3.5 Flash-Lite prices (for `pricing.toml`)

Pricing page `ai.google.dev/gemini-api/docs/pricing`, "Last updated 2026-10-22 UTC", read
2026-10-23, status `official`. Model id `gemini-3.5-flash-lite`, USD per 1M tokens, paid tier:

| tier | input | output (incl. thinking) | context cache read | cache storage |
|---|---|---|---|---|
| Standard | 0.30 | 2.50 | 0.03 | 1.00 per 1M tokens per hour |
| Batch | 0.15 | 1.25 | 0.02 | 1.00 per 1M tokens per hour |
| Flex | 0.15 | 1.25 | 0.02 | 1.00 per 1M tokens per hour |

Free tier: input and output free, context caching "Not available". Cache **creation** has no
listed price: billing it at the input price stays an `assumption`, as the cost rules in `harness_cache.cost` says.

## Anthropic prompt caching (docs.claude.com)

- **Mechanism**: `cache_control` on up to 4 breakpoints, or automatic caching (uses one of the
  4 slots). Prefix order is tools, system, messages. The lookback checks 20 blocks per
  breakpoint and only finds entries written by earlier requests.
- **Price**: write 1.25x base input (5 minute TTL) or 2x (1 hour TTL); read 0.1x base input,
  with model exceptions listed on the page (2.5% or 5% on some current models).
- **Minimum**: 512 to 4,096 tokens depending on the model; shorter prompts are processed
  uncached with no error.
- **Invalidation**: a change at a level invalidates it and all later levels (tool
  definitions invalidate everything); TTL 5 minutes by default, refreshed free on each hit.
- **Isolation**: per workspace on the Claude API; per organization on Bedrock and Google Cloud.

## OpenAI prompt caching (platform.openai.com)

- **Mechanism**: on by default; exact prefix; routing by a hash of the initial tokens and an
  optional `prompt_cache_key`. Cached input "discounted up to 95%".
- **Minimum**: 1,024 tokens for GPT-5.6 and later; varies for earlier models.
- **Lifetime**: GPT-5.6+: `ttl` "30m" (write 1.25x input) or "12h" (write 2.5x); earlier
  models: `in_memory` (about 5 to 10 minutes idle, up to 1 hour) or `24h`, no write charge.
- **Isolation**: within an organization and processing region.

## "Don't Break the Cache" (arXiv 2601.06007v2, Lumer et al.)

- **What**: the only study found that measures provider prompt caching on agent sessions
  (DeepResearch Bench, 10,000-token system prompt, 40 sessions per model, Section 3.1).
- **Numbers**: cost reduction 79.6% (GPT-5.2, excluding tool results), 78.5% (Claude Sonnet
  4.5, system prompt), 41.4% (Gemini 2.5 Pro, system prompt), 45.9% (GPT-4o) (Table 1). The
  abstract says TTFT improves 13% to 31%; the results text beside Table 1 says 6% to 31%
  (Gemini 6.1%).
- **Finding**: naive full-context caching can increase latency; excluding dynamic tool
  results and putting dynamic content last is more consistent (abstract, Section 5.1).
- **CAMA must show**: a gain over its best strategy ("system prompt only" or "exclude tool
  results"), not only over no caching. This is the baseline closest to CAMA's setting.

## Manus, "Context Engineering for AI Agents" (blog, Yichao Ji, 2025-07-18)

- **What**: practitioner guidance, not a measured system. Calls KV-cache hit rate "the single
  most important metric" for a production agent; advises a stable prefix (no timestamp at the
  top), an append-only context, deterministic serialization, masking tools instead of
  removing them, and the file system as external memory.
- **Unverified**: the 100:1 input to output ratio is a self-report about Manus, with no data.
- **CAMA must show**: that it mechanises these rules for a memory that changes (the advice
  assumes an append-only context; CAMA's active set evicts and admits blocks) and measures it.
