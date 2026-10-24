# Prior art: LLM gateways that route with the provider cache in mind

Read on 2026-10-24 in each vendor's own documentation (URLs below). These pages are primary
sources for what the products do; their savings figures are **vendor reported**, measured by
the vendor on its own data, and are cited as such, never as independent results.

Why this page exists: the owner proposed turning CAMA into a cache orchestrator that exploits
existing caches. The review below shows that cache-aware routing between providers and models
already ships in gateways, so that part is not new.

## OpenRouter: provider sticky routing

- After a request that used prompt caching, later requests of the same conversation go to
  the same provider endpoint, for implicit caches (OpenAI, DeepSeek, Gemini 2.5) and explicit
  ones (Anthropic `cache_control`).
- Only when the provider's cache read price is below its prompt price. Disabled when the user
  fixes a provider order. Falls back to the next provider when the sticky one fails.
- A conversation is identified by hashing the first system message and the first non-system
  message, or by an explicit `session_id`. Stickiness expires after **10 minutes** idle, reset
  by each successful request.
- Numbers: none on that page.

## LiteLLM: session affinity, pod affinity and a cache-aware auto router

- **Session affinity** (router pre-call check): pins a conversation to the deployment that
  served its first request; recommended when deployments do not share provider-side caches.
- **Cache-aware routing to Kubernetes pods**: a session id selects the same inference pod so a
  self-hosted server (vLLM, SGLang) reuses its prefix cache. LiteLLM states it "does not
  inspect the contents of each pod's cache or guarantee a cache hit".
- **Auto Router with prompt caching**: picks a model tier per request and claims that routing
  plus caching beats caching alone on one fixed model. Vendor reported, unverified by us:
  68.7% cheaper on a WildChat-1M simulation (30,769 conversations), 46% on DevGPT (1,011),
  37.4% on real agent traces (95 sessions, 8,174 calls), 44 to 50% on TwinRouterBench; of 4,684
  switch-backs to an earlier model, 97.4% found the cache warm at a 5 minute TTL, 99.3% at 1 h.
  Its own advice: pin a session when history holds provider state (thinking blocks, tool
  formats) or when the user would notice a change of model.

## Portkey: response cache, simple and semantic

- Caches whole responses: `simple` (exact input) or `semantic` (similar requests, needs a
  vector store), with `max_age` as TTL. A response cache, like GPTCache, not a prefix cache.

## What CAMA must show beyond them

- Gateways decide **where** a request goes (provider, deployment, pod, model tier) and keep the
  cache warm by affinity. None of the pages read changes **what** the prompt contains so that
  it stays cacheable. CAMA's measured gains came from that: memory frozen per epoch, page-ins
  at the tail, extraction scheduling (`docs/measured/long-session.md`, `recall.md`).
- An orchestrator claim must therefore be stated as prompt shaping plus routing, measured
  against LiteLLM's Auto Router or OpenRouter sticky routing as baselines, not against no
  routing at all.
- Their idle limits (10 minutes for OpenRouter stickiness, 5 minutes and 1 hour TTLs in
  LiteLLM's accounting) are inputs CAMA's cost model must take, as the owner noted.

## Sources (accessed 2026-10-24)

- https://openrouter.ai/docs/guides/best-practices/prompt-caching.md (section "Provider Sticky Routing")
- https://docs.litellm.ai/docs/routing (section "Session Affinity (Sticky Sessions)")
- https://docs.litellm.ai/docs/proxy/cache_aware_routing
- https://docs.litellm.ai/docs/auto_router/prompt_caching
- https://docs.portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic.md
- Not read: LiteLLM's blog posts on cache-aware routing (listed in its sitemap).
