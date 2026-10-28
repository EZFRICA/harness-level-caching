---
id: 0008
status: accepted
evidence: g35lite-onnx-agentic-cama-v1, g35lite-onnx-agentic-cama-v2, g35lite-onnx-agentic-cama-v3, g35lite-onnx-agentic-nopc-v1
held_by: tests/test_cost.py::test_the_prefix_cache_mode_follows_the_model_capability, tests/test_cost.py::test_the_budget_guard_refuses_a_call_before_it_is_made
---
# 0008: The prefix cache mode follows the model, and explicit caching is a cost for latency trade

Decided: 2026-10-23, proposed by Claude, within the owner's plan (cost and gain, prior art).

## Context
The prior art review found that Gemini 4 Argon, the owner's production model, does not offer
explicit caching and caches implicitly only from 4096 tokens. On gemini-3.5-flash-lite the
agentic arms measured the explicit cache cutting the hot path cost by about 60% against its
ablation, while turns took about 2.4 to 2.7 s against 1.8 s without it (`docs/measured/agentic.md`).

## Decision
- `HC_PREFIX_CACHE=auto` (default) reads the model's capability in `pricing.toml`: explicit
  where offered, implicit otherwise (stable order kept, no cache created, provider counts
  recorded). `explicit` on a model without it is refused, `none` is the ablation.
- A paid model without a price is refused unless `HC_ALLOW_UNPRICED=1`: the budget guard
  (`HC_MAX_USD`, checked before each call, storage charged up front for the full TTL) could
  not hold otherwise.
- The explicit cache is reported as a trade: cost down, latency up on this provider. Which side
  matters is a deployment choice, not a default the research can settle.

## Consequences
- Argon runs need a price read from the official page before they can run capped.
- An implicit-only model rewards a larger stable prefix (above 4096 tokens): to be measured.

## Alternatives set aside
- Always explicit: impossible on Argon and slower on flash-lite.
- Padding the prefix to reach the implicit minimum: paying input tokens to earn a discount nobody
  guarantees.
