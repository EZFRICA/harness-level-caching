---
id: 0010
status: accepted
evidence: echo-onnx-s1-zipf-cama-v1, echo-onnx-s1-zipf-cama-stable-v1, echo-onnx-s1-zipf-topk-v1, echo-onnx-s1-zipf-topk-stable-v1
held_by: tests/test_harness_caches.py::test_a_prefix_cache_is_created_only_once_the_prefix_was_sent_unchanged
---
# 0010: A provider prefix cache is created only for a prefix expected to be sent again

Decided: 2026-10-23, proposed by Claude after the 200 turn profile, within the owner's plan.

## Context
Over 200 turns where memory changed every turn, the cold path created a cache after every
turn (200 creations), each stale by the next turn; per query prefixes (topk) gave 198
creations for 2 hits. At the official prices a creation pays back only if it is read again.

## Decision
`HC_CACHE_ADMISSION=stable` (default): a prefix is cached when the turns that already sent it,
plus one when extract's rendered set is unchanged since the last turn, reach two. `always` keeps
the previous behaviour and is the ablation.

## Consequences
- topk: 13 creations instead of 198, projected cost 0.122 instead of 0.154 USD on the profile.
- extract with memory rewritten every turn creates nothing and pays full input, still cheaper in
  total than creating every turn (0.182 against 0.194 USD, projected). With a model that writes
  memory rarely, the cache is created from the second turn of each stable stretch.
- The echo extractor writes every turn, which is not what a real model does: the real effect
  is measured on Gemini separately.

## Alternatives set aside
- A learned reuse predictor: more machinery than two hundred turns can calibrate.
