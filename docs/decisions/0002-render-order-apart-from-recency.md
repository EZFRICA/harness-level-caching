---
id: 0002
status: accepted
evidence: g35lite-onnx-cama-cache-v1
held_by: tests/test_core.py::test_render_order_ignores_access_recency, tests/test_core.py::test_modified_block_moves_to_end_of_its_class
---
# 0002: The prompt order of memory blocks never follows access recency

Decided: 2026-10-23, proposed by Claude, accepted by the owner.

## Context
The APU's Move-to-Front placed the last block read at the head of the list and injected the
list in that order. Every read therefore reshuffled the prompt, and a provider or KV cache of
the prefix could not survive two turns.

## Decision
The active set keeps two orders. Recency (a doubly linked list) decides eviction only. The
render order is (stability kind, last modification, id): fundamental, then project, then
temporary blocks; a modified block moves to the end of its kind.

## Consequences
- A read changes nothing in the prompt; a write invalidates only what follows the block.
- Measured: with the explicit cache, 5 creations served 15 of 23 turns
  (`g35lite-onnx-cama-cache-v1`, `docs/measured/prefix-cache.md`).
- The model no longer sees the most recently used block first; the effect on answer quality
  is not measured yet.

## Alternatives set aside
- Recency order (the APU): defeats prefix caching.
- Per query top k selection: measured as the `topk` mode, cheaper without a cache, but its
  prefix changes with every query.
