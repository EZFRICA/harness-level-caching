---
id: 0004
status: accepted
evidence: gemini-cama-v1, gemini-onnx-cama-v1
held_by: tests/test_core.py::test_store_rejects_non_unit_vectors_and_foreign_spaces
---
# 0004: Query embeddings are computed locally, in both online and offline profiles

Decided: 2026-10-23, by the owner (the local embedder already used in Akili and Urithi).

## Context
With Gemini embeddings, the routing cost of a turn was dominated by the embedding round trip
(`docs/measured/embeddings.md`).

## Decision
`paraphrase-multilingual-MiniLM-L12-v2`, quantised ONNX through fastembed, read from a local
cache mounted read only, never downloaded at runtime, threads sized to the container's CPU
quota.

## Consequences
- Memory cost per turn falls by an order of magnitude (see the figure).
- The process holds the model: peak RSS rises to about 700 MB, which a 2 GB device cannot
  afford; recorded as a limit to stress (S4).
- Thresholds are per embedding space; the current ones are in sample.

## Alternatives set aside
- Gemini embeddings: network on the hot path.
- EmbeddingGemma 2 ONNX: available as `HC_EMBEDDER=gemma` with its own prefixes and in sample
  thresholds; its search finds more (`docs/measured/verbatim-search-1500.md`) but costs about
  60 ms per uncached query and most of a 1 GB container. Not the default before a paid run.
