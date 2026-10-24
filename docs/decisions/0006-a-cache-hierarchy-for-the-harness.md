---
id: 0006
status: accepted
evidence: g35lite-onnx-cama-cache-v1
held_by:
---
# 0006: CAMA caches the agent harness, not only the model: levels H1 to H4

Decided: 2026-10-23, by the owner (tool results and embeddings chosen for this iteration).

## Context
An agent pays on every step for more than the model call: routing, embeddings, tool calls,
and the prompt prefix it sends again and again. Existing systems cache one of these at a time,
mostly inside the inference server.

## Decision
One hierarchy with one invalidation story, in the harness:
- H1 routing: lexical, then active set embeddings, then L3 (exists);
- H2 embeddings: content addressed by text and embedding space;
- H3 tool results: keyed by tool and normalised arguments, with a class per tool (pure,
  time to live, never for side effects);
- H4 provider prefix (exists, decision 0005).
Each level has a hit rate, a cost line and an ablation switch.

## Consequences
- Agentic turns (several model and tool steps) are needed to measure H3 and the multiplied
  effect of H4: a new scenario.
- Plan caching and semantic response caching are out of scope for now.

## Alternatives set aside
- A semantic response cache (GPTCache style): returns whole answers, risky for an agent with
  changing memory, and already well covered.
