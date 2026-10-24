---
id: 0005
status: accepted
evidence: g35lite-onnx-cama-v2, g35lite-onnx-cama-cache-v1
held_by: tests/test_core.py::test_end_to_end_scenario_echo
---
# 0005: The provider prefix cache is explicit, created on the cold path, looked up on the hot path

Decided: 2026-10-23, by the owner after the probe.

## Context
No run had any cached token. A probe on 2026-10-23 found Gemini implicit caching absent below
about 5000 tokens and best effort above it, while explicit caching worked from 1024 tokens
(the API's own minimum) on gemini-3.5-flash-lite and gemini-3.8-flash.

## Decision
The cacheable part (instructions and the active set in render order) is cached explicitly.
The hot path only looks up a ready cache for the exact prefix hash. The cold path creates it
after extraction, deletes the previous one, and marks prefixes under the minimum unusable.

## Consequences
- 58.3% of prompt tokens served from cache on the scenario, 15 of 23 turns.
- Creation (about 1.6 s) is never paid by a waiting user; a page fault inside a turn still
  causes one miss.
- Creation and storage are billed; the gain in money awaits the cost model.

## Alternatives set aside
- Relying on implicit caching: absent at our prompt sizes.
- Creating the cache inside the turn: the user would wait for it.
