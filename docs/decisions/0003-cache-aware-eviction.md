---
id: 0003
status: accepted
evidence:
held_by: tests/test_core.py::test_cache_aware_eviction_prefers_latest_rendered_among_lru_window, tests/test_core.py::test_pure_lru_when_window_is_one
---
# 0003: Eviction weighs the cached prefix it would invalidate

Decided: 2026-10-23, proposed by Claude, accepted by the owner.

## Context
Evicting a block early in the render order invalidates every cached token after it. Plain LRU
ignores that position.

## Decision
Among the `victim_window` least recently used unpinned blocks, evict the one rendered last.
`victim_window=1` is plain LRU and is kept as the ablation.

## Consequences
- The policy has no measured effect yet: no run compared window 3 with window 1. Until one
  does, this decision rests on its tests only, and says so.
- A block both old and early in the order can outlive a slightly more recent late block.

## Alternatives set aside
- Plain LRU: kept as the baseline, not the default.
- Cost-based eviction with a learned reuse model (as in recent multi-agent caching work): more
  parameters than the data available can support today.
