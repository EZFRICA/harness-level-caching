---
id: 0012
status: proposed
evidence:
held_by: tests/test_web_search.py::test_an_h3_hit_returns_exactly_what_the_miss_returned, tests/test_web_search.py::test_a_recorded_query_is_replayed_identically_without_a_live_call, tests/test_web_search.py::test_the_web_tool_is_offered_only_when_switched_on
---
# 0012: H3 is measured on a real web search, made repeatable by a tape

Proposed: 2026-10-24, by Claude, within the owner's plan (Tavily track). Awaits the owner.

## Context
Every H3 figure so far used simulated tools with fixed latencies. The closest prior art
(hierarchical agentic caching, MDPI 2026, `docs/prior-art-tools.md`) also measured only
synthetic Zipfian workloads, and its plain LRU baseline matched its full system. A real tool
answers differently from one call to the next, so two arms would not see the same answers.

## Decision
- A `web_search` tool (Tavily, basic depth, 5 results trimmed to 500 characters each) joins
  H3 behind `CAMA_WEB_SEARCH=on`. Off by default: a new tool changes the tool declarations,
  hence the cached prefix of every existing scenario.
- Class `ttl`, 3600 s. Key: the query lower-cased with spaces collapsed. Nothing else is
  normalised: two phrasings of one question are two keys (exact match, unlike Cortex).
- A tape (`CAMA_WEB_TAPE`, JSONL) records the first live answer to a query with its measured
  latency; every later sighting, in any run reading the tape, replays it with that latency.
  `CAMA_WEB_MODE=replay` never calls live. Live calls are capped (`CAMA_WEB_MAX_CALLS`).
- A real tool is not slept by the runner (`simulated=False`); it takes its own time.

## Consequences
- Arms that share a tape see identical answers, so an H3 hit is compared with exactly what a
  miss would have returned, and the time a hit saves is the time a real call took.
- The tape is data: a figure built on it must keep it beside the frozen runs.
- The model writes the queries, so exact repeats depend on the model, not on the scenario.

## Alternatives set aside
- Calling live in every arm: answers and latencies differ between arms, and credits are spent
  on the same queries again.
- Semantic matching of queries (Cortex): a judge model on the hot path; to be measured apart.
