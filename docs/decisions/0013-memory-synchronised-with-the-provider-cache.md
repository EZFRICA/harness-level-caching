---
id: 0013
status: proposed
evidence: nebius-dsv4flash-onnx-s1-zipf-naive-v1, nebius-dsv4flash-onnx-s1-zipf-cama-v1, nebius-dsv4flash-onnx-s1-zipf-cama-h7e20-v1, nebius-dsv4flash-onnx-s1-zipf-cama-h8e20-v1, nebius-dsv4flash-onnx-memory-cama-v1, nebius-dsv4flash-onnx-memory-cama-v2, nebius-dsv4flash-onnx-memory-cama-replace-every2-v1, nebius-dsv4flash-onnx-memory-cama-replace-every2-v2, nebius-dsv4flash-onnx-memory-cama-h8e10-v1, nebius-dsv4flash-onnx-memory-cama-h8e10-v2
held_by: tests/test_hypotheses.py::test_with_epoch_refresh_memory_changes_wait_for_the_window_drop, tests/test_hypotheses.py::test_with_tail_page_ins_the_cached_head_stays_and_the_block_goes_last, tests/test_extract_replace.py::test_with_the_switch_on_a_correction_leaves_no_stale_line
---
# 0013: The rendered memory changes only when the window drops

Proposed: 2026-10-24, by Claude, from the hypothesis series of that day. Awaits the owner.

## Context
On a provider whose cache serves any repeated prefix (Nebius, and by its docs Gemini 4
Argon), what pays is a prompt that only grows at its end. Over 200 turns, naive was 97.7%
cached and cost half of cama, which was 53% cached: the memory sits before the window, so
each memory change (57 of 199 turns) and each page-in re-billed everything after it
(`docs/measured/long-session.md`). Separately, append-only extraction kept a corrected fact
next to its stale value, and cama recalled the stale one (`docs/measured/recall.md`).

## Decision (proposed as the cama defaults on implicit-cache providers)
- `CAMA_MEMORY_REFRESH=epoch`, `CAMA_WINDOW_EPOCH=E`: the memory is rendered from a copy
  refreshed only when the window drops, every E turns; between drops the window grows, so the
  prompt is append-only. Facts learnt meanwhile are read in the window.
- `CAMA_PAGE_IN_TAIL=on`: a block a query pages in is sent in front of the question, at the end
  of the prompt, and kept in the window as sent; it joins the memory at the next drop.
- `CAMA_EXTRACT_REPLACE=on`: a corrected fact replaces its line.
- `CAMA_EXTRACT_EVERY=2`: until batched extraction is shown not to drop corrections (one
  correction was lost in a batch of 8 exchanges).

## Consequences
- 200 turns: cama 0.0236 USD against 0.0435 before and 0.0219 for naive, with a hot path
  cheaper than naive's; recall 13 of 13 in every run with these switches.
- The prompt is bounded by the memory plus E + window exchanges, not by the session length.
- Decision 0002 (render order apart from recency) still holds inside the head; this adds when
  the head may change.
- The explicit prefix cache (0005, 0010) is not involved: these runs were implicit only.

## Alternatives set aside
- A growing window with memory refreshed every turn: 60% cached, no saving.
- Semantic or per-query selection (topk): lower recall (8 and 10 of 13).
