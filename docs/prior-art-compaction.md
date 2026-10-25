# Prior art: compaction and memory for long sessions

Read on 2026-10-25 in the arXiv PDFs and official docs listed at the end. Written after the
decisive test (`docs/measured/decisive.md`), where a history folded into a summary beat cama
with decision 0013 over 500 turns. It maps what exists against cama's two measured defects:
(1) memory extraction on every turn costs more than an occasional summary; (2) a bounded
memory loses facts that a summary keeps. And against the open lead: compaction that follows
each provider's cache rules.

## Defect 2, lost facts: keep everything, recall by address

- **ARC, Addressable Recall Compaction** (arXiv 2607.25066v1). Tool observations go to an
  append-only, ID-addressable log; compaction replaces old ones with citations the agent can
  fetch back by ID. Needle in a haystack: 99.40% exact answers against 88.12% for the best
  baseline (abstract, Table 2, Qwen3-8B 16k and Qwen3-32B 32k); LongBench-v2 hard 29.97%
  against 28.25% (Table 1). Baselines include LLM summarisation.
- **CWL, Context Window Lifecycle** (arXiv 2606.11213v1). The agent annotates typed,
  dependency-linked episodes; a deterministic, LLM-free policy evicts the most recoverable
  content first and keeps user turns. One session ran 89 Terminal Bench tasks over 80 million
  tokens "with no measurable degradation" against isolated sessions (abstract, Section 6,
  GPT-5.4). Argues summaries are lossy, blocking, and can hallucinate.
- **Context Compaction Theory** (arXiv 2608.01326v1). Compaction as one-way communication:
  generating a summary can need strictly less budget than selecting a subset (a proof), which
  is consistent with cama (selection) losing to compact (generation). Its case study found
  Anthropic's compaction endpoint near random guessing on set membership over 15,000 items
  (Section 1, Appendix A): summaries lose exact items too.

## Defect 1, cost: take memory work off the turn, or off the model

- **LightMem** (arXiv 2510.18866, ICLR 2026). Filter, group by topic, summarise, and
  consolidate long-term memory offline ("sleep-time update"). Abstract: up to 38x fewer tokens
  and 30x fewer API calls on LongMemEval, with QA accuracy up to 7.7% higher.
- **Sleep-time compute** (Letta, arXiv 2504.13171). Compute on the context while idle:
  about 5x less test-time compute at equal accuracy on two stateful reasoning tasks (abstract).
- **Parallel compaction** (arXiv 2605.23296v1). Summarisation stalls the agent "for tens of
  seconds" and its output volume is unpredictable; compacting blocks in parallel gives
  controllable volume and shorter wall time (abstract).
- **Products**: Anthropic compaction on demand, in the background, keeping recent turns, or at
  a token threshold (minimum 50,000, default 150,000); Google ADK compaction by turn interval
  with overlap, or by token threshold with recent events kept raw. Both are summaries.

## The open lead, cache-aware compaction: already explored

- **TokenPilot** (arXiv 2606.17016v2). Pruning and eviction mutate the prompt and break the
  provider prefix cache; TokenPilot stabilises the prefix at ingestion and evicts segments in
  batches only when their task relevance has expired. GPT-5.4-mini, against a Summary baseline
  among others (Tables 1, 2): PinchBench continuous 2.79 USD against 7.12 at a score of 81.3
  against 78.4; Claw-Eval continuous 10.58 USD against 16.59 at 60.8 against 61.6.
- Gateways already keep a session on the warm cache (`docs/prior-art-gateways.md`).

## What this changes for cama

- Compaction that respects the prefix cache is TokenPilot's contribution, published June 2026;
  cama's decision 0013 is a later, narrower instance and cannot claim it.
- The two defects have published answers: lossless archive with addressed recall (ARC) for
  lost facts, offline consolidation (LightMem) or LLM-free eviction (CWL) for cost.
- Not found in the papers read: compaction timed by measured provider cache life and replica
  routing (`docs/measured/probes/`). TokenPilot models a cache discount, not its expiry.
- Any next experiment must use summary compaction and TokenPilot as baselines, not naive.

## Sources (accessed 2026-10-25)

- https://arxiv.org/abs/2607.25066 (ARC), https://arxiv.org/abs/2606.11213 (CWL)
- https://arxiv.org/abs/2608.01326 (Context Compaction Theory)
- https://arxiv.org/abs/2510.18866 (LightMem), https://arxiv.org/abs/2504.13171 (Sleep-time)
- https://arxiv.org/abs/2605.23296 (Parallel compaction), https://arxiv.org/abs/2606.17016 (TokenPilot)
- https://platform.claude.com/docs/en/build-with-claude/compaction and .../compaction-threshold
- https://adk.dev/context/compaction/
- Seen as titles only, not read: ACON 2510.00615, Context-Folding 2510.11967, AgentFold
  2510.24699, ReSum 2509.13313, Self-Compacting Agents 2606.23525, Active Context Compression
  2601.07190, Self-GC 2607.00692, ICLR compression 2609.29875, Recursive LMs 2512.24601.
