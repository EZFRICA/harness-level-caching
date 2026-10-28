# Prior art: compaction and memory for long sessions

Read on 2026-10-25 in the arXiv PDFs and official docs listed at the end. Written after the
decisive test (`docs/measured/decisive.md`), where a history folded into a summary beat extract
with decision 0013 over 500 turns. It maps what exists against extract's two measured defects:
(1) memory extraction on every turn costs more than an occasional summary; (2) facts the
summary kept were missed by extract. Correction after checking both runs' stores: those facts
were extracted and paged out to L3, and the page fault did not bring them back. Defect (2) is
a recall failure of a mechanism extract inherits from APU and Urithi, which already keep every
paged-out block in L3 and page it back on relevance (apu-urithi apu/mmu/dll.py), and cite
retrieved units by id with a code check (learner/tutor/answer.py). The lead: compaction that
follows each provider's cache rules.

## Defect 2, missed recall: what others add to page-in

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
  is consistent with extract (selection) losing to compact (generation). Its case study found
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

## Cache engineering below the prompt: AKV (Tsinghua, arXiv 2610.28577v1, 20 Oct 2026)

- Trimming or summarising changes the prefix and forces a re-prefill; AKV lets the agent act on
  its own KV cache instead: `drop` excludes spans from attention, `repos` re-indexes positions,
  without changing the text. A serving engine on SGLang v0.5.20 keeps a stateless API and
  handles that identical tokens can carry different cache states (Sections 3, 4, A).
- Tool-response rolling drop (keep the latest 12 tool responses) against threshold summaries
  at 96K or 192K and at 48K tokens: average score +1.7 points, average cost -27.7%, on
  GPT-OSS-120B, MiniMax-M2.7, MiroThinker-1.7-Mini over BrowseComp-Plus, xbench-DS, GAIA-Text,
  SWE-Verified (abstract, Section 5). KV cache reads are 82.4% of the API cost (Figure 13).
- It needs a serving engine it controls; a provider API exposes no such operation. Same
  family as Leyline (`docs/prior-art-serving.md`), which it does not cite; it cites TokenPilot.

## What this changes for extract

- Compaction that respects the prefix cache is TokenPilot's contribution, published June 2026;
  extract's decision 0013 is a later, narrower instance and cannot claim it.
- Keeping everything and recalling it is APU and Urithi's design already; ARC's addition is an
  explicit recall by id that the agent requests, beside similarity page-in. The defect to fix
  is the page fault's miss (thresholds uncalibrated, `docs/limits.md`). For cost: offline
  consolidation (LightMem) or LLM-free eviction (CWL).
- Not found in the papers read: compaction timed by measured provider cache life and replica
  routing (`docs/measured/probes/`). TokenPilot models a cache discount, not its expiry.
- Any next experiment must use summary compaction and TokenPilot as baselines, not naive.

## Sources (accessed 2026-10-25)

- https://arxiv.org/abs/2607.25066 (ARC), https://arxiv.org/abs/2606.11213 (CWL)
- https://arxiv.org/abs/2608.01326 (Context Compaction Theory)
- https://arxiv.org/abs/2510.18866 (LightMem), https://arxiv.org/abs/2504.13171 (Sleep-time)
- https://arxiv.org/abs/2605.23296 (Parallel compaction), https://arxiv.org/abs/2606.17016 (TokenPilot)
- https://arxiv.org/abs/2610.28577 (AKV, Cache Engineering for Agents), read 2026-10-27
- https://platform.claude.com/docs/en/build-with-claude/compaction and .../compaction-threshold
- https://adk.dev/context/compaction/
- Seen as titles only, not read: ACON 2510.00615, Context-Folding 2510.11967, AgentFold
  2510.24699, ReSum 2509.13313, Self-Compacting Agents 2606.23525, Active Context Compression
  2601.07190, Self-GC 2607.00692, ICLR compression 2609.29875, Recursive LMs 2512.24601.
