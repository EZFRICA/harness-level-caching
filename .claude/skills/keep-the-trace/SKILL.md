---
name: keep-the-trace
description: How the study records what happens (the hash-chained ledger of every action, run, decision, measurement, fault and review; journal pages rendered from it; decisions with evidence front matter; figures generated from frozen runs). Use after every action that changes the project or its data, when a choice between alternatives is made, and when a run must become a figure.
when_to_use: Just finished an edit, a run, a review or a decision; the owner asks what was done; a document seems out of date.
---

# Keeping the trace

One source per kind of fact, everything else is generated from it.

| fact | source | generated view | checked by |
|---|---|---|---|
| what was done | `docs/ledger.jsonl` (append only, each event hashes the previous line) | `docs/journal/YYYY-MM-DD*.md` | `uv run harness-cache ledger check`, `tests/test_trace.py` |
| why | `docs/decisions/NNNN-*.md` with `evidence:` and `held_by:` | none | `uv run harness-cache decisions` |
| what was measured | `docs/measured/runs/<run>.json` and `figures.toml` | `docs/measured/*.md` | `uv run harness-cache figures check` |

## The ledger

After each action, before the next:

```bash
uv run harness-cache ledger add --kind <action|run|decision|measurement|fault|review> \
  --title "<one line>" --why "<the request or defect>" --files a.py b.py \
  --verified "<tests, mutants>" --live "<what ran, numbers>" --runs <ids> --result <done|partial|failed>
```

The ledger refuses an em or en dash and anything shaped like a key. Never edit
`docs/ledger.jsonl` or a journal page by hand: the chain breaks and the test says where. A
correction is a new event that names the one it corrects.

## Decisions

Front matter first (`id`, `status`, `evidence`, `held_by`), then Context with the numbers of
the time, Decision, Consequences (what holds it), Alternatives set aside. A reversal is a new
decision; the old one gets `status: replaced by NNNN`. A decision with empty evidence says in
its Consequences that it rests on tests only.

## Figures

Never type a measured number into a document. Freeze the run, declare the figure, generate
(`run-an-experiment`). A new measurement is a new run and a new row, beside the old one.
