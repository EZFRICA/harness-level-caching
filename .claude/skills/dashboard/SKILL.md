---
name: dashboard
description: The contract of CAMA's dashboard (read only FastAPI app on port 8797, no external asset, every figure computed by cama.report and cama.cost, plain language for non specialists, verdicts only where evidence supports them, absent values shown as absent) and its look. Use when changing src/cama/dash, adding a view, a chart, a KPI or a verdict, or explaining a metric on screen.
when_to_use: Editing app.py, data.py, app.js or style.css; adding a comparison or a cost view; styling.
---

# The dashboard

## Contract

- **Read only.** It reads run directories, frozen runs (`/docs/measured`), the ledger, live
  stores (query only) and the coordinator's view; it never writes and never calls a model.
- **One computation.** Figures come from `cama.report.metrics` and `cama.cost.run_cost`, the
  code behind the reports and the generated pages. The browser formats; it computes no money.
- **Offline.** No CDN, no external font or script (`tests/test_dash.py` checks it).
- **Absent is not zero.** An unpriced run, a frozen run without stores, an unreachable
  coordinator: each says so in a dashed note, never as 0.
- **Verdicts are earned.** `verdicts()` states a latency difference only when the 95% intervals
  are apart, labels a cost difference "one run each", warns when scenarios differ, and always
  says that answer quality is not measured.

## Views

Overview (plain words, the turn in five steps, the four levels, a glossary), Live run, Compare
A/B (verdicts, side by side, setup differences, overlaid curves), Costs (posts, provenance of
prices, assumptions), Cluster (workers, leases, events), Stress (load and fault runs), Trace
(the ledger).

## Look

One colour per meaning, used everywhere: H1 routing cyan, H2 embeddings green, H3 tools amber,
H4 prefix violet; hot path coral, cold path blue; memory kinds indigo, teal, orange. Light and
dark themes from the system, switchable. Numbers in tabular figures, data in mono. Interface
text in English, short, and every metric with a one line explanation on hover.
