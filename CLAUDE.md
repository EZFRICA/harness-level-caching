# Working in this repository

CAMA (Cache-Aware Memory Architecture) is the cache of the agent harness: everything around
the model that an agent pays for on every step (routing, embeddings, tool results, the prompt
prefix), organised so that what is stable stays reusable. It continues the owner's Agent OS
line: APU (memory hierarchy), CLE (lifecycle, measurement discipline), Urithi (guarantees,
metering). The research question is **what an agent gains, in cost and latency, from a
harness-level cache, and where it breaks**. The deliverable is architectures with their
trade-offs and their limits, measured; never a winner by assertion.

Read `.claude/skills/house-rules/SKILL.md` before anything else.

| task | skill |
|---|---|
| any command, edit, document, commit, paid call, report to the owner | `house-rules` |
| producing, comparing or publishing any number | `run-an-experiment` |
| adding or changing a cache level, key, TTL or invalidation | `add-a-cache-level` |
| finishing or reviewing a change | `review-a-change` |
| recording what was done, decided or measured | `keep-the-trace` |
| positioning against existing systems, citing a paper | `prior-art` |
| load, faults, breaking points, RPO and RTO | `run-a-stress` |
| leases, epochs, replication, recovery, shared caches | `cluster-contract` |
| the dashboard, its views, verdicts and look | `dashboard` |

Agents in `.claude/agents/`: `change-reviewer` (read only review), `experiment-runner`
(measured figures), `chaos-tester` (stress and faults), `prior-art-researcher` (verified
sources).

## Where things are

- Design: `docs/architecture.md`. Why: `docs/decisions/` (with evidence front matter). What
  was done: `docs/ledger.jsonl`, rendered to `docs/journal/`. What was measured:
  `docs/measured/` (generated). Existing systems: `docs/prior-art*.md`.
- Code: `src/harness_cache/` (core is network free; providers live in `src/harness_cache/adapters/`).
- Runs: `runs/<run_id>/` (manifest, turns, tasks, log), gitignored.

## True for every change

- **The hot path never waits for the cold path.** Extraction, cache creation and
  replication are scheduled, never awaited inside a turn.
- **A number carries its pinning key and the command that reproduces it**, and a new
  measurement is added beside the old one, never over it.
- **No test needs a key or the network.** Live runs are separate and budget capped.
- **A guarantee ships with a test that a mutant would make fail** (`uv run mutmut run`).
- **Every action is a ledger event** (`uv run harness-cache ledger add ...`) before the next one
  starts; `docs/journal/` is generated from the ledger, never edited.
- **Measured numbers are generated** from frozen runs (`uv run harness-cache figures write`), never
  typed into a document.

## The verification series

```bash
uv run pytest -q                              # offline; includes trace and writing rules
uv run mutmut run                             # survivors are invariants nobody holds
uv run harness-cache ledger check && uv run harness-cache figures check && uv run harness-cache decisions
docker compose up -d --build                  # then a scenario run, see run-an-experiment
```

Green offline is not done: each change is also run live in the container and reported with
its numbers, or reported as not run live.
