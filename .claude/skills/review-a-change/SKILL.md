---
name: review-a-change
description: How a change to the code is reviewed before it is called done (which invariants it touches, the test and the mutant that hold each, the mutation run, the live run, the trace). Use after any change under src/harness_cache, before reporting a step as finished, and when the owner asks whether something is solid.
when_to_use: Finishing a feature or a fix; reviewing someone else's change; deciding whether a guarantee is really held.
---

# Reviewing a change

A passing suite says the code does something; it does not say a test would notice if it
stopped. The study answers that with mutation testing (mutmut, configured in `pyproject.toml`).

## 1. Name the invariants touched

From `docs/architecture.md`, "Invariants held by tests", plus any the change introduces:
bounds of the active set, render order apart from recency, cache-aware eviction, hot path
never creating a cache or awaiting a cold task, unit-norm vectors and one embedding space per
store, the ledger chain, generated pages equal to their regeneration.

## 2. Each invariant has a test that would fail without it

- Named as the sentence it proves; a docstring that says what was measured or why.
- Over many inputs when it is a rule over sequences: a property test (hypothesis).
- No key and no network in any test; providers are faked.

## 3. Mutation

```bash
uv run mutmut run            # mutates src/harness_cache/memory and src/harness_cache/store, runs the suite per mutant
uv run mutmut results        # survivors are invariants nobody holds
```

A surviving mutant inside the changed code is a blocking finding: add the test or explain in
the review why the mutant is equivalent.

## 4. Live

The container rebuilt with the change, a scenario run that exercises it, the run id and the
numbers. Not run live is a finding, reported as such.

## 5. Trace

A ledger event for the change; a decision record when an alternative was set aside; figures
regenerated if a frozen run or `figures.toml` changed (`uv run harness-cache figures check`).

## Report

Blocking findings first (file:line, what breaks, the fix), then the rest; what was run and
its result. In French to the owner.
