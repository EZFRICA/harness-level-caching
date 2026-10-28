---
name: run-an-experiment
description: How a CAMA question becomes a measured figure (the question, the arms and their ablation, fresh stores, the container limits, the budget, the report, freezing the runs, declaring the figure, the ledger event). Use whenever a number is to be produced, compared, published or claimed, including a claim that one mode, model or setting is better.
when_to_use: Comparing modes, models, embedders, cache settings or limits; answering "is it faster or cheaper"; adding to docs/measured; re-running an old figure.
---

# Running an experiment

A figure in CAMA is never typed: it is generated from frozen runs. So the path from a
question to a published number is always the same.

## 1. Write the question and the arms

One sentence a non specialist understands ("how many prompt tokens does the explicit prefix
cache save on 23 turns?"). Then the arms: the policy under test, its **ablation** (the same
run with the mechanism switched off: `HC_PREFIX_CACHE=none`, `HC_VICTIM_WINDOW=1`, ...),
and a baseline mode (`naive`, `topk`) when relevant. Everything else identical: scenario,
seed blocks, limits, model.

## 2. Run each arm on a fresh store

```bash
HC_LLM=gemini HC_EMBEDDER=onnx docker compose up -d
docker compose exec hc harness-cache run /scenarios/<scenario>.jsonl --fresh --mode <mode> --run-id <name>
docker compose exec hc harness-cache report /runs/<name> --skip-first 1
```

Name runs so they sort and read: `<model>-<embedder>-<mode>-<variant>-v<n>`. Real models run
small and under `HC_MAX_USD`; repeat an arm when the CIs overlap rather than concluding.

## 3. Freeze, declare, generate

```bash
uv run harness-cache figures snapshot runs/<name> ...     # docs/measured/runs/<name>.json
# add a [[figure]] to docs/measured/figures.toml: question, runs, columns, reproduce, reading, limits
uv run harness-cache figures write
```

`reading` states only what the table shows. `limits` always says: how many runs, synthetic or
real scenario, echo or real model, in-sample thresholds, quality not measured.

## 4. Say what it means, carefully

- A difference inside overlapping CIs is "no measured difference".
- Estimated tokens (chars/4) and provider tokens are never mixed in one claim.
- A gain on the hot path that is paid back on the cold path is not a gain: look at both.
- An embedding threshold read on the scenario it was tuned on is in sample.

## 5. Record it

```bash
uv run harness-cache ledger add --kind measurement --title "..." --why "<the question>" \
  --runs <names> --live "<one line with the key numbers>" --result done
```

If the result changes a design choice, write or replace a decision citing the runs.
